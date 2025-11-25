# 修复导入路径：使用本地MineStudio
import sys
from pathlib import Path
import ast
import json

# 添加当前目录到路径（修复相对导入问题）
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 添加MineStudio到路径
minestudio_path = Path(__file__).parent.parent / "MineStudio"
if str(minestudio_path) not in sys.path:
    sys.path.insert(0, str(minestudio_path))

# 现在可以导入MineStudio组件
from minestudio.simulator.entry import MinecraftSim
from minestudio.simulator.callbacks import (
    SpeedTestCallback,
    RecordCallback,
    SummonMobsCallback,
    MaskActionsCallback,
    RewardsCallback,
    CommandsCallback,
    TaskCallback,
    FastResetCallback,
    JudgeResetCallback,
    InitInventoryCallback
)

# 创建ToolBase的简单替代品（如果需要）
class ToolBase:
    """简单的ToolBase替代品"""
    def __init__(self, name=None, **kwargs):
        self.name = name

from utils.action_converter import ActionFromLLMConverter
from utils.action_mapping import OneActionTokenizer
from utils.sim_callbacks_loader import load_simulator_setup_from_yaml
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image
from rich import print, console
import numpy as np
import logging
import time

rich_console = console.Console()

logger = logging.getLogger("mc_simulator")
logger.setLevel(logging.INFO)

# 确保logs目录存在
import os
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"mc_simulator_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
# Configure file handler (once)
if not logger.handlers:
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S.%f")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Optional console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

logger.info(f"Logging to file: {os.path.abspath(log_file)}")

class MCSimulator(ToolBase):
    name = "MCSimulator"

    def __init__(self, _name=None, _desc=None, _params=None, config=None, **kwargs):
        """初始化 MCSimulator

        Args:
            _name: 工具名称(兼容性参数)
            _desc: 描述(兼容性参数)
            _params: 参数(兼容性参数)
            config: 配置,可以是:
                - 字符串/Path: YAML配置文件路径
                - 字典: 预处理的配置(MVP2模式)
            **kwargs: 其他参数
        """
        super().__init__(name=self.name)
        start_time = time.perf_counter()
        try:
            # 区分两种配置来源:
            # 1. 字符串/Path -> 从 YAML 加载
            # 2. 字典 -> 使用预处理的配置 (MVP2 模式)
            if isinstance(config, dict):
                # MVP2 模式: 使用预处理的配置
                sim_callbacks = config.get("sim_callbacks", [])
                env_overrides = {k: v for k, v in config.items() if k != "sim_callbacks"}
                rich_console.log(f"[mc_simulator] Using pre-processed config (MVP2 mode)")

                # Debug: 检查 RecordCallback 的 record_path
                for cb in sim_callbacks:
                    if hasattr(cb, 'record_path'):
                        rich_console.log(f"[mc_simulator] RecordCallback.record_path = {cb.record_path}")
            else:
                # 传统模式: 从 YAML 加载
                _desc = config
                sim_callbacks, env_overrides = load_simulator_setup_from_yaml(_desc) if _desc else ([], {})
                rich_console.log(f"[mc_simulator] Loaded config from YAML (traditional mode)")

            rich_console.log(f"[mc_simulator] env_overrides={env_overrides}")
            if 'seed' in env_overrides:
                rich_console.log(f"[mc_simulator] seed={env_overrides['seed']}")

            # Ensure default recording callback exists
            if not any(getattr(cb, "__class__", type(None)).__name__ == "RecordCallback" for cb in sim_callbacks):
                sim_callbacks.append(RecordCallback(record_path="./output", fps=30, frame_type="pov"))

            obs_w = int(env_overrides.get('obs_width', env_overrides.get('obs_w', 640)))
            obs_h = int(env_overrides.get('obs_height', env_overrides.get('obs_h', 360)))

            # Support 'resolution' key: [W, H] or {width: W, height: H}
            resolution = env_overrides.get('resolution')
            rich_console.log(f"[mc_simulator] resolution={resolution}")
            try:
                if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
                    obs_w, obs_h = int(resolution[0]), int(resolution[1])
                elif isinstance(resolution, dict):
                    obs_w = int(resolution.get('width', obs_w))
                    obs_h = int(resolution.get('height', obs_h))
            except Exception:
                pass

            rich_console.log(f"[mc_simulator] resolved obs_size=({obs_w},{obs_h})")
            preferred_biome = env_overrides.get('preferred_spawn_biome', 'plains')
            action_type = env_overrides.get('action_type', 'agent')
            timestep_limit = int(env_overrides.get('timestep_limit', 1000))
            rich_console.log(obs_w)
            rich_console.log(obs_h)

            # 添加 FastResetCallback 以加速后续 reset（如果还没有）
            if not any(isinstance(cb, FastResetCallback) for cb in sim_callbacks):
                fast_reset_biomes = env_overrides.get('fast_reset_biomes', ['plains', 'forest', 'mountains'])
                fast_reset_range = int(env_overrides.get('fast_reset_range', 1000))
                fast_reset_time = int(env_overrides.get('fast_reset_time', 1000))
                fast_reset_weather = env_overrides.get('fast_reset_weather', 'clear')
                
                sim_callbacks.append(FastResetCallback(
                    biomes=fast_reset_biomes,
                    random_tp_range=fast_reset_range,
                    start_time=fast_reset_time,
                    start_weather=fast_reset_weather
                ))
                rich_console.log(f"[mc_simulator] Added FastResetCallback: biomes={fast_reset_biomes}, range={fast_reset_range}")

            # 在 __init__ 中创建 MinecraftSim
            self.simulator = MinecraftSim(
                obs_size=(obs_w, obs_h),
                preferred_spawn_biome=preferred_biome,
                action_type=action_type,
                timestep_limit=timestep_limit,
                callbacks=sim_callbacks,
            )

            # Debug verification of action mapping in use
            try:
                rich_console.log(f"[mc_simulator] action_type={self.simulator.action_type}")
                mapper = getattr(self.simulator, "action_mapper", None)
                rich_console.log(f"[mc_simulator] action_mapper={type(mapper).__name__}")
                cam_tbl = getattr(mapper, "CAMERA_IDX_TO_FACTORED", None)
                if cam_tbl is not None:
                    rich_console.log(f"[mc_simulator] camera_table_size={len(cam_tbl)}")
                else:
                    rich_console.log("[mc_simulator] camera_table_size=N/A")
            except Exception as _e:
                rich_console.log(f"[mc_simulator] debug print failed: {_e}")

            self.action_converter = ActionFromLLMConverter(
                map_camera_to_11=True,   # 21×21 → 11×11
                return_numpy=True        # 返回 np.array([int])
            )
            self.action_tokenizer = OneActionTokenizer(tokenizer_type="qwen2_vl")
            self._obs_shape = (obs_h, obs_w, 3)
            self.last_obs = None

        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(f"[mc_simulator] __init__ elapsed={elapsed_ms:.2f} ms")


    # def execute(self, action_input, **kwargs) -> tuple:
    #     """执行动作

    #     Args:
    #         action_input: 支持两种格式
    #             - str: LLM格式 '[{"action": "attack", "yaw": -25.0, "pitch": 8.0}]'
    #             - dict: Agent格式 {'buttons': 5, 'camera': 222}
    #     """
    #     # 判断action格式并转换
    #     if isinstance(action_input, str):
    #         # LLM格式：使用ActionConverter转换
    #         print(f' before convert {type(action_input)=}')
    #         # action = self.action_converter.convert(action_input, self._obs_shape)
    #         action_input = json.loads(action_input)
    #         # 转化完要是一个list
    #         print(f'after convert, {action_input=}, {type(action_input)=}, {type(action_input[0])=}')
    #         action = self.action_tokenizer.decode(action_input)
    #         # LLM action是special token，但是actoin是button和camera
    #         # rich_console.log(f"[DEBUG] LLM action: {action_input}")
    #         rich_console.log(f"[DEBUG] converted to: {action}, {type(action)=}")
    #     elif isinstance(action_input, dict):
    #         # Agent格式：将整数转换为numpy数组
    #         import numpy as np
    #         action = {}
    #         for key, value in action_input.items():
    #             if isinstance(value, int):
    #                 action[key] = np.array(value)
    #             else:
    #                 action[key] = value
    #         rich_console.log(f"[DEBUG] Agent action (dict): {action_input}")
    #         rich_console.log(f"[DEBUG] converted to numpy: {action}")
    #     else:
    #         raise ValueError(f"Unsupported action type: {type(action_input)}")

    #     obs, reward, terminated, truncated, info = self.simulator.step(action)
    #     done =  (terminated or truncated)
    #     self.last_obs = obs  # 更新最后观察状态
    #     return obs, reward, done, info

    def step(self, action_input, **kwargs) -> tuple:
        """执行动作

        Args:
            action_input: 支持两种格式
                - str: LLM格式 '[{"action": "attack", "yaw": -25.0, "pitch": 8.0}]'
                - dict: Agent格式 {'buttons': 5, 'camera': 222}
        """
        # 判断action格式并转换
        if isinstance(action_input, str):
            # LLM格式：使用ActionConverter转换
            print(f' before convert {type(action_input)=}')
            # action = self.action_converter.convert(action_input, self._obs_shape)
            action_input = json.loads(action_input)
            # 转化完要是一个list
            print(f'after convert, {action_input=}, {type(action_input)=}, {type(action_input[0])=}')
            action = self.action_tokenizer.decode(action_input)[0] # 这里如果是action chunk还要改
            # LLM action是special token，但是actoin是button和camera
            # rich_console.log(f"[DEBUG] LLM action: {action_input}")
            rich_console.log(f"[DEBUG] converted to: {action}, {type(action)=}")


            # action = self.action_converter.convert(action_input, self._obs_shape)
            # rich_console.log(f"[DEBUG] LLM action: {action_input}")
            # rich_console.log(f"[DEBUG] converted to: {action}")
        elif isinstance(action_input, dict):
            # Agent格式：将整数转换为numpy数组
            import numpy as np
            action = {}
            for key, value in action_input.items():
                if isinstance(value, int):
                    action[key] = np.array(value)
                else:
                    action[key] = value
            rich_console.log(f"[DEBUG] Agent action (dict): {action_input}")
            rich_console.log(f"[DEBUG] converted to numpy: {action}")
        else:
            raise ValueError(f"Unsupported action type: {type(action_input)}")

        obs, reward, terminated, truncated, info = self.simulator.step(action)
        done = (terminated or truncated)
        self.last_obs = obs  # 更新最后观察状态
        return obs, reward, done, info

    def close(self):
        # 现在应能够正常保存视频
        print(f'[DEBUG] close with success')
        self.simulator.close()
        return 

    def reset(self, raw_prompt=None, origin_multi_modal_data=None, config=None):
        """重置环境

        现在只调用 self.simulator.reset(),返回其结果
        self.simulator 在 __init__ 时已经创建

        Args:
            raw_prompt: 未使用(兼容性参数)
            origin_multi_modal_data: 未使用(兼容性参数)
            config: 未使用(兼容性参数,配置在 __init__ 时已处理)

        Returns:
            (obs, info): 重置后的观察和信息
        """
        start_time = time.perf_counter()
        try:
            obs, info = self.simulator.reset()
            self.last_obs = obs  # 更新最后观察状态
            return (obs, info)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(f"[mc_simulator] reset elapsed={elapsed_ms:.2f} ms")

    def get_observation(self):
        """获取当前观察状态

        Returns:
            最后一次观察的数据，如果还没有初始化则返回None
        """
        return self.last_obs

if __name__ == "__main__":
    # 直接运行时需要修复相对导入
    # import sys
    # from pathlib import Path
    # current_dir = Path(__file__).parent
    # if str(current_dir) not in sys.path:
    #     sys.path.insert(0, str(current_dir))

    # # 直接从当前目录导入,避免触发 raycraft.__init__ 的循环导入
    # from utils.action_converter import ActionFromLLMConverter
    # from utils.sim_callbacks_loader import load_simulator_setup_from_yaml

    # tool = MCSimulator(
    #     "mc_simulator",
    #     "/mnt/shared-storage-user/tanxin/wuxiongbin/JarvisVLA/jarvisvla/evaluate/config/kill/kill_zombie.yaml",  # 相对路径
    #     {},
    #     config="/mnt/shared-storage-user/tanxin/wuxiongbin/JarvisVLA/jarvisvla/evaluate/config/kill/kill_zombie.yaml"  # 相对路径
    # )

    # # 现在 reset 只需要调用,不需要传 config
    # obs, info = tool.reset()

    # test_actions = [
    #     '<think> I need to target a Zombie in the distance to complete the given task \'kill_entity:zombie\'. From the current viewpoint, there appears to be a zombie visible on the horizon. The appropriate action would be to face towards the zombie and attack it.</think><answer> [{"action": "attack", "yaw": 0.0, "pitch": 0.0}] </answer>',
    #     '<think> I need to target a Zombie in the distance to complete the given task \'kill_entity:zombie\'. From the current viewpoint, there appears to be a zombie visible on the horizon. The appropriate action would be to face towards the zombie and attack it.</think><answer> [{"action": "attack", "yaw": 0.0, "pitch": 0.0}] </answer>',
    #     '<think> I need to target a Zombie in the distance to complete the given task \'kill_entity:zombie\'. From the current viewpoint, there appears to be a zombie visible on the horizon. The appropriate action would be to face towards the zombie and attack it.</think><answer> [{"action": "attack", "yaw": 0.0, "pitch": 0.0}] </answer>'
    # ]


    # for idx, resp in enumerate(test_actions, start=1):
    #     for _ in range(46):
    #         logger.info(f"[mc_simulator][test] action={resp}")
    #         obs, reward, done, info = tool.execute(resp)

    tokenizer = OneActionTokenizer(tokenizer_type="qwen2_vl")
    
    action_string = "[677, 151835, 151878, 151897, 151836, 4432, 151645]"
    res = tokenizer.decode(ast.literal_eval(action_string))
    breakpoint()
    # tool.simulator.close()
