"""
环境池管理器 - MVP2核心组件
支持UUID-环境映射和批量环境管理
"""

import ray
import time
import logging
from typing import Dict, List, Optional
from uuid import uuid4
import os
from pathlib import Path

# ===== 新增：日志配置 =====
# LOG_DIR = "/mnt/shared-storage-user/tanxin/wuxiongbin/AgentGym/agentenv-mc/raycraft/ray/logs"
LOG_DIR = Path(__file__).parent.parent.parent / "logs" / "ray"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "envpool.log")
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

# 为Ray Actor环境配置日志
logger = logging.getLogger("EnvPool")
logger.setLevel(logging.INFO)

# 清除现有的handlers（避免重复）
logger.handlers.clear()

# 添加文件处理器
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(LOG_FORMAT)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 添加控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)

# 防止日志向上传播到根logger
logger.propagate = False

logger.info("EnvPool logger configured successfully")
# =======================

@ray.remote
class EnvPool:
    """Ray环境池管理器

    功能：
    - 管理多个MCEnvActor实例
    - 支持UUID->环境的映射
    - 支持不同配置的环境创建
    - 环境状态管理和回收
    """

    def __init__(self):
        """初始化环境池"""
        # 在Ray Actor进程中重新配置日志
        import logging
        import os
        import sys
        
        # 确保项目根目录在sys.path中（Ray Actor在独立进程中运行）
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # 确保MineStudio也在sys.path中
        minestudio_path = project_root / "MineStudio"
        if minestudio_path.exists() and str(minestudio_path) not in sys.path:
            sys.path.insert(0, str(minestudio_path))
        
        # 确保日志目录存在
        log_dir = Path(__file__).parent.parent.parent / "logs" / "ray"
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置Actor进程中的日志
        actor_logger = logging.getLogger("EnvPool")
        actor_logger.setLevel(logging.INFO)
        
        # 清除现有handlers
        actor_logger.handlers.clear()
        
        # 添加文件处理器
        log_file = os.path.join(log_dir, "envpool.log")
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
        file_handler.setFormatter(formatter)
        actor_logger.addHandler(file_handler)
        
        # 防止日志向上传播
        actor_logger.propagate = False
        
        # 初始化环境池状态
        self.env_registry = {}    # uuid -> MCEnvActor
        self.env_configs = {}     # uuid -> config_dict
        self.env_status = {}      # uuid -> "idle"/"busy"/"failed"
        self.created_time = {}    # uuid -> creation_timestamp
        self.logger = actor_logger  # 保存logger引用供其他方法使用

        self.logger.info("EnvPool initialized in Ray Actor process")

    def create_envs(self, uuids: List[str], configs: List[dict]) -> bool:
        """批量创建环境（并行创建并等待验证）

        Args:
            uuids: UUID列表
            configs: 配置列表，与uuids一一对应

        Returns:
            bool: 是否全部创建成功
        """
        try:
            from .actors import MCEnvActor

            # 第一步：并行启动所有环境创建
            actor_refs = []
            uuid_to_actor = {}

            for uuid, config in zip(uuids, configs):
                try:
                    # 创建环境Actor（异步，立即返回）
                    env_actor = MCEnvActor.remote(config)

                    # 注册到池中，状态标记为"creating"
                    self.env_registry[uuid] = env_actor
                    self.env_configs[uuid] = config
                    self.env_status[uuid] = "creating"
                    self.created_time[uuid] = time.time()

                    uuid_to_actor[uuid] = env_actor
                    self.logger.debug(f"Environment {uuid} creation started (async)")

                except Exception as e:
                    self.logger.error(f"Failed to start environment {uuid}: {e}")
                    self.env_status[uuid] = "failed"

            self.logger.info(f"Started creating {len(uuid_to_actor)} environments in parallel")

            # 第二步：并行等待所有环境初始化完成
            successful_uuids = []
            failed_uuids = []

            # 并行调用所有actor的get_stats（不等待）
            uuid_to_future = {
                uuid: actor.get_stats.remote()
                for uuid, actor in uuid_to_actor.items()
            }

            # 并行等待所有future完成
            for uuid, future in uuid_to_future.items():
                try:
                    # 并行等待（Ray会并发执行所有remote调用）
                    ray.get(future, timeout=120)  # 增加timeout，因为可能需要等待MinecraftSim初始化
                    # 成功：标记为idle
                    self.env_status[uuid] = "idle"
                    successful_uuids.append(uuid)
                    self.logger.info(f"Environment {uuid} initialized successfully")
                except Exception as e:
                    # 失败：标记为failed，并从注册表移除
                    self.logger.error(f"Environment {uuid} initialization failed: {e}")
                    self.env_status[uuid] = "failed"
                    failed_uuids.append(uuid)
                    # 清理失败的环境
                    try:
                        del self.env_registry[uuid]
                        del self.env_configs[uuid]
                        del self.env_status[uuid]
                        del self.created_time[uuid]
                    except:
                        pass

            self.logger.info(
                f"Environment creation finished: {len(successful_uuids)} succeeded, "
                f"{len(failed_uuids)} failed"
            )

            return len(failed_uuids) == 0

        except Exception as e:
            self.logger.error(f"Failed to create environments: {e}")
            return False

    def env_exists(self, uuid: str) -> bool:
        """检查环境是否存在

        Args:
            uuid: 环境UUID

        Returns:
            bool: 环境是否存在且状态正常
        """
        return uuid in self.env_registry and self.env_status.get(uuid) != "failed"

    def get_env_by_uuid(self, uuid: str):
        """根据UUID获取环境

        Args:
            uuid: 环境UUID

        Returns:
            MCEnvActor: 环境Actor引用

        Raises:
            ValueError: UUID不存在或环境状态异常
        """
        if uuid not in self.env_registry:
            raise ValueError(f"Environment with UUID {uuid} not found")

        if self.env_status[uuid] == "failed":
            raise ValueError(f"Environment {uuid} is in failed state")

        # 标记为忙碌状态
        self.env_status[uuid] = "busy"

        self.logger.debug(f"Environment {uuid} allocated")
        return self.env_registry[uuid]

    def return_env(self, uuid: str) -> bool:
        """归还环境到池中

        Args:
            uuid: 环境UUID

        Returns:
            bool: 是否成功归还
        """
        if uuid not in self.env_registry:
            self.logger.warning(f"Trying to return non-existent environment {uuid}")
            return False

        # 标记为空闲状态
        self.env_status[uuid] = "idle"
        self.logger.debug(f"Environment {uuid} returned to pool")
        return True

    def destroy_env(self, uuid: str) -> bool:
        """销毁指定环境

        Args:
            uuid: 环境UUID

        Returns:
            bool: 是否成功销毁
        """
        if uuid not in self.env_registry:
            self.logger.warning(f"Trying to destroy non-existent environment {uuid}")
            return False

        try:
            env_actor = self.env_registry[uuid]
        
            # 👇 先正常关闭（关键！）
            try:
                ray.get(env_actor.close.remote(), timeout=30)  # 等待 close 完成
                self.logger.info(f"Environment {uuid} closed successfully")
            except Exception as e:
                self.logger.error(f"Environment {uuid} close failed: {e}")
                # 即使 close 失败，仍继续销毁
                
            # 销毁Ray Actor
            ray.kill(self.env_registry[uuid])

            # 从注册表中移除
            del self.env_registry[uuid]
            del self.env_configs[uuid]
            del self.env_status[uuid]
            del self.created_time[uuid]

            self.logger.info(f"Environment {uuid} destroyed")
            return True

        except Exception as e:
            self.logger.error(f"Failed to destroy environment {uuid}: {e}")
            return False

    def batch_destroy(self, uuids: List[str]) -> int:
        """批量销毁环境

        Args:
            uuids: 要销毁的UUID列表

        Returns:
            int: 成功销毁的环境数量
        """
        success_count = 0
        for uuid in uuids:
            if self.destroy_env(uuid):
                success_count += 1

        self.logger.info(f"Batch destroyed {success_count}/{len(uuids)} environments")
        return success_count

    def get_pool_stats(self) -> Dict:
        """获取环境池统计信息

        Returns:
            Dict: 包含各种统计信息的字典
        """
        total_envs = len(self.env_registry)
        idle_envs = sum(1 for status in self.env_status.values() if status == "idle")
        busy_envs = sum(1 for status in self.env_status.values() if status == "busy")
        failed_envs = sum(1 for status in self.env_status.values() if status == "failed")

        return {
            "total_environments": total_envs,
            "idle_environments": idle_envs,
            "busy_environments": busy_envs,
            "failed_environments": failed_envs,
            "uptime": time.time() - min(self.created_time.values()) if self.created_time else 0,
            "environment_list": list(self.env_registry.keys())
        }

    def health_check(self) -> Dict:
        """环境池健康检查

        Returns:
            Dict: 健康状态信息
        """
        stats = self.get_pool_stats()

        # 简单的健康检查逻辑
        healthy = (
            stats["total_environments"] > 0 and
            stats["failed_environments"] == 0
        )

        return {
            "healthy": healthy,
            "total_envs": stats["total_environments"],
            "failed_envs": stats["failed_environments"],
            "timestamp": time.time()
        }

    # ===== HTTP API 支持方法 =====

    def create_env(self, env_id: str, env_name: str, env_kwargs: Dict) -> "ray.ActorHandle":
        """创建单个环境（HTTP API专用）

        Args:
            env_id: 环境UUID
            env_name: 环境名称（暂未使用，预留）
            env_kwargs: 环境配置参数

        Returns:
            ray.ActorHandle: 环境Actor引用
        """
        # 使用现有的create_envs方法
        success = self.create_envs([env_id], [env_kwargs])
        if not success:
            raise RuntimeError(f"Failed to create environment {env_id}")

        return self.env_registry[env_id]

    def get_env(self, env_id: str) -> Optional["ray.ActorHandle"]:
        """获取环境（HTTP API专用）

        Args:
            env_id: 环境UUID

        Returns:
            ray.ActorHandle: 环境Actor引用，如果不存在返回None
        """
        if not self.env_exists(env_id):
            return None

        return self.get_env_by_uuid(env_id)

    def close_env(self, env_id: str) -> bool:
        """关闭环境（HTTP API专用）

        Args:
            env_id: 环境UUID

        Returns:
            bool: 是否成功关闭
        """
        return self.destroy_env(env_id)

    def list_envs(self) -> List[str]:
        """列出所有环境ID（HTTP API专用）

        Returns:
            List[str]: 环境UUID列表
        """
        return list(self.env_registry.keys())

    def get_num_envs(self) -> int:
        """获取环境数量（HTTP API专用）

        Returns:
            int: 环境总数
        """
        return len(self.env_registry)