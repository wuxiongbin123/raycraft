"""
RayCraft HTTP Server

依赖：
- fastapi
- uvicorn
- ray
- pillow
"""

import uuid
import base64
import io
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Union

import ray
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from raycraft.ray.global_pool import get_global_env_pool

import sys
minestudio_path = "/mnt/shared-storage-user/tanxin/wuxiongbin/raycraft/MineStudio"
sys.path.insert(0, minestudio_path)

# ============================================================================
# 数据模型
# ============================================================================

class BatchCreateRequest(BaseModel):
    count: int = Field(..., ge=1, description="环境数量，必须>=1")
    env_name: str = Field(default="minecraft", description="环境名称")
    env_kwargs: Union[Dict[str, Any], str] = Field(default_factory=dict, description="环境参数（dict）或YAML配置文件路径（str）")


class BatchCreateResponse(BaseModel):
    env_ids: list


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any]


class StepRequest(BaseModel):
    action: Union[str, Dict[str, int]] = Field(
        ...,
        description="Action格式支持两种：1) LLM格式的JSON字符串 '[{\"action\": \"forward\"}]' 2) Agent格式的dict {'buttons': 5, 'camera': 222}"
    )


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    ray_initialized: bool
    num_environments: int


# ============================================================================
# 辅助函数
# ============================================================================

def _prepare_env_config(base_config: Union[Dict[str, Any], str], env_id: str) -> Dict[str, Any]:
    """为单个环境准备独立的配置，确保record_path使用UUID子目录

    Args:
        base_config: 基础配置（dict或YAML路径）
        env_id: 环境UUID

    Returns:
        修改后的配置字典（sim_callbacks为实例化的对象）
    """
    import copy
    from pathlib import Path
    from raycraft.utils.sim_callbacks_loader import load_simulator_setup_from_yaml

    # 1. 如果是YAML路径，先加载并实例化callbacks
    if isinstance(base_config, str):
        sim_callbacks, env_overrides = load_simulator_setup_from_yaml(base_config)

        # 修改RecordCallback的record_path（已实例化的对象）
        for callback in sim_callbacks:
            if hasattr(callback, 'record_path'):
                base_path = Path(callback.record_path)
                # 在基础路径下创建UUID子目录（保持为Path对象）
                callback.record_path = base_path / env_id
                # 创建新的UUID子目录
                callback.record_path.mkdir(parents=True, exist_ok=True)

        # 重新组合config（sim_callbacks为实例化的对象）
        config = dict(env_overrides)
        config['sim_callbacks'] = sim_callbacks
        return config

    # 2. 如果是dict，深拷贝并修改
    config = copy.deepcopy(base_config)

    # 3. 修改sim_callbacks中RecordCallback的record_path（已实例化的对象）
    if 'sim_callbacks' in config and isinstance(config['sim_callbacks'], list):
        for callback in config['sim_callbacks']:
            if hasattr(callback, 'record_path'):
                base_path = Path(callback.record_path)
                # 保持为Path对象
                callback.record_path = base_path / env_id
                # 创建新的UUID子目录
                callback.record_path.mkdir(parents=True, exist_ok=True)

    return config


def serialize_value(value: Any) -> Any:
    """递归序列化值，处理numpy数组和Ray对象"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    elif 'ray' in str(type(value).__module__):
        # Ray对象（如ActorID, ObjectRef等）转为字符串
        return str(value)
    else:
        return value


def serialize_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """序列化observation，压缩RGB图像"""
    result = {}
    for key, value in obs.items():
        if key == 'rgb' and isinstance(value, np.ndarray):
            # 压缩为JPEG
            pil_image = Image.fromarray(value)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            jpeg_bytes = buffer.getvalue()
            result[key] = {
                'type': 'jpeg',
                'data': base64.b64encode(jpeg_bytes).decode('utf-8')
            }
        elif isinstance(value, np.ndarray):
            # 其他numpy数组转list
            result[key] = value.tolist()
        else:
            result[key] = serialize_value(value)
    return result


# ============================================================================
# FastAPI 应用
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化Ray"""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    get_global_env_pool()  # 确保命名actor存在
    yield


app = FastAPI(
    title="RayCraft HTTP API",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# API 路由
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    try:
        env_pool = get_global_env_pool()
        num_envs = ray.get(env_pool.get_num_envs.remote())
        return HealthResponse(
            status="healthy",
            ray_initialized=ray.is_initialized(),
            num_environments=num_envs
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/batch/envs", response_model=BatchCreateResponse)
async def batch_create_envs(request: BatchCreateRequest):
    """批量创建环境"""
    try:
        env_pool = get_global_env_pool()

        # 生成UUIDs
        env_ids = [str(uuid.uuid4()) for _ in range(request.count)]

        # 为每个环境创建独立的config（避免共享同一个record_path）
        configs = []
        for env_id in env_ids:
            config = _prepare_env_config(request.env_kwargs, env_id)
            configs.append(config)

        # 并行创建
        success = ray.get(env_pool.create_envs.remote(env_ids, configs))

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create some environments"
            )

        # 后台自动触发 reset（异步，不等待完成）
        # 客户端可通过 get_reset_result 获取结果
        for env_id in env_ids:
            env_ref = ray.get(env_pool.get_env.remote(env_id))
            if env_ref:
                # 触发异步 reset，不等待完成
                env_ref.reset.remote()

        return BatchCreateResponse(env_ids=env_ids)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Failed to create environments: {str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] {error_detail}")  # 打印到控制台
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )


@app.post("/envs/{env_id}/reset", response_model=ResetResponse)
async def reset_env(env_id: str):
    """Reset环境"""
    try:
        env_pool = get_global_env_pool()
        env_ref = ray.get(env_pool.get_env.remote(env_id))

        if env_ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        result = ray.get(env_ref.reset.remote())

        # MCEnvActor.reset() 只返回 obs，不是 (obs, info)
        # 兼容处理
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        return ResetResponse(
            observation=serialize_observation(obs),
            info=serialize_value(info)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset: {str(e)}"
        )


@app.get("/envs/{env_id}/reset_result", response_model=ResetResponse)
async def get_reset_result(env_id: str, wait: int = 0):
    """获取后台 reset 的结果（用于 batch_create_envs 后的异步 reset）

    Args:
        env_id: 环境ID
        wait: 等待时间（秒），如果reset未完成，最多等待这么多秒
    """
    try:
        import time
        env_pool = get_global_env_pool()
        env_ref = ray.get(env_pool.get_env.remote(env_id))

        if env_ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        # 轮询等待 reset 完成
        start_time = time.time()
        while True:
            obs, info = ray.get(env_ref.get_reset_result.remote())

            if obs is not None:
                # reset 完成
                return ResetResponse(
                    observation=serialize_observation(obs),
                    info=serialize_value(info)
                )

            # 检查是否超时
            elapsed = time.time() - start_time
            if elapsed >= wait:
                # 返回202表示reset正在进行中
                raise HTTPException(
                    status_code=status.HTTP_202_ACCEPTED,
                    detail=f"Environment {env_id} reset is still in progress"
                )

            # 等待一小段时间后重试
            await asyncio.sleep(1)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get reset result: {str(e)}"
        )


@app.post("/envs/{env_id}/step", response_model=StepResponse)
async def step_env(env_id: str, request: StepRequest):
    """Step环境"""
    try:
        
        env_pool = get_global_env_pool()
        env_ref = ray.get(env_pool.get_env.remote(env_id))

        if env_ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        result = ray.get(env_ref.step.remote(request.action))

        # MCEnvActor.step() 返回旧版Gym格式 (obs, reward, done, info)
        # 需要转换为新版格式 (obs, reward, terminated, truncated, info)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False  # 旧版没有truncated，默认False
        else:
            obs, reward, terminated, truncated, info = result

        return StepResponse(
            observation=serialize_observation(obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=serialize_value(info)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to step: {str(e)}"
        )


@app.delete("/envs/{env_id}", status_code=status.HTTP_204_NO_CONTENT)
async def close_env(env_id: str):
    """关闭环境"""
    print(f'[DEBUG] destroy {env_id=}!')
    try:
        env_pool = get_global_env_pool()
        success = ray.get(env_pool.close_env.remote(env_id))

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        return None

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to close: {str(e)}"
        )


# ============================================================================
# 启动
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
