"""
RayCraft HTTP Client

只依赖：
- requests (HTTP库)
- 标准库 (typing, json等)

不依赖：
- ray
- numpy
- PIL
- minestudio
"""

import requests
from typing import Tuple, Dict, Any, List, Optional, Union


class RemoteEnv:
    """远程环境客户端（轻量级）"""

    def __init__(self, server_url: str, env_id: str):
        """
        Args:
            server_url: 服务器地址，如 "http://10.0.1.100:8000"
            env_id: 环境UUID
        """
        self.server_url = server_url.rstrip('/')
        self.env_id = env_id
        self.session = requests.Session()  # 连接复用
        
        self.reset_obs = None
        self.reset_info = None

    def reset(self, timeout: int = 120) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset环境

        Args:
            timeout: 超时时间（秒），首次reset慢，建议120秒

        Returns:
            (observation, info)
        """
        resp = self.session.post(
            f"{self.server_url}/envs/{self.env_id}/reset",
            timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        self.reset_obs = data["observation"]
        self.reset_info = data["info"]
        
        return data["observation"], data["info"]

    def step(
        self,
        action: Union[str, Dict[str, int]],
        timeout: int = 30
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Step环境

        Args:
            action: Action格式支持两种：
                - str: LLM格式 '[{"action": "forward"}]'
                - dict: Agent格式 {'buttons': 5, 'camera': 222}
            timeout: 超时时间（秒），默认30秒

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        resp = self.session.post(
            f"{self.server_url}/envs/{self.env_id}/step",
            json={"action": action},
            timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data["observation"],
            data["reward"],
            data["terminated"],
            data["truncated"],
            data["info"]
        )

    def close(self, timeout: int = 30):
        """关闭环境

        Args:
            timeout: 超时时间（秒），需要等待MP4保存
        """
        print(f'[DEBUG] close the env!')
        resp = self.session.delete(
            f"{self.server_url}/envs/{self.env_id}",
            timeout=timeout
        )
        resp.raise_for_status()
        self.session.close()

    def get_reset_result(self, wait: int = 600, timeout: int = 600) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """获取最后一次 reset 的结果

        用于 batch_create_envs 后台异步 reset 的结果获取

        Args:
            wait: 服务器端等待时间（秒），如果reset未完成，服务器最多等待这么多秒
            timeout: HTTP超时时间（秒），应该 >= wait

        Returns:
            (observation, info)

        Raises:
            HTTPError: 如果reset超时或环境不存在
        """
        resp = self.session.get(
            f"{self.server_url}/envs/{self.env_id}/reset_result",
            params={"wait": wait},
            timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        self.reset_obs = data["observation"]
        self.reset_info = data["info"]

        return data["observation"], data["info"]
    
    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出，自动close"""
        try:
            self.close()
        except:
            pass  # 忽略close错误


def create_remote_envs(
    server_url: str,
    count: int,
    env_kwargs: Optional[Dict[str, Any]] = None
) -> List[RemoteEnv]:
    """批量创建RemoteEnv对象

    Args:
        server_url: 服务器地址
        count: 环境数量
        env_kwargs: 环境参数

    Returns:
        RemoteEnv对象列表
    """
    resp = requests.post(
        f"{server_url.rstrip('/')}/batch/envs",
        json={
            "count": count,
            "env_name": "minecraft",
            "env_kwargs": env_kwargs or {}
        },
        timeout=60
    )
    resp.raise_for_status()
    env_ids = resp.json()["env_ids"]

    return [RemoteEnv(server_url, env_id) for env_id in env_ids]
