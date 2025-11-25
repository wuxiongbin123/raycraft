#!/usr/bin/env python3
"""
并行性能测试 - 测试多环境并行执行性能
记录 __init__、reset、step 以及总耗时
"""
import ray
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from raycraft.ray.client import MCRayClient
from raycraft.ray.global_pool import get_global_env_pool


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    env_uuid: str
    init_time: float  # Client 初始化耗时
    reset_time: float  # Reset 耗时
    step_times: List[float]  # 每步耗时列表
    total_time: float  # 总耗时

    @property
    def avg_step_time(self) -> float:
        """平均步骤耗时"""
        return np.mean(self.step_times) if self.step_times else 0.0

    @property
    def max_step_time(self) -> float:
        """最大步骤耗时"""
        return max(self.step_times) if self.step_times else 0.0

    @property
    def min_step_time(self) -> float:
        """最小步骤耗时"""
        return min(self.step_times) if self.step_times else 0.0


@ray.remote
def run_env_remote(uuid: str, num_steps: int) -> PerformanceMetrics:
    """
    Ray remote function - 在 worker 进程中执行
    必须在函数内部处理所有导入，因为 worker 是独立进程
    """
    import sys
    from pathlib import Path
    import time
    import traceback as tb

    # 添加 raycraft 到 sys.path（worker 进程需要）
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from raycraft.ray.client import MCRayClient

    metrics = PerformanceMetrics(
        env_uuid=uuid,
        init_time=0.0,
        reset_time=0.0,
        step_times=[],
        total_time=0.0
    )

    total_start = time.perf_counter()
    error_msg = None

    try:
        print(f"[Worker {uuid[:8]}] 开始执行...")

        # 1. Client 初始化
        print(f"[Worker {uuid[:8]}] 初始化 Client...")
        init_start = time.perf_counter()
        client = MCRayClient(uuid=uuid)
        metrics.init_time = time.perf_counter() - init_start
        print(f"[Worker {uuid[:8]}] Client 初始化完成: {metrics.init_time*1000:.2f}ms")

        # 2. Reset
        print(f"[Worker {uuid[:8]}] Reset 环境...")
        reset_start = time.perf_counter()
        obs = client.reset()
        metrics.reset_time = time.perf_counter() - reset_start
        print(f"[Worker {uuid[:8]}] Reset 完成: {metrics.reset_time*1000:.2f}ms, obs type: {type(obs)}")

        # 3. 执行多步
        print(f"[Worker {uuid[:8]}] 执行 {num_steps} 步...")
        for i in range(num_steps):
            step_start = time.perf_counter()
            result = client.step('[{"action": "forward"}]')
            step_time = time.perf_counter() - step_start
            metrics.step_times.append(step_time)
            if i == 0:
                print(f"[Worker {uuid[:8]}] Step 0 完成: {step_time*1000:.2f}ms, reward={result.reward}, done={result.done}")

        print(f"[Worker {uuid[:8]}] 所有步骤完成，共 {len(metrics.step_times)} 步")

        # 4. 关闭
        print(f"[Worker {uuid[:8]}] 关闭环境...")
        client.close()

        metrics.total_time = time.perf_counter() - total_start
        print(f"[Worker {uuid[:8]}] 总耗时: {metrics.total_time:.3f}s")

    except Exception as e:
        error_msg = f"环境 {uuid[:8]} 执行失败: {e}\n{tb.format_exc()}"
        print(f"❌ {error_msg}")
        # 重新抛出异常，让主进程能看到
        raise

    return metrics


def run_parallel_with_ray(uuids: List[str], num_steps: int = 20) -> List[PerformanceMetrics]:
    """
    使用 Ray 异步特性实现真正并行

    Args:
        uuids: 环境UUID列表
        num_steps: 每个环境执行的步数

    Returns:
        性能指标列表
    """
    print("\n🚀 使用 Ray 异步并行执行...")
    print(f"   启动 {len(uuids)} 个并行任务，每个执行 {num_steps} 步")

    parallel_start = time.perf_counter()

    # 并行启动所有任务
    futures = [run_env_remote.remote(uuid, num_steps) for uuid in uuids]
    print(f"   ✅ 所有任务已提交")

    # 等待所有任务完成，带超时和错误处理
    try:
        print(f"   ⏳ 等待任务完成...")
        results = ray.get(futures, timeout=300)  # 5分钟超时
    except Exception as e:
        print(f"   ❌ 任务执行失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    parallel_total = time.perf_counter() - parallel_start

    print(f"✅ 并行执行完成，总耗时: {parallel_total:.3f}s")
    print(f"   成功完成 {len(results)} 个任务")

    return results


def run_env_sequential(uuid: str, num_steps: int = 20) -> PerformanceMetrics:
    """
    顺序运行单个环境（用于线程池）

    Args:
        uuid: 环境UUID
        num_steps: 执行步数

    Returns:
        PerformanceMetrics: 性能指标
    """
    metrics = PerformanceMetrics(
        env_uuid=uuid,
        init_time=0.0,
        reset_time=0.0,
        step_times=[],
        total_time=0.0
    )

    total_start = time.perf_counter()

    try:
        # 1. Client 初始化
        init_start = time.perf_counter()
        client = MCRayClient(uuid=uuid)
        metrics.init_time = time.perf_counter() - init_start

        # 2. Reset
        reset_start = time.perf_counter()
        _ = client.reset()  # obs 仅用于验证成功
        metrics.reset_time = time.perf_counter() - reset_start

        # 3. 执行多步
        for _ in range(num_steps):
            step_start = time.perf_counter()
            _ = client.step('[{"action": "forward"}]')  # result 仅用于验证成功
            step_time = time.perf_counter() - step_start
            metrics.step_times.append(step_time)

        # 4. 关闭
        client.close()

        metrics.total_time = time.perf_counter() - total_start

    except Exception as e:
        print(f"❌ 环境 {uuid[:8]} 执行失败: {e}")
        import traceback
        traceback.print_exc()

    return metrics


def run_parallel_with_threads(uuids: List[str], num_steps: int = 20) -> List[PerformanceMetrics]:
    """
    使用线程池实现并行（备选方案）

    Args:
        uuids: 环境UUID列表
        num_steps: 每个环境执行的步数

    Returns:
        性能指标列表
    """
    print("\n🧵 使用线程池并行执行...")

    parallel_start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=len(uuids)) as executor:
        futures = {executor.submit(run_env_sequential, uuid, num_steps): uuid for uuid in uuids}

        for future in as_completed(futures):
            uuid = futures[future]
            try:
                metrics = future.result()
                results.append(metrics)
                print(f"   ✅ 环境 {uuid[:8]} 完成")
            except Exception as e:
                print(f"   ❌ 环境 {uuid[:8]} 失败: {e}")

    parallel_total = time.perf_counter() - parallel_start
    print(f"✅ 并行执行完成，总耗时: {parallel_total:.3f}s")

    return results


def print_performance_report(metrics_list: List[PerformanceMetrics]):
    """打印性能报告"""
    print("\n" + "="*80)
    print("📊 性能报告")
    print("="*80)

    # 单个环境详细数据
    print("\n【单环境详细数据】")
    print(f"{'UUID':<10} {'Init(ms)':<12} {'Reset(ms)':<12} {'Avg Step(ms)':<15} {'Total(s)':<10}")
    print("-" * 80)

    for m in metrics_list:
        print(f"{m.env_uuid[:8]:<10} "
              f"{m.init_time*1000:<12.2f} "
              f"{m.reset_time*1000:<12.2f} "
              f"{m.avg_step_time*1000:<15.2f} "
              f"{m.total_time:<10.3f}")

    # 汇总统计
    print("\n【汇总统计】")
    all_init_times = [m.init_time * 1000 for m in metrics_list]
    all_reset_times = [m.reset_time * 1000 for m in metrics_list]
    all_avg_step_times = [m.avg_step_time * 1000 for m in metrics_list]
    all_total_times = [m.total_time for m in metrics_list]

    print(f"环境数量: {len(metrics_list)}")
    print(f"\nInit 耗时 (ms):")
    print(f"  平均: {np.mean(all_init_times):.2f}")
    print(f"  最大: {np.max(all_init_times):.2f}")
    print(f"  最小: {np.min(all_init_times):.2f}")

    print(f"\nReset 耗时 (ms):")
    print(f"  平均: {np.mean(all_reset_times):.2f}")
    print(f"  最大: {np.max(all_reset_times):.2f}")
    print(f"  最小: {np.min(all_reset_times):.2f}")

    print(f"\n平均 Step 耗时 (ms):")
    print(f"  平均: {np.mean(all_avg_step_times):.2f}")
    print(f"  最大: {np.max(all_avg_step_times):.2f}")
    print(f"  最小: {np.min(all_avg_step_times):.2f}")

    print(f"\n单环境总耗时 (s):")
    print(f"  平均: {np.mean(all_total_times):.3f}")
    print(f"  最大: {np.max(all_total_times):.3f}")
    print(f"  最小: {np.min(all_total_times):.3f}")

    # 性能分析
    print("\n【性能分析】")
    total_steps = sum(len(m.step_times) for m in metrics_list)
    total_time = np.max(all_total_times)  # 并行执行取最慢的
    throughput = total_steps / total_time if total_time > 0 else 0

    print(f"总步数: {total_steps}")
    print(f"并行总耗时: {total_time:.3f}s")
    print(f"吞吐量: {throughput:.2f} steps/s")
    print(f"平均每步: {1000/throughput:.2f} ms/step" if throughput > 0 else "N/A")

    print("="*80)


def main():
    """主测试函数"""
    # 配置
    PROJECT_ROOT = Path(__file__).parent.parent
    config_path = PROJECT_ROOT / "configs/kill/kill_zombie_with_record.yaml"

    NUM_ENVS = 2  # 环境数量
    NUM_STEPS = 20  # 每个环境执行步数
    USE_RAY_PARALLEL = True  # True=Ray并行, False=线程池并行

    print("="*80)
    print("🎯 并行性能测试")
    print("="*80)
    print(f"📍 配置文件: {config_path}")
    print(f"🔢 环境数量: {NUM_ENVS}")
    print(f"📊 每环境步数: {NUM_STEPS}")
    print(f"⚙️  并行方式: {'Ray 异步' if USE_RAY_PARALLEL else '线程池'}")

    # 初始化 Ray（启用 log_to_driver 以查看 worker 输出）
    print("\n🚀 初始化Ray...")
    ray.init(ignore_reinit_error=True, log_to_driver=True, include_dashboard=False)
    print("✅ Ray 初始化成功")

    # 清理旧的全局环境池（确保干净的测试环境）
    print("\n🧹 清理旧环境池...")
    from raycraft.ray.global_pool import reset_global_env_pool
    reset_global_env_pool()
    print("✅ 环境池已清理")

    # 批量创建环境
    print(f"\n📦 批量创建 {NUM_ENVS} 个环境...")
    create_start = time.perf_counter()
    try:
        uuids = MCRayClient.create_batch(num_envs=NUM_ENVS, config_path=str(config_path))
        create_time = time.perf_counter() - create_start
        print(f"✅ 环境创建成功! 耗时: {create_time:.3f}s")
        print(f"   UUID: {[u[:8] for u in uuids]}")
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        ray.shutdown()
        return

    # 验证环境
    print("\n🔍 验证环境...")
    env_pool = get_global_env_pool()
    for uuid in uuids:
        exists = ray.get(env_pool.env_exists.remote(uuid))
        status = '✅' if exists else '❌'
        print(f"   {status} UUID {uuid[:8]}")

    # 并行性能测试
    if USE_RAY_PARALLEL:
        metrics_list = run_parallel_with_ray(uuids, NUM_STEPS)
    else:
        metrics_list = run_parallel_with_threads(uuids, NUM_STEPS)

    # 打印性能报告
    print_performance_report(metrics_list)

    # 清理
    print("\n🧹 清理资源...")
    ray.shutdown()
    print("✅ 测试完成")


if __name__ == "__main__":
    main()