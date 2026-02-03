#!/usr/bin/env python3
"""
性能对比测试脚本
比较原始模拟器和优化模拟器的性能差异
"""

import time
import gc
import psutil
import os
from typing import Dict, List
import numpy as np

# 导入两个版本的模拟器
from real_time_di_ruo_dai_chang import RealTimeDiRuoDaiChangSimulator as OriginalSimulator
from optimized_real_time_di_ruo_dai_chang import OptimizedRealTimeSimulator as OptimizedSimulator

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_simulator(simulator_class, name: str, steps: int = 1000) -> Dict:
    """基准测试模拟器性能"""
    print(f"\n{'='*50}")
    print(f"测试 {name}")
    print(f"{'='*50}")
    
    # 清理内存
    gc.collect()
    
    # 记录初始状态
    initial_memory = get_memory_usage()
    start_time = time.time()
    
    # 创建模拟器实例
    simulator = simulator_class(grid_size=50, initial_cells=100)
    
    creation_time = time.time() - start_time
    creation_memory = get_memory_usage() - initial_memory
    
    print(f"创建时间: {creation_time:.3f}s")
    print(f"创建内存: {creation_memory:.1f}MB")
    
    # 运行指定步数的模拟
    step_times = []
    memory_usage = []
    
    for i in range(steps):
        step_start = time.time()
        initial_step_memory = get_memory_usage()
        
        simulator.simulation_step()
        
        step_time = time.time() - step_start
        step_memory = get_memory_usage() - initial_step_memory
        
        step_times.append(step_time)
        memory_usage.append(get_memory_usage())
        
        if (i + 1) % 200 == 0:
            print(f"  完成 {i+1}/{steps} 步")
    
    total_time = time.time() - start_time - creation_time
    avg_step_time = np.mean(step_times)
    max_step_time = np.max(step_times)
    min_step_time = np.min(step_times)
    final_memory = get_memory_usage()
    peak_memory = max(memory_usage)
    
    # 获取统计信息
    stats = simulator.calculate_statistics() if hasattr(simulator, 'calculate_statistics') else {}
    
    results = {
        'name': name,
        'creation_time': creation_time,
        'creation_memory': creation_memory,
        'total_simulation_time': total_time,
        'avg_step_time': avg_step_time,
        'max_step_time': max_step_time,
        'min_step_time': min_step_time,
        'final_memory': final_memory - initial_memory,
        'peak_memory': peak_memory - initial_memory,
        'steps_per_second': 1.0 / avg_step_time if avg_step_time > 0 else 0,
        'final_stats': stats
    }
    
    print(f"\n{name} 性能结果:")
    print(f"  总模拟时间: {total_time:.3f}s")
    print(f"  平均步时: {avg_step_time*1000:.3f}ms")
    print(f"  最快步时: {min_step_time*1000:.3f}ms")
    print(f"  最慢步时: {max_step_time*1000:.3f}ms")
    print(f"  步频: {results['steps_per_second']:.1f} 步/秒")
    print(f"  峰值内存: {results['peak_memory']:.1f}MB")
    print(f"  最终内存: {results['final_memory']:.1f}MB")
    
    return results

def compare_simulators():
    """对比两个模拟器的性能"""
    print("🚀 开始递弱代偿模拟器性能对比测试")
    print("测试环境: 1000步模拟，50x50网格，100个初始细胞")
    
    # 测试原始版本
    original_results = benchmark_simulator(OriginalSimulator, "原始版本", steps=1000)
    
    # 清理内存
    gc.collect()
    time.sleep(2)
    
    # 测试优化版本
    optimized_results = benchmark_simulator(OptimizedSimulator, "优化版本", steps=1000)
    
    # 生成对比报告
    print(f"\n{'='*60}")
    print("📊 性能对比分析报告")
    print(f"{'='*60}")
    
    # 性能提升计算
    speedup = original_results['avg_step_time'] / optimized_results['avg_step_time']
    memory_reduction = (1 - optimized_results['final_memory'] / original_results['final_memory']) * 100
    
    print(f"\n⚡ 速度提升:")
    print(f"  平均步时改善: {speedup:.2f}x 更快")
    print(f"  步频提升: {optimized_results['steps_per_second'] - original_results['steps_per_second']:.1f} 步/秒")
    
    print(f"\n💾 内存优化:")
    print(f"  内存减少: {memory_reduction:.1f}%")
    print(f"  峰值内存减少: {(1 - optimized_results['peak_memory'] / original_results['peak_memory']) * 100:.1f}%")
    
    print(f"\n🎯 关键优化点:")
    print("  1. 使用集合存储活细胞位置 - O(1)查找")
    print("  2. 数组缓存活细胞列表 - 避免重复遍历")
    print("  3. 降低统计记录频率 - 每5步记录一次")
    print("  4. 减少历史记录长度 - 从1000降到500")
    print("  5. 批量处理细胞更新 - 减少字典操作")
    print("  6. 优化散点图绘制 - 使用坐标数组")
    
    print(f"\n📈 24小时模拟预估:")
    original_24h_steps = 24 * 3600 * 20  # 假设20 FPS
    optimized_24h_steps = 24 * 3600 * 20
    
    original_24h_time = original_24h_steps * original_results['avg_step_time']
    optimized_24h_time = optimized_24h_steps * optimized_results['avg_step_time']
    
    print(f"  原始版本24小时CPU时间: {original_24h_time/3600:.2f}小时")
    print(f"  优化版本24小时CPU时间: {optimized_24h_time/3600:.2f}小时")
    print(f"  CPU时间节省: {(original_24h_time - optimized_24h_time)/3600:.2f}小时")
    
    return {
        'original': original_results,
        'optimized': optimized_results,
        'speedup': speedup,
        'memory_reduction': memory_reduction
    }

if __name__ == "__main__":
    try:
        results = compare_simulators()
        
        # 保存结果到文件
        import json
        with open('performance_comparison_results.json', 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python原生类型
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (np.integer, np.floating)):
                            json_results[key][k] = float(v)
                        else:
                            json_results[key][k] = v
                else:
                    json_results[key] = float(value) if isinstance(value, (np.integer, np.floating)) else value
            
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 性能对比完成！结果已保存到 performance_comparison_results.json")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()