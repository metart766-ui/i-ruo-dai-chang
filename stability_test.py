#!/usr/bin/env python3
"""
稳定性测试脚本
测试优化后的模拟器在长时间运行下的稳定性
"""

import time
import gc
import psutil
import os
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from optimized_real_time_di_ruo_dai_chang import OptimizedRealTimeSimulator

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def signal_handler(signum, frame):
    """处理中断信号"""
    print(f"\n\n🛑 收到中断信号 {signum}，正在保存测试结果...")
    if hasattr(stability_test, 'current_results'):
        save_results(stability_test.current_results)
    sys.exit(0)

def save_results(results: Dict):
    """保存测试结果到文件"""
    try:
        import json
        with open('stability_test_results.json', 'w', encoding='utf-8') as f:
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
                elif isinstance(value, list):
                    json_results[key] = []
                    for item in value:
                        if isinstance(item, (np.integer, np.floating)):
                            json_results[key].append(float(item))
                        else:
                            json_results[key].append(item)
                else:
                    json_results[key] = float(value) if isinstance(value, (np.integer, np.floating)) else value
            
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        print("✅ 稳定性测试结果已保存到 stability_test_results.json")
    except Exception as e:
        print(f"❌ 保存结果时出错: {e}")

def stability_test(duration_minutes: int = 10, target_steps: int = 10000):
    """
    运行稳定性测试
    
    Args:
        duration_minutes: 测试持续时间（分钟）
        target_steps: 目标模拟步数
    """
    print("🧪 开始递弱代偿模拟器稳定性测试")
    print(f"测试参数: {duration_minutes}分钟，目标{target_steps}步")
    print("="*60)
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 清理内存
    gc.collect()
    
    # 初始化测试参数
    test_start_time = datetime.now()
    memory_samples = []
    step_times = []
    step_memory_usage = []
    errors = []
    warnings = []
    
    print("🚀 创建模拟器实例...")
    initial_memory = get_memory_usage()
    start_time = time.time()
    
    try:
        # 创建模拟器（不显示GUI）
        simulator = OptimizedRealTimeSimulator(grid_size=50, initial_cells=100)
        
        # 禁用GUI更新以提高性能
        simulator.running = True
        simulator.start_time = datetime.now()
        
        creation_time = time.time() - start_time
        creation_memory = get_memory_usage() - initial_memory
        
        print(f"✅ 模拟器创建完成")
        print(f"   创建时间: {creation_time:.3f}s")
        print(f"   创建内存: {creation_memory:.1f}MB")
        print(f"   初始细胞: 100个")
        print()
        
        # 开始稳定性测试
        print("🔬 开始稳定性测试循环...")
        print("   每1000步报告一次状态")
        print("   按 Ctrl+C 可随时停止测试")
        print()
        
        step_count = 0
        last_report_time = time.time()
        last_report_step = 0
        
        while step_count < target_steps:
            step_start = time.time()
            initial_step_memory = get_memory_usage()
            
            try:
                # 执行一步模拟
                simulator.simulation_step()
                step_count += 1
                
                # 记录性能数据
                step_time = time.time() - step_start
                current_memory = get_memory_usage()
                step_memory = current_memory - initial_step_memory
                
                step_times.append(step_time)
                memory_samples.append(current_memory)
                step_memory_usage.append(step_memory)
                
                # 检查内存泄漏
                if len(memory_samples) > 100:
                    recent_avg = np.mean(memory_samples[-50:])
                    early_avg = np.mean(memory_samples[-100:-50])
                    if recent_avg > early_avg * 1.5:  # 内存增长超过50%
                        warnings.append(f"Step {step_count}: 检测到可能的内存泄漏")
                
                # 检查性能退化
                if len(step_times) > 100:
                    recent_avg_time = np.mean(step_times[-50:])
                    early_avg_time = np.mean(step_times[-100:-50])
                    if recent_avg_time > early_avg_time * 2.0:  # 性能下降超过100%
                        warnings.append(f"Step {step_count}: 检测到性能退化")
                
                # 定期报告
                if step_count % 1000 == 0:
                    elapsed = datetime.now() - test_start_time
                    current_fps = 1000 / (time.time() - last_report_time) if time.time() > last_report_time else 0
                    
                    print(f"📊 Step {step_count:6d} | "
                          f"时间: {str(elapsed).split('.')[0]:>8s} | "
                          f"FPS: {current_fps:6.1f} | "
                          f"内存: {current_memory:6.1f}MB | "
                          f"活细胞: {len(simulator.alive_cells):4d}")
                    
                    last_report_time = time.time()
                    last_report_step = step_count
                
                # 检查测试时间限制
                if datetime.now() - test_start_time > timedelta(minutes=duration_minutes):
                    print(f"\n⏰ 达到时间限制 ({duration_minutes}分钟)，停止测试")
                    break
                    
            except Exception as e:
                errors.append(f"Step {step_count}: {str(e)}")
                print(f"❌ Step {step_count} 出错: {e}")
                
                # 如果错误太多，停止测试
                if len(errors) > 10:
                    print("🛑 错误过多，停止测试")
                    break
        
        # 测试完成，收集结果
        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        
        # 分析结果
        if step_times:
            avg_step_time = np.mean(step_times)
            max_step_time = np.max(step_times)
            min_step_time = np.min(step_times)
            std_step_time = np.std(step_times)
        else:
            avg_step_time = max_step_time = min_step_time = std_step_time = 0
        
        if memory_samples:
            initial_test_memory = memory_samples[0] if memory_samples else initial_memory
            memory_growth = final_memory - initial_test_memory
            max_memory = np.max(memory_samples)
            avg_memory = np.mean(memory_samples)
        else:
            memory_growth = 0
            max_memory = final_memory
            avg_memory = final_memory
        
        # 生成测试报告
        results = {
            'test_summary': {
                'start_time': test_start_time.isoformat(),
                'duration_minutes': (datetime.now() - test_start_time).total_seconds() / 60,
                'total_steps': step_count,
                'target_steps': target_steps,
                'completion_rate': step_count / target_steps * 100,
                'errors_count': len(errors),
                'warnings_count': len(warnings)
            },
            'performance_metrics': {
                'total_simulation_time': total_time,
                'average_step_time': avg_step_time,
                'max_step_time': max_step_time,
                'min_step_time': min_step_time,
                'std_step_time': std_step_time,
                'steps_per_second': 1.0 / avg_step_time if avg_step_time > 0 else 0,
                'estimated_24h_steps': (1.0 / avg_step_time * 3600 * 24) if avg_step_time > 0 else 0
            },
            'memory_metrics': {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_growth_mb': memory_growth,
                'max_memory_mb': max_memory,
                'average_memory_mb': avg_memory,
                'memory_growth_rate': memory_growth / step_count if step_count > 0 else 0
            },
            'simulation_state': {
                'final_alive_cells': len(simulator.alive_cells),
                'final_avg_complexity': simulator.calculate_statistics().get('avg_complexity', 0),
                'final_avg_existence': simulator.calculate_statistics().get('avg_existence_degree', 0),
                'final_pc_product': simulator.calculate_statistics().get('p_times_c', 0)
            },
            'errors': errors,
            'warnings': warnings,
            'step_times_sample': step_times[-100:] if len(step_times) > 100 else step_times,
            'memory_samples': memory_samples[-100:] if len(memory_samples) > 100 else memory_samples
        }
        
        # 保存当前结果供信号处理函数使用
        stability_test.current_results = results
        
        # 打印测试报告
        print("\n" + "="*60)
        print("📋 稳定性测试报告")
        print("="*60)
        
        print(f"\n📊 测试概况:")
        print(f"   总步数: {step_count:,} / {target_steps:,} ({results['test_summary']['completion_rate']:.1f}%)")
        print(f"   总时间: {results['test_summary']['duration_minutes']:.1f} 分钟")
        print(f"   错误数: {len(errors)}")
        print(f"   警告数: {len(warnings)}")
        
        print(f"\n⚡ 性能指标:")
        print(f"   平均步时: {avg_step_time*1000:.3f}ms")
        print(f"   最快步时: {min_step_time*1000:.3f}ms")
        print(f"   最慢步时: {max_step_time*1000:.3f}ms")
        print(f"   步频: {results['performance_metrics']['steps_per_second']:.1f} 步/秒")
        print(f"   预估24小时步数: {results['performance_metrics']['estimated_24h_steps']:,.0f}")
        
        print(f"\n💾 内存指标:")
        print(f"   初始内存: {initial_memory:.1f}MB")
        print(f"   最终内存: {final_memory:.1f}MB")
        print(f"   内存增长: {memory_growth:.1f}MB")
        print(f"   峰值内存: {max_memory:.1f}MB")
        print(f"   每步内存增长: {results['memory_metrics']['memory_growth_rate']:.4f}MB/步")
        
        print(f"\n🔬 模拟状态:")
        print(f"   最终活细胞: {len(simulator.alive_cells)}")
        print(f"   平均复杂度: {results['simulation_state']['final_avg_complexity']:.2f}")
        print(f"   平均存在度: {results['simulation_state']['final_avg_existence']:.3f}")
        print(f"   P×C乘积: {results['simulation_state']['final_pc_product']:.3f}")
        
        if errors:
            print(f"\n❌ 错误记录 ({len(errors)}个):")
            for error in errors[-5:]:  # 只显示最近5个错误
                print(f"   {error}")
            if len(errors) > 5:
                print(f"   ... 还有 {len(errors)-5} 个错误")
        
        if warnings:
            print(f"\n⚠️  警告记录 ({len(warnings)}个):")
            for warning in warnings[-5:]:  # 只显示最近5个警告
                print(f"   {warning}")
            if len(warnings) > 5:
                print(f"   ... 还有 {len(warnings)-5} 个警告")
        
        # 稳定性评估
        print(f"\n🎯 稳定性评估:")
        stability_score = 100.0
        
        # 基于错误的扣分
        if len(errors) > 0:
            stability_score -= min(30.0, len(errors) * 3.0)
        
        # 基于内存增长的扣分
        if memory_growth > 10:  # 超过10MB内存增长
            stability_score -= min(20.0, (memory_growth - 10) * 2.0)
        
        # 基于性能稳定性的扣分
        if avg_step_time > 0 and std_step_time / avg_step_time > 0.5:  # 变异系数过大
            stability_score -= min(20.0, (std_step_time / avg_step_time - 0.5) * 40)
        
        stability_score = max(0.0, stability_score)
        
        if stability_score >= 90:
            print("   🟢 优秀 - 系统非常稳定")
        elif stability_score >= 70:
            print("   🟡 良好 - 系统基本稳定，有轻微问题")
        elif stability_score >= 50:
            print("   🟠 一般 - 系统存在一些问题，需要关注")
        else:
            print("   🔴 较差 - 系统不稳定，需要优化")
        
        print(f"   稳定性评分: {stability_score:.1f}/100")
        
        print("\n" + "="*60)
        
        # 保存结果
        save_results(results)
        
        return results
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现严重错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试保存部分结果
        if 'step_count' in locals():
            partial_results = {
                'error': str(e),
                'partial_steps': step_count,
                'test_summary': {
                    'start_time': test_start_time.isoformat(),
                    'errors_count': 1
                }
            }
            save_results(partial_results)
        
        return None

if __name__ == "__main__":
    # 运行稳定性测试
    # 默认测试10分钟或10000步，以先到者为准
    results = stability_test(duration_minutes=10, target_steps=10000)
    
    if results:
        print("\n✅ 稳定性测试完成！")
        
        # 基于测试结果给出建议
        if results['test_summary']['errors_count'] == 0 and results['memory_metrics']['memory_growth_mb'] < 5:
            print("🎉 恭喜！模拟器表现非常稳定，可以安全运行24小时模拟。")
        elif results['test_summary']['errors_count'] == 0:
            print("🟡 模拟器基本稳定，但建议监控内存使用情况。")
        else:
            print("🔴 模拟器存在稳定性问题，建议先解决错误再运行长时间模拟。")
    else:
        print("\n❌ 稳定性测试失败，请检查错误信息。")