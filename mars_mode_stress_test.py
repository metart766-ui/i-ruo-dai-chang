import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from siyan_experiment import SiyanSimulator
from concurrent.futures import ProcessPoolExecutor
import time
import os

class MarsModeStressTest:
    """
    马斯克式'火星模式'压力测试 (Mars Mode Stress Test)
    
    核心理念:
    1. First Principles: 测试系统的物理极限，而不是温和的统计规律。
    2. Extreme Environment: 环境波动(Volatility)随时间呈指数级增长，模拟极端环境。
    3. Failure Analysis: 关注系统是如何"断裂"的，寻找反脆弱的临界点。
    """
    
    def __init__(self, output_dir="mars_stress_test"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def run_extreme_simulation(self, difficulty_level="Hardcore"):
        """
        运行极端环境模拟
        
        difficulty_level:
        - "Hardcore": 环境恶化速度快，波动大
        - "Starship": 极其严苛，几乎不可能存活
        """
        print(f"🚀 启动火星模式压力测试 - 难度: {difficulty_level}")
        
        # 基础参数配置 - 即使是基础参数也比普通实验严苛
        base_params = {
            'grid_size': 60,        # 扩大空间
            'r': 0.98,              # 基础可靠性降低
            'base_death': 0.05      # 基础死亡率提高
        }
        
        # 根据难度设定动态环境参数
        if difficulty_level == "Hardcore":
            # 困难模式：环境敏感度高，维护成本高
            env_params = {
                'beta': 0.8,        # 极高的环境敏感度
                'gamma': 1.5,       # 极高的维护成本指数
                'volatility_growth': 1.001 # 环境波动每步增长 0.1%
            }
        elif difficulty_level == "Starship":
            # 星舰模式：地狱级难度
            env_params = {
                'beta': 1.2,        # 超敏感
                'gamma': 1.8,       # 惩罚性维护成本
                'volatility_growth': 1.002
            }
            
        # 运行模拟
        # 手动注入环境参数，因为SiyanSimulator可能不接受所有参数
        experiment = SiyanSimulator(
            grid_size=base_params['grid_size'],
            r=base_params['r'],
            base_death=base_params['base_death'],
            gamma=env_params['gamma'],
            beta=env_params['beta']
        )
        
        # 注入自定义的极端环境逻辑
        # 我们通过继承或动态修改属性来实现环境的指数级恶化
        history = []
        
        print("🔴 系统点火... 倒计时 3, 2, 1...")
        
        start_time = time.time()
        
        for step in range(3000):
            # 1. 动态调整环境波动性 (Volatility)
            # 在火星模式下，环境不仅是随机的，而且随机的幅度在变大
            current_volatility = 0.2 * (env_params['volatility_growth'] ** step)
            
            # 强制修改实验内部的环境波动
            # 注意：这里我们侵入式地修改环境参数，模拟外部气候恶化
            experiment.env_volatility = current_volatility
            
            # 2. 执行一步演化
            experiment.simulation_step()
            stats = experiment.history
            
            # 手动构建stats字典，因为simulation_step只更新history列表
            current_stats = {
                'step': step,
                'alive_ratio': stats['alive_ratio'][-1],
                'c_mean': stats['c_mean'][-1],
                'p_mean_serial': stats['p_mean_serial'][-1],
                'pc_serial': stats['pc_serial'][-1],
                'env_volatility': current_volatility
            }
            history.append(current_stats)
            
            # 3. 监控崩溃 (Rapid Unscheduled Disassembly)
            if current_stats['alive_ratio'] < 0.05:
                print(f"💥 系统在第 {step} 步发生'快速计划外解体' (RUD)！")
                break
                
            if step % 100 == 0:
                print(f"⏱️ Step {step}: 存活率 {current_stats['alive_ratio']:.2%}, 环境波动 {current_volatility:.4f}, 代偿度 {current_stats['c_mean']:.4f}")
        
        print(f"✅ 模拟结束。耗时: {time.time() - start_time:.2f}s")
        return pd.DataFrame(history)

    def analyze_failure_point(self, df):
        """分析故障点 (Failure Point Analysis)"""
        plt.figure(figsize=(15, 10))
        
        # 1. 存活率 vs 环境波动
        plt.subplot(2, 2, 1)
        plt.plot(df['step'], df['alive_ratio'], 'r-', label='Survival Rate')
        plt.plot(df['step'], df['env_volatility'], 'k--', alpha=0.5, label='Env Volatility')
        plt.title('Survival vs. Mars Environment')
        plt.legend()
        plt.grid(True)
        
        # 2. P vs C 的崩溃轨迹
        plt.subplot(2, 2, 2)
        plt.scatter(df['c_mean'], df['p_mean_serial'], c=df['step'], cmap='inferno', alpha=0.6)
        plt.colorbar(label='Time Step')
        plt.xlabel('Compensation (C)')
        plt.ylabel('Existence (P)')
        plt.title('The Path to Collapse (P vs C)')
        plt.grid(True)
        
        # 3. 熵增速率 (C的变化率)
        plt.subplot(2, 2, 3)
        # 计算C的移动平均变化率
        c_change = df['c_mean'].diff().rolling(window=20).mean()
        plt.plot(df['step'], c_change, 'b-')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.title('Rate of Complexity Growth (dC/dt)')
        plt.ylabel('Change in C')
        
        # 4. P*C 守恒失效分析
        plt.subplot(2, 2, 4)
        plt.plot(df['step'], df['pc_serial'], 'g-', label='P*C Product')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Theoretical Constant')
        plt.title('Conservation Law Breakdown')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mars_stress_test_dashboard.png'))
        print(f"📊 仪表盘已生成: {os.path.join(self.output_dir, 'mars_stress_test_dashboard.png')}")

if __name__ == "__main__":
    # 像马斯克一样思考：直接上强度
    tester = MarsModeStressTest()
    
    # 运行 "Starship" 级难度的测试
    print("\n==========================================")
    print("   MARS MODE: SYSTEM STRESS TEST PROTOCOL   ")
    print("==========================================")
    df = tester.run_extreme_simulation(difficulty_level="Starship")
    
    if not df.empty:
        tester.analyze_failure_point(df)
        
        # 输出关键遥测数据
        max_c = df['c_mean'].max()
        final_step = df['step'].iloc[-1]
        print("\n--- 任务遥测数据 ---")
        print(f"最大代偿度 (Max C): {max_c:.4f}")
        print(f"存活时长 (Steps): {final_step}")
        print(f"结论: 系统在面临指数级环境压力时，{'成功存活' if final_step == 2999 else '发生崩溃'}")
