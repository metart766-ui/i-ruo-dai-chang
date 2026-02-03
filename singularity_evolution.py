import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from siyan_experiment import SiyanSimulator, Individual
import copy
import random
import os

class SingularityIndividual(Individual):
    """
    拥有'奇点'能力的个体
    具备 Neuralink/Refactoring 能力，可以主动降低自身熵(复杂度)
    """
    def __init__(self, x, y, complexity=1, energy=5.0):
        super().__init__(x, y, complexity, energy)
        self.refactored_count = 0

    def refactor(self, efficiency=0.5, cost=2.0):
        """
        重构代码/基因：降低复杂度，但保持功能
        efficiency: 复杂度降低比例
        cost: 重构消耗的能量
        """
        if self.energy > cost and self.complexity > 1:
            self.energy -= cost
            # 奇点时刻：复杂度降低，但我们假设它的有效功能保持不变
            # 在模型中，这意味着它回到了低复杂度状态，但保留了当前的生存经验（这里简化为直接降低C）
            old_c = self.complexity
            self.complexity = max(1, int(self.complexity * (1 - efficiency)))
            self.refactored_count += 1
            return True
        return False

class SingularitySimulator(SiyanSimulator):
    """
    奇点演化模拟器
    """
    def __init__(self, enable_singularity=False, refactor_threshold=5, refactor_cost=3.0, **kwargs):
        self.enable_singularity = enable_singularity
        self.refactor_threshold = refactor_threshold
        self.refactor_cost = refactor_cost
        self.singularity_events = 0
        super().__init__(**kwargs)
        
        # 扩展历史记录
        self.history['singularity_events'] = []

    def initialize_individuals(self):
        """重写初始化，使用 SingularityIndividual"""
        target_count = int(self.grid_size * self.grid_size * self.initial_density)
        positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        random.shuffle(positions)
        
        for i in range(target_count):
            x, y = positions[i]
            # 使用新的个体类
            if self.enable_singularity:
                individual = SingularityIndividual(x, y, self.initial_complexity, self.initial_energy)
            else:
                individual = Individual(x, y, self.initial_complexity, self.initial_energy)
            self.grid[x][y] = individual
            self.individuals.append(individual)

    def simulation_step(self):
        """重写步进逻辑，加入奇点干预"""
        super().simulation_step()
        
        current_events = 0
        if self.enable_singularity:
            # 奇点干预逻辑：遍历所有个体，检查是否需要重构
            for ind in self.individuals:
                if isinstance(ind, SingularityIndividual):
                    # 如果复杂度过高，且有足够能量，触发重构
                    if ind.complexity >= self.refactor_threshold:
                        # 只有一定概率触发（技术突破不是天天有的）
                        if random.random() < 0.1: 
                            if ind.refactor(cost=self.refactor_cost):
                                current_events += 1
        
        self.singularity_events += current_events
        self.history['singularity_events'].append(current_events)

    def record_statistics(self):
        super().record_statistics()
        # 确保新字段长度一致
        if len(self.history['singularity_events']) < len(self.history['step']):
            self.history['singularity_events'].append(0)

def run_comparison_experiment():
    """运行对照实验：自然演化 vs 奇点干预"""
    print("🚀 启动 'Project Singularity' 对照实验...")
    
    common_params = {
        'grid_size': 50,
        'steps': 2000,
        'gamma': 1.8,       # 高维护成本，迫使系统崩溃
        'base_death': 0.02,
        'p_up': 0.1,        # 快速变异增加复杂度
        'output': 'singularity_comparison'
    }
    
    # 1. 对照组：自然演化 (Natural Evolution)
    print("\n[Group A] 运行自然演化组 (The Old World)...")
    sim_params = common_params.copy()
    steps = sim_params.pop('steps')
    output = sim_params.pop('output')
    
    sim_natural = SingularitySimulator(enable_singularity=False, **sim_params)
    sim_natural.run_simulation(steps)
    
    # 2. 实验组：奇点干预 (The Neuralink Future)
    print("\n[Group B] 运行奇点干预组 (The New World)...")
    sim_singularity = SingularitySimulator(
        enable_singularity=True, 
        refactor_threshold=4,   # 当复杂度达到4时就开始优化
        refactor_cost=2.0,      # 优化成本
        **sim_params
    )
    sim_singularity.run_simulation(steps)
    
    return sim_natural, sim_singularity

def plot_comparison(sim_natural, sim_singularity):
    """绘制对比图表"""
    steps = sim_natural.history['step']
    
    plt.figure(figsize=(15, 10))
    plt.style.use('dark_background') # 马斯克风格
    
    # 1. 存活率对比
    plt.subplot(2, 2, 1)
    plt.plot(steps, sim_natural.history['alive_ratio'], 'r-', label='Natural Evolution', alpha=0.8)
    plt.plot(steps, sim_singularity.history['alive_ratio'], 'c-', label='Singularity (AI/Refactor)', linewidth=2)
    plt.title('Survival Rate Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Alive Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 平均复杂度对比
    plt.subplot(2, 2, 2)
    plt.plot(steps, sim_natural.history['c_mean'], 'r--', label='Natural Complexity')
    plt.plot(steps, sim_singularity.history['c_mean'], 'c-', label='Optimized Complexity')
    plt.title('Complexity (Entropy) Growth')
    plt.xlabel('Time Step')
    plt.ylabel('Avg Complexity (C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. P*C 守恒打破情况
    plt.subplot(2, 2, 3)
    plt.plot(steps, sim_natural.history['pc_serial'], 'r--', label='Natural P*C')
    plt.plot(steps, sim_singularity.history['pc_serial'], 'c-', label='Singularity P*C')
    plt.axhline(y=1.0, color='w', linestyle=':', alpha=0.5)
    plt.title('Breaking the Conservation Law (P*C)')
    plt.xlabel('Time Step')
    plt.ylabel('P * C Product')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 奇点事件统计
    plt.subplot(2, 2, 4)
    # 确保事件列表长度一致
    events = np.array(sim_singularity.history['singularity_events'][:len(steps)])
    cumulative_events = np.cumsum(events)
    plt.plot(steps, cumulative_events, 'g-', label='Total Refactoring Events')
    plt.fill_between(steps, cumulative_events, color='g', alpha=0.2)
    plt.title('Technological Interventions (Cumulative)')
    plt.xlabel('Time Step')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'singularity_comparison_result.png'
    plt.savefig(output_path)
    print(f"\n📊 对比图表已生成: {output_path}")

if __name__ == "__main__":
    # 解决参数传递问题，适配 SiyanSimulator 的构造函数
    # 注意：这里我们假设 SiyanSimulator 接受 **kwargs 并传给 super 或 忽略多余参数
    # 如果 SiyanSimulator 定义很严格，我们需要确保参数匹配
    
    sim_nat, sim_sing = run_comparison_experiment()
    plot_comparison(sim_nat, sim_sing)
    
    # 最终结果摘要
    print("\n=== 实验结果摘要 ===")
    print(f"自然组最终存活率: {sim_nat.history['alive_ratio'][-1]:.2%}")
    print(f"奇点组最终存活率: {sim_sing.history['alive_ratio'][-1]:.2%}")
    
    print(f"自然组最终复杂度: {sim_nat.history['c_mean'][-1]:.4f}")
    print(f"奇点组最终复杂度: {sim_sing.history['c_mean'][-1]:.4f}")
    
    if sim_sing.history['alive_ratio'][-1] > sim_nat.history['alive_ratio'][-1]:
        print("\n🏆 结论: 技术奇点成功打破了递弱代偿的诅咒！")
    else:
        print("\n💀 结论: 即使有技术干预，熵增依然不可战胜。")
