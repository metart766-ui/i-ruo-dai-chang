import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from typing import List, Dict, Tuple

class DiRuoDaiChangCell:
    """
    元胞自动机中的单个细胞
    模拟递弱代偿理论中的基本单位
    """
    
    def __init__(self, x: int, y: int, complexity: int = 1):
        self.x = x
        self.y = y
        self.complexity = complexity  # 代偿度 C：系统复杂度
        self.energy = 100.0  # 能量储备
        self.age = 0
        self.alive = True
        self.mutation_rate = 0.01
        
        # 根据复杂度计算存在度 P
        self.update_existence_degree()
        
    def update_existence_degree(self):
        """
        存在度 P：系统的稳定性，与复杂度成反比
        P = 1 / (1 + alpha * C^beta)
        其中 alpha 和 beta 是调节参数
        """
        alpha = 0.1
        beta = 1.5
        self.existence_degree = 1.0 / (1.0 + alpha * (self.complexity ** beta))
        
    def energy_consumption_rate(self) -> float:
        """
        能量消耗率：复杂度越高，维持生存所需的能量越多
        """
        base_rate = 0.5
        complexity_factor = 1.0 + 0.2 * self.complexity
        return base_rate * complexity_factor
        
    def survival_probability(self, environment_stress: float) -> float:
        """
        生存概率：存在度越高，在环境压力下的生存概率越大
        """
        # 基础生存概率由存在度决定
        base_survival = self.existence_degree
        
        # 环境压力会降低生存概率
        stress_factor = 1.0 / (1.0 + environment_stress)
        
        # 复杂度高的系统在面对环境变化时更脆弱
        complexity_vulnerability = 1.0 / (1.0 + 0.1 * self.complexity)
        
        return base_survival * stress_factor * complexity_vulnerability
        
    def reproduce(self) -> 'DiRuoDaiChangCell':
        """
        繁殖：有一定概率产生更复杂的后代（代偿增加）
        """
        if random.random() < 0.1:  # 10%的突变概率
            new_complexity = self.complexity + 1
        else:
            new_complexity = self.complexity
            
        # 在相邻位置创建新细胞
        dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        new_x = (self.x + dx) % 50  # 假设网格大小为50x50
        new_y = (self.y + dy) % 50
        
        return DiRuoDaiChangCell(new_x, new_y, new_complexity)
        
    def update(self, environment_stress: float):
        """
        更新细胞状态
        """
        if not self.alive:
            return
            
        self.age += 1
        
        # 消耗能量
        self.energy -= self.energy_consumption_rate()
        
        # 检查生存概率
        survival_prob = self.survival_probability(environment_stress)
        
        if random.random() > survival_prob or self.energy <= 0:
            self.alive = False
            return
            
        # 更新存在度
        self.update_existence_degree()

class DiRuoDaiChangSimulation:
    """
    递弱代偿元胞自动机模拟器
    """
    
    def __init__(self, width: int = 50, height: int = 50):
        self.width = width
        self.height = height
        self.grid = {}
        self.time_step = 0
        self.history = []
        
        # 环境参数
        self.base_environment_stress = 0.05  # 降低基础环境压力
        self.environment_variability = 0.02  # 降低环境变化
        
        # 初始化一些简单细胞
        for i in range(20):  # 增加初始细胞数量
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            cell = DiRuoDaiChangCell(x, y, complexity=1)
            self.grid[(x, y)] = cell
            
    def get_environment_stress(self) -> float:
        """
        获取当前环境压力
        """
        # 环境压力随时间变化，模拟环境的不确定性
        time_variation = self.environment_variability * np.sin(self.time_step * 0.05)
        return self.base_environment_stress + time_variation + random.gauss(0, 0.01)
        
    def calculate_system_metrics(self) -> Dict:
        """
        计算系统整体指标
        """
        alive_cells = [cell for cell in self.grid.values() if cell.alive]
        
        if not alive_cells:
            return {
                'total_cells': 0,
                'avg_complexity': 0,
                'avg_existence_degree': 0,
                'total_energy': 0,
                'p_times_c': 0,  # P * C 的值
                'environment_stress': self.get_environment_stress()
            }
            
        total_complexity = sum(cell.complexity for cell in alive_cells)
        total_existence = sum(cell.existence_degree for cell in alive_cells)
        total_energy = sum(cell.energy for cell in alive_cells)
        
        avg_complexity = total_complexity / len(alive_cells)
        avg_existence = total_existence / len(alive_cells)
        
        # 计算 P * C
        p_times_c = avg_existence * avg_complexity
        
        return {
            'total_cells': len(alive_cells),
            'avg_complexity': avg_complexity,
            'avg_existence_degree': avg_existence,
            'total_energy': total_energy,
            'p_times_c': p_times_c,
            'environment_stress': self.get_environment_stress()
        }
        
    def step(self):
        """
        执行一个时间步的模拟
        """
        self.time_step += 1
        environment_stress = self.get_environment_stress()
        
        # 更新所有细胞
        new_cells = []
        cells_to_remove = []
        
        for pos, cell in self.grid.items():
            if cell.alive:
                cell.update(environment_stress)
                
                # 存活的细胞有机会繁殖
                if cell.alive and random.random() < 0.2:  # 降低繁殖概率到20%
                    new_cell = cell.reproduce()
                    new_pos = (new_cell.x, new_cell.y)
                    if new_pos not in self.grid:
                        new_cells.append(new_cell)
                        
                if not cell.alive:
                    cells_to_remove.append(pos)
                    
        # 移除死亡的细胞
        for pos in cells_to_remove:
            del self.grid[pos]
            
        # 添加新细胞
        for cell in new_cells:
            self.grid[(cell.x, cell.y)] = cell
            
        # 记录历史数据
        metrics = self.calculate_system_metrics()
        self.history.append(metrics)
        
    def run_simulation(self, steps: int = 1000):
        """
        运行完整模拟
        """
        for _ in range(steps):
            self.step()
            
            # 如果所有细胞都死亡，提前结束
            if self.calculate_system_metrics()['total_cells'] == 0:
                print(f"所有细胞在第 {self.time_step} 步死亡")
                break
                
    def plot_results(self):
        """
        绘制模拟结果
        """
        if not self.history:
            print("没有历史数据可绘制")
            return
            
        time_steps = range(len(self.history))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('DiRuoDaiChang Cellular Automata Simulation Results', fontsize=16)
        
        # 细胞数量变化
        axes[0,0].plot(time_steps, [h['total_cells'] for h in self.history], 'b-', linewidth=2)
        axes[0,0].set_title('Number of Living Cells')
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('Cell Count')
        axes[0,0].grid(True, alpha=0.3)
        
        # 平均复杂度变化（代偿度）
        axes[0,1].plot(time_steps, [h['avg_complexity'] for h in self.history], 'r-', linewidth=2)
        axes[0,1].set_title('Average Complexity (Compensation C)')
        axes[0,1].set_xlabel('Time Step')
        axes[0,1].set_ylabel('Complexity')
        axes[0,1].grid(True, alpha=0.3)
        
        # 平均存在度变化
        axes[0,2].plot(time_steps, [h['avg_existence_degree'] for h in self.history], 'g-', linewidth=2)
        axes[0,2].set_title('Average Existence Degree (P)')
        axes[0,2].set_xlabel('Time Step')
        axes[0,2].set_ylabel('Existence Degree')
        axes[0,2].grid(True, alpha=0.3)
        
        # P * C 乘积
        axes[1,0].plot(time_steps, [h['p_times_c'] for h in self.history], 'm-', linewidth=2, label='P × C')
        axes[1,0].set_title('P × C Product')
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel('P × C')
        axes[1,0].axhline(y=np.mean([h['p_times_c'] for h in self.history]), 
                         color='k', linestyle='--', alpha=0.7, label='Mean')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 总能量变化
        axes[1,1].plot(time_steps, [h['total_energy'] for h in self.history], 'c-', linewidth=2)
        axes[1,1].set_title('Total System Energy')
        axes[1,1].set_xlabel('Time Step')
        axes[1,1].set_ylabel('Energy')
        axes[1,1].grid(True, alpha=0.3)
        
        # 环境压力
        axes[1,2].plot(time_steps, [h['environment_stress'] for h in self.history], 'orange', linewidth=2)
        axes[1,2].set_title('Environmental Stress')
        axes[1,2].set_xlabel('Time Step')
        axes[1,2].set_ylabel('Stress Level')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('di_ruo_dai_chang_simulation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def print_final_statistics(self):
        """
        打印最终统计信息
        """
        if not self.history:
            return
            
        final_metrics = self.history[-1]
        initial_metrics = self.history[0]
        
        print("\n=== DiRuoDaiChang Simulation Final Statistics ===")
        print(f"Simulation Steps: {len(self.history)}")
        print(f"Initial Cell Count: {initial_metrics['total_cells']}")
        print(f"Final Cell Count: {final_metrics['total_cells']}")
        print(f"Initial Avg Complexity: {initial_metrics['avg_complexity']:.3f}")
        print(f"Final Avg Complexity: {final_metrics['avg_complexity']:.3f}")
        print(f"Initial Avg Existence: {initial_metrics['avg_existence_degree']:.3f}")
        print(f"Final Avg Existence: {final_metrics['avg_existence_degree']:.3f}")
        
        # 计算 P * C 的稳定性
        pc_values = [h['p_times_c'] for h in self.history if h['total_cells'] > 0]
        if pc_values:
            pc_mean = np.mean(pc_values)
            pc_std = np.std(pc_values)
            print(f"P × C Mean: {pc_mean:.3f}")
            print(f"P × C Std Dev: {pc_std:.3f}")
            print(f"P × C Coefficient of Variation: {pc_std/pc_mean:.3f}")
            
            # 分析递弱代偿趋势
            complexity_trend = final_metrics['avg_complexity'] - initial_metrics['avg_complexity']
            existence_trend = final_metrics['avg_existence_degree'] - initial_metrics['avg_existence_degree']
            
            print(f"\n=== DiRuoDaiChang Trend Analysis ===")
            print(f"Complexity Trend: {'+' if complexity_trend > 0 else ''}{complexity_trend:.3f}")
            print(f"Existence Trend: {'+' if existence_trend > 0 else ''}{existence_trend:.3f}")
            
            if complexity_trend > 0 and existence_trend < 0:
                print("✓ Observed DiRuoDaiChang pattern: Increasing complexity, decreasing existence")
            elif complexity_trend > 0:
                print("Partial DiRuoDaiChang pattern: Increasing complexity")
            elif existence_trend < 0:
                print("Partial DiRuoDaiChang pattern: Decreasing existence")
            else:
                print("No clear DiRuoDaiChang pattern observed")

class EnhancedDiRuoDaiChangSimulation(DiRuoDaiChangSimulation):
    """
    增强版模拟器，包含更多参数和实验选项
    """
    
    def __init__(self, width: int = 50, height: int = 50, 
                 adaptation_rate: float = 0.1,
                 environmental_chaos: float = 0.1):
        super().__init__(width, height)
        self.adaptation_rate = adaptation_rate
        self.environmental_chaos = environmental_chaos
        
    def get_environment_stress(self) -> float:
        """
        更复杂的环境压力模型
        """
        # 基础周期性变化
        periodic = self.environment_variability * np.sin(self.time_step * 0.05)
        
        # 随机混沌成分
        chaotic = self.environmental_chaos * random.gauss(0, 1)
        
        # 灾难性事件（小概率）
        catastrophe = 0
        if random.random() < 0.001:  # 0.1%概率的大灾难
            catastrophe = random.uniform(0.5, 2.0)
            print(f"Catastrophe at step {self.time_step}: stress += {catastrophe:.2f}")
            
        return self.base_environment_stress + periodic + chaotic + catastrophe

def run_multiple_experiments(n_experiments: int = 5, steps: int = 200):
    """
    运行多个实验，统计分析结果
    """
    results = []
    
    print(f"Running {n_experiments} experiments...")
    
    for i in range(n_experiments):
        print(f"Experiment {i+1}/{n_experiments}")
        sim = EnhancedDiRuoDaiChangSimulation(
            width=30, height=30,
            adaptation_rate=0.1 + 0.05 * i,
            environmental_chaos=0.05 + 0.02 * i
        )
        sim.run_simulation(steps)
        
        # 收集关键指标
        if sim.history:
            final_metrics = sim.history[-1]
            initial_metrics = sim.history[0]
            
            results.append({
                'experiment': i+1,
                'adaptation_rate': sim.adaptation_rate,
                'environmental_chaos': sim.environmental_chaos,
                'initial_complexity': initial_metrics['avg_complexity'],
                'final_complexity': final_metrics['avg_complexity'],
                'initial_existence': initial_metrics['avg_existence_degree'],
                'final_existence': final_metrics['avg_existence_degree'],
                'pc_stability': np.std([h['p_times_c'] for h in sim.history if h['total_cells'] > 0]),
                'survival_time': len(sim.history)
            })
    
    # 分析结果
    print("\n=== Multi-Experiment Analysis ===")
    df = pd.DataFrame(results)
    if not df.empty:
        print(f"Average complexity increase: {(df['final_complexity'] - df['initial_complexity']).mean():.3f}")
        print(f"Average existence decrease: {(df['final_existence'] - df['initial_existence']).mean():.3f}")
        print(f"Average P×C stability: {df['pc_stability'].mean():.3f}")
        print(f"Average survival time: {df['survival_time'].mean():.1f} steps")
    
    return results

if __name__ == "__main__":
    # 创建模拟器并运行
    print("Starting DiRuoDaiChang Cellular Automata Simulation...")
    
    # 基础模拟
    sim = DiRuoDaiChangSimulation(width=30, height=30)
    sim.run_simulation(steps=300)
    sim.plot_results()
    sim.print_final_statistics()
    
    # 运行多实验分析
    try:
        import pandas as pd
        results = run_multiple_experiments(n_experiments=3, steps=150)
    except ImportError:
        print("pandas not available, skipping multi-experiment analysis")
    
    print("\nSimulation completed! Results saved as 'di_ruo_dai_chang_simulation.png'")
    
    # 理论分析
    print("\n=== Theoretical Analysis ===")
    print("This simulation attempts to model Wang Dongyue's DiRuoDaiChang theory:")
    print("1. Systems evolve towards higher complexity (compensation)")
    print("2. Higher complexity leads to lower stability (weakened existence)")
    print("3. The product P × C should remain relatively constant")
    print("4. Energy requirements increase with complexity")
    print("5. Environmental stress disproportionately affects complex systems")