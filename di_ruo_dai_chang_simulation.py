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
        self.base_environment_stress = 0.1
        self.environment_variability = 0.05
        
        # 初始化一些简单细胞
        for i in range(10):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            cell = DiRuoDaiChangCell(x, y, complexity=1)
            self.grid[(x, y)] = cell
            
    def get_environment_stress(self) -> float:
        """
        获取当前环境压力
        """
        # 环境压力随时间变化，模拟环境的不确定性
        time_variation = self.environment_variability * np.sin(self.time_step * 0.1)
        return self.base_environment_stress + time_variation + random.gauss(0, 0.02)
        
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
                if cell.alive and random.random() < 0.3:  # 30%繁殖概率
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
        fig.suptitle('递弱代偿理论元胞自动机模拟结果', fontsize=16)
        
        # 细胞数量变化
        axes[0,0].plot(time_steps, [h['total_cells'] for h in self.history])
        axes[0,0].set_title('存活细胞数量')
        axes[0,0].set_xlabel('时间步')
        axes[0,0].set_ylabel('细胞数量')
        
        # 平均复杂度变化（代偿度）
        axes[0,1].plot(time_steps, [h['avg_complexity'] for h in self.history], 'r-')
        axes[0,1].set_title('平均复杂度 (代偿度 C)')
        axes[0,1].set_xlabel('时间步')
        axes[0,1].set_ylabel('复杂度')
        
        # 平均存在度变化
        axes[0,2].plot(time_steps, [h['avg_existence_degree'] for h in self.history], 'g-')
        axes[0,2].set_title('平均存在度 (P)')
        axes[0,2].set_xlabel('时间步')
        axes[0,2].set_ylabel('存在度')
        
        # P * C 乘积
        axes[1,0].plot(time_steps, [h['p_times_c'] for h in self.history], 'm-')
        axes[1,0].set_title('P × C 乘积')
        axes[1,0].set_xlabel('时间步')
        axes[1,0].set_ylabel('P × C')
        axes[1,0].axhline(y=np.mean([h['p_times_c'] for h in self.history]), 
                         color='k', linestyle='--', alpha=0.7, label='平均值')
        axes[1,0].legend()
        
        # 总能量变化
        axes[1,1].plot(time_steps, [h['total_energy'] for h in self.history], 'c-')
        axes[1,1].set_title('系统总能量')
        axes[1,1].set_xlabel('时间步')
        axes[1,1].set_ylabel('能量')
        
        # 环境压力
        axes[1,2].plot(time_steps, [h['environment_stress'] for h in self.history], 'orange')
        axes[1,2].set_title('环境压力')
        axes[1,2].set_xlabel('时间步')
        axes[1,2].set_ylabel('压力值')
        
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
        
        print("\n=== 递弱代偿模拟最终统计 ===")
        print(f"模拟时间步: {len(self.history)}")
        print(f"初始细胞数量: {initial_metrics['total_cells']}")
        print(f"最终细胞数量: {final_metrics['total_cells']}")
        print(f"初始平均复杂度: {initial_metrics['avg_complexity']:.3f}")
        print(f"最终平均复杂度: {final_metrics['avg_complexity']:.3f}")
        print(f"初始平均存在度: {initial_metrics['avg_existence_degree']:.3f}")
        print(f"最终平均存在度: {final_metrics['avg_existence_degree']:.3f}")
        
        # 计算 P * C 的稳定性
        pc_values = [h['p_times_c'] for h in self.history if h['total_cells'] > 0]
        if pc_values:
            pc_mean = np.mean(pc_values)
            pc_std = np.std(pc_values)
            print(f"P × C 平均值: {pc_mean:.3f}")
            print(f"P × C 标准差: {pc_std:.3f}")
            print(f"P × C 变异系数: {pc_std/pc_mean:.3f}")

if __name__ == "__main__":
    # 创建模拟器并运行
    sim = DiRuoDaiChangSimulation(width=50, height=50)
    print("开始递弱代偿元胞自动机模拟...")
    
    sim.run_simulation(steps=500)
    sim.plot_results()
    sim.print_final_statistics()
    
    print("\n模拟完成！结果已保存为 'di_ruo_dai_chang_simulation.png'")