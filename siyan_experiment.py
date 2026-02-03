#!/usr/bin/env python3
"""
递弱代偿-元胞自动机实验方案实现
基于siyan实验方案，模拟"系统复杂性上升带来鲁棒性下降"的机制
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import random


class Individual:
    """个体类，包含复杂度、能量等状态"""
    
    def __init__(self, x: int, y: int, complexity: int = 1, energy: float = 5.0):
        self.x = x
        self.y = y
        self.complexity = complexity  # 代偿度 C
        self.energy = energy
        self.alive = True
        self.age = 0
        
    def get_maintenance_cost(self, base_cost: float, gamma: float) -> float:
        """计算维护成本：cost(c) = base_cost * c^gamma"""
        return base_cost * (self.complexity ** gamma)
    
    def get_resource_gain(self, base_gain: float, alpha: float, local_resource: float) -> float:
        """计算资源获取：gain(c) = base_gain * (1 + alpha * (c-1))"""
        return base_gain * (1 + alpha * (self.complexity - 1)) * local_resource
    
    def get_reliability_survival_prob(self, r: float, n0: float, n_scale: float) -> float:
        """可靠性串联近似：R = r^n，其中 n = n0 + n_scale * c"""
        n = n0 + n_scale * self.complexity
        return r ** n
    
    def get_environment_death_prob(self, base_death: float, beta: float, delta_e: float) -> float:
        """环境敏感性：death_prob = base_death + beta * c * ΔE"""
        return min(1.0, base_death + beta * self.complexity * delta_e)


class Environment:
    """环境类，管理资源场和宏观参数"""
    
    def __init__(self, grid_size: int, r_mean: float, r_noise: float, env_sigma: float):
        self.grid_size = grid_size
        self.r_mean = r_mean
        self.r_noise = r_noise
        self.env_sigma = env_sigma
        self.resource_field = np.random.normal(r_mean, r_noise, (grid_size, grid_size))
        self.macro_parameter = 0.0  # E_t
        self.prev_macro_parameter = 0.0
        
    def update(self):
        """更新环境状态"""
        # 宏观参数随机游走
        self.prev_macro_parameter = self.macro_parameter
        self.macro_parameter += np.random.normal(0, self.env_sigma)
        
        # 更新资源场（带随机扰动）
        self.resource_field = np.random.normal(self.r_mean, self.r_noise, (self.grid_size, self.grid_size))
        
    def get_delta_e(self) -> float:
        """获取环境扰动幅度 ΔE = |E_t - E_{t-1}|"""
        return abs(self.macro_parameter - self.prev_macro_parameter)
    
    def get_local_resource(self, x: int, y: int) -> float:
        """获取指定位置的资源"""
        return max(0, self.resource_field[x, y])


class SiyanSimulator:
    """递弱代偿元胞自动机模拟器"""
    
    def __init__(self, 
                 grid_size: int = 50,
                 initial_density: float = 0.3,
                 initial_complexity: int = 1,
                 initial_energy: float = 5.0,
                 alpha: float = 0.2,
                 base_cost: float = 0.3,
                 gamma: float = 1.5,
                 r: float = 0.98,
                 n0: float = 1.0,
                 n_scale: float = 0.6,
                 base_death: float = 0.01,
                 beta: float = 0.5,
                 p_up: float = 0.05,
                 p_down: float = 0.03,
                 birth_energy_threshold: float = 3.0,
                 r_mean: float = 1.0,
                 r_noise: float = 0.2,
                 env_sigma: float = 0.05):
        
        self.grid_size = grid_size
        self.initial_density = initial_density
        self.initial_complexity = initial_complexity
        self.initial_energy = initial_energy
        self.alpha = alpha
        self.base_cost = base_cost
        self.gamma = gamma
        self.r = r
        self.n0 = n0
        self.n_scale = n_scale
        self.base_death = base_death
        self.beta = beta
        self.p_up = p_up
        self.p_down = p_down
        self.birth_energy_threshold = birth_energy_threshold
        self.r_mean = r_mean
        self.r_noise = r_noise
        self.env_sigma = env_sigma
        
        # 初始化环境和个体
        self.environment = Environment(grid_size, r_mean, r_noise, env_sigma)
        self.grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        self.individuals = []
        self.step_count = 0
        
        # 初始化个体
        self.initialize_individuals()
        
        # 记录历史数据
        self.history = {
            'step': [],
            'alive_ratio': [],
            'c_mean': [],
            'p_mean_serial': [],
            'p_mean_env': [],
            'pc_serial': [],
            'pc_env': []
        }
        
    def initialize_individuals(self):
        """初始化个体"""
        target_count = int(self.grid_size * self.grid_size * self.initial_density)
        positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        random.shuffle(positions)
        
        for i in range(target_count):
            x, y = positions[i]
            individual = Individual(x, y, self.initial_complexity, self.initial_energy)
            self.grid[x][y] = individual
            self.individuals.append(individual)
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """获取邻居位置（8邻域）"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbors.append((nx, ny))
        return neighbors
    
    def get_empty_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """获取空的邻居位置"""
        empty_neighbors = []
        for nx, ny in self.get_neighbors(x, y):
            if self.grid[nx][ny] is None:
                empty_neighbors.append((nx, ny))
        return empty_neighbors
    
    def simulation_step(self):
        """执行一步模拟"""
        self.step_count += 1
        
        # 更新环境
        self.environment.update()
        delta_e = self.environment.get_delta_e()
        
        # 处理每个个体
        new_individuals = []
        dead_individuals = []
        
        for individual in self.individuals:
            if not individual.alive:
                continue
                
            x, y = individual.x, individual.y
            
            # 1. 资源分配
            local_resource = self.environment.get_local_resource(x, y)
            resource_gain = individual.get_resource_gain(1.0, self.alpha, local_resource)
            individual.energy += resource_gain
            
            # 2. 维护消耗
            maintenance_cost = individual.get_maintenance_cost(self.base_cost, self.gamma)
            individual.energy -= maintenance_cost
            
            # 3. 可靠性生存判定
            survival_prob = individual.get_reliability_survival_prob(self.r, self.n0, self.n_scale)
            if random.random() > survival_prob:
                individual.alive = False
                dead_individuals.append(individual)
                self.grid[x][y] = None
                continue
            
            # 4. 环境敏感性生存判定
            death_prob = individual.get_environment_death_prob(self.base_death, self.beta, delta_e)
            if random.random() < death_prob:
                individual.alive = False
                dead_individuals.append(individual)
                self.grid[x][y] = None
                continue
            
            # 5. 能量检查
            if individual.energy <= 0:
                individual.alive = False
                dead_individuals.append(individual)
                self.grid[x][y] = None
                continue
            
            # 6. 繁衍
            if individual.energy >= self.birth_energy_threshold:
                empty_neighbors = self.get_empty_neighbors(x, y)
                if empty_neighbors:
                    # 选择一个空位置进行繁衍
                    nx, ny = random.choice(empty_neighbors)
                    
                    # 复杂度变异
                    new_complexity = individual.complexity
                    if random.random() < self.p_up:
                        new_complexity += 1
                    elif random.random() < self.p_down:
                        new_complexity = max(1, new_complexity - 1)
                    
                    # 创建新个体
                    new_individual = Individual(nx, ny, new_complexity, self.initial_energy)
                    new_individuals.append(new_individual)
                    self.grid[nx][ny] = new_individual
                    
                    # 母体消耗能量
                    individual.energy -= self.birth_energy_threshold
            
            # 更新年龄
            individual.age += 1
        
        # 更新个体列表
        self.individuals.extend(new_individuals)
        self.individuals = [ind for ind in self.individuals if ind.alive]
        
        # 记录统计数据
        self.record_statistics()
    
    def record_statistics(self):
        """记录统计数据"""
        if not self.individuals:
            self.history['step'].append(self.step_count)
            self.history['alive_ratio'].append(0.0)
            self.history['c_mean'].append(0.0)
            self.history['p_mean_serial'].append(0.0)
            self.history['p_mean_env'].append(0.0)
            self.history['pc_serial'].append(0.0)
            self.history['pc_env'].append(0.0)
            return
        
        total_cells = self.grid_size * self.grid_size
        alive_ratio = len(self.individuals) / total_cells
        
        # 计算平均复杂度
        complexities = [ind.complexity for ind in self.individuals]
        c_mean = np.mean(complexities)
        
        # 计算平均鲁棒性（可靠性路径）
        survival_probs = [ind.get_reliability_survival_prob(self.r, self.n0, self.n_scale) 
                         for ind in self.individuals]
        p_mean_serial = np.mean(survival_probs)
        
        # 计算平均鲁棒性（环境敏感性路径）
        delta_e = self.environment.get_delta_e()
        death_probs = [1.0 - ind.get_environment_death_prob(self.base_death, self.beta, delta_e) 
                      for ind in self.individuals]
        p_mean_env = np.mean(death_probs)
        
        # 计算 P·C 乘积
        pc_serial = p_mean_serial * c_mean
        pc_env = p_mean_env * c_mean
        
        # 记录历史
        self.history['step'].append(self.step_count)
        self.history['alive_ratio'].append(alive_ratio)
        self.history['c_mean'].append(c_mean)
        self.history['p_mean_serial'].append(p_mean_serial)
        self.history['p_mean_env'].append(p_mean_env)
        self.history['pc_serial'].append(pc_serial)
        self.history['pc_env'].append(pc_env)
    
    def run_simulation(self, steps: int):
        """运行完整模拟"""
        for step in range(steps):
            self.simulation_step()
            
            # 打印进度
            if (step + 1) % 100 == 0:
                alive_ratio = self.history['alive_ratio'][-1]
                c_mean = self.history['c_mean'][-1]
                print(f"Step {step + 1}: 存活率={alive_ratio:.3f}, 平均复杂度={c_mean:.3f}")
    
    def detect_collapse(self, threshold: float = 0.1) -> bool:
        """检测是否出现崩盘"""
        if len(self.history['alive_ratio']) < 10:
            return False
        
        recent_ratios = self.history['alive_ratio'][-10:]
        return all(ratio < threshold for ratio in recent_ratios)
    
    def get_current_state(self) -> Dict:
        """获取当前状态"""
        if not self.individuals:
            return {
                'alive_ratio': 0.0,
                'c_mean': 0.0,
                'c_variance': 0.0,
                'p_mean_serial': 0.0,
                'p_variance': 0.0,
                'pc_serial': 0.0,
                'pc_env': 0.0,
                'robustness_mean': 0.0,
                'robustness_variance': 0.0,
                'energy_mean': 0.0,
                'energy_variance': 0.0,
                'birth_rate': 0.0,
                'death_rate': 0.0,
                'mutation_events': 0
            }
        
        total_cells = self.grid_size * self.grid_size
        alive_ratio = len(self.individuals) / total_cells
        
        # 计算详细统计
        complexities = [ind.complexity for ind in self.individuals]
        energies = [ind.energy for ind in self.individuals]
        ages = [ind.age for ind in self.individuals]
        
        # 计算鲁棒性
        delta_e = self.environment.get_delta_e()
        survival_probs = [1.0 - ind.get_environment_death_prob(self.base_death, self.beta, delta_e) 
                         for ind in self.individuals]
        
        # 计算出生率和死亡率（基于最近的变化）
        if len(self.history['alive_ratio']) >= 2:
            current_alive = len(self.individuals)
            prev_alive = int(self.history['alive_ratio'][-2] * total_cells) if len(self.history['alive_ratio']) >= 2 else current_alive
            birth_rate = max(0, (current_alive - prev_alive)) / total_cells
            death_rate = max(0, (prev_alive - current_alive)) / total_cells
        else:
            birth_rate = 0.0
            death_rate = 0.0
        
        return {
            'alive_ratio': alive_ratio,
            'c_mean': np.mean(complexities),
            'c_variance': np.var(complexities),
            'p_mean_serial': np.mean(survival_probs),
            'p_variance': np.var(survival_probs),
            'pc_serial': np.mean(survival_probs) * np.mean(complexities),
            'pc_env': np.mean(survival_probs) * np.mean(complexities),  # 简化处理
            'robustness_mean': np.mean(survival_probs),
            'robustness_variance': np.var(survival_probs),
            'energy_mean': np.mean(energies),
            'energy_variance': np.var(energies),
            'birth_rate': birth_rate,
            'death_rate': death_rate,
            'mutation_events': 0  # 需要在simulation_step中跟踪
        }
    
    def save_results(self, filename: str):
        """保存结果到文件"""
        df = pd.DataFrame(self.history)
        df.to_csv(f"{filename}.csv", index=False)
        
        # 保存参数信息
        params = {
            'grid_size': self.grid_size,
            'initial_density': self.initial_density,
            'initial_complexity': self.initial_complexity,
            'initial_energy': self.initial_energy,
            'alpha': self.alpha,
            'base_cost': self.base_cost,
            'gamma': self.gamma,
            'r': self.r,
            'n0': self.n0,
            'n_scale': self.n_scale,
            'base_death': self.base_death,
            'beta': self.beta,
            'p_up': self.p_up,
            'p_down': self.p_down,
            'birth_energy_threshold': self.birth_energy_threshold,
            'r_mean': self.r_mean,
            'r_noise': self.r_noise,
            'env_sigma': self.env_sigma,
            'total_steps': self.step_count,
            'final_alive_ratio': self.history['alive_ratio'][-1] if self.history['alive_ratio'] else 0,
            'final_c_mean': self.history['c_mean'][-1] if self.history['c_mean'] else 0,
            'collapse_detected': self.detect_collapse()
        }
        
        with open(f"{filename}_params.json", 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
    
    def plot_results(self):
        """绘制结果图表"""
        if not self.history['step']:
            print("没有数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 存活率
        axes[0, 0].plot(self.history['step'], self.history['alive_ratio'], 'b-', label='存活率')
        axes[0, 0].set_xlabel('步骤')
        axes[0, 0].set_ylabel('存活率')
        axes[0, 0].set_title('群体存活率变化')
        axes[0, 0].grid(True)
        
        # 平均复杂度
        axes[0, 1].plot(self.history['step'], self.history['c_mean'], 'r-', label='平均复杂度')
        axes[0, 1].set_xlabel('步骤')
        axes[0, 1].set_ylabel('平均复杂度')
        axes[0, 1].set_title('群体平均复杂度变化')
        axes[0, 1].grid(True)
        
        # 鲁棒性
        axes[1, 0].plot(self.history['step'], self.history['p_mean_serial'], 'g-', label='可靠性鲁棒性')
        axes[1, 0].plot(self.history['step'], self.history['p_mean_env'], 'orange', label='环境鲁棒性')
        axes[1, 0].set_xlabel('步骤')
        axes[1, 0].set_ylabel('平均鲁棒性')
        axes[1, 0].set_title('群体平均鲁棒性变化')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # P·C 乘积
        axes[1, 1].plot(self.history['step'], self.history['pc_serial'], 'purple', label='P·C (可靠性)')
        axes[1, 1].plot(self.history['step'], self.history['pc_env'], 'brown', label='P·C (环境)')
        axes[1, 1].set_xlabel('步骤')
        axes[1, 1].set_ylabel('P·C 乘积')
        axes[1, 1].set_title('P·C 乘积变化')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('siyan_experiment_results.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='递弱代偿元胞自动机实验')
    parser.add_argument('--grid', type=int, default=50, help='网格大小')
    parser.add_argument('--steps', type=int, default=1000, help='模拟步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--alpha', type=float, default=0.2, help='功能收益系数')
    parser.add_argument('--base_cost', type=float, default=0.3, help='基础维护成本')
    parser.add_argument('--gamma', type=float, default=1.5, help='维护成本超线性系数')
    parser.add_argument('--r', type=float, default=0.98, help='环节可靠性')
    parser.add_argument('--n0', type=float, default=1.0, help='基础依赖环节数')
    parser.add_argument('--n_scale', type=float, default=0.6, help='复杂度-环节数比例')
    parser.add_argument('--base_death', type=float, default=0.01, help='基础死亡率')
    parser.add_argument('--beta', type=float, default=0.5, help='环境敏感性系数')
    parser.add_argument('--p_up', type=float, default=0.05, help='复杂度上调概率')
    parser.add_argument('--p_down', type=float, default=0.03, help='复杂度下调概率')
    parser.add_argument('--env_sigma', type=float, default=0.05, help='环境扰动尺度')
    parser.add_argument('--output', type=str, default='siyan_results', help='输出文件名前缀')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print(f"开始递弱代偿实验...")
    print(f"网格大小: {args.grid}x{args.grid}")
    print(f"模拟步数: {args.steps}")
    print(f"随机种子: {args.seed}")
    
    # 创建模拟器
    simulator = SiyanSimulator(
        grid_size=args.grid,
        alpha=args.alpha,
        base_cost=args.base_cost,
        gamma=args.gamma,
        r=args.r,
        n0=args.n0,
        n_scale=args.n_scale,
        base_death=args.base_death,
        beta=args.beta,
        p_up=args.p_up,
        p_down=args.p_down,
        env_sigma=args.env_sigma
    )
    
    # 运行模拟
    simulator.run_simulation(args.steps)
    
    # 保存结果
    simulator.save_results(args.output)
    
    # 检测崩盘
    collapse_detected = simulator.detect_collapse()
    print(f"\n实验完成！")
    print(f"最终存活率: {simulator.history['alive_ratio'][-1]:.4f}")
    print(f"最终平均复杂度: {simulator.history['c_mean'][-1]:.4f}")
    print(f"是否检测到崩盘: {collapse_detected}")
    
    # 绘制结果
    simulator.plot_results()
    
    # 统计分析
    df = pd.DataFrame(simulator.history)
    if len(df) > 10:
        print(f"\n统计分析:")
        print(f"P·C (可靠性) 相关系数: {df['pc_serial'].corr(df['step']):.4f}")
        print(f"P·C (环境) 相关系数: {df['pc_env'].corr(df['step']):.4f}")
        print(f"复杂度-鲁棒性相关系数: {df['c_mean'].corr(df['p_mean_serial']):.4f}")


if __name__ == "__main__":
    main()