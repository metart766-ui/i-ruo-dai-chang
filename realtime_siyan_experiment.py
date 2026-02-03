#!/usr/bin/env python3
"""
递弱代偿-siyan实验实时可视化版本
结合siyan实验方案的严谨性和实时可视化的直观性
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import argparse
from datetime import datetime
import json
import random
from typing import Dict, List, Tuple, Optional


class RealTimeIndividual:
    """实时可视化版本的个体类"""
    
    def __init__(self, x: int, y: int, complexity: int = 1, energy: float = 5.0):
        self.x = x
        self.y = y
        self.complexity = complexity  # 代偿度 C
        self.energy = energy
        self.alive = True
        self.age = 0
        self.color = self.get_complexity_color()
        
    def get_complexity_color(self) -> str:
        """根据复杂度返回颜色"""
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        return colors[min(self.complexity - 1, len(colors) - 1)]
    
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


class RealTimeEnvironment:
    """实时环境类"""
    
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


class RealTimeSiyanSimulator:
    """实时可视化递弱代偿模拟器"""
    
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
        
        # 初始化网格和环境
        self.grid = np.empty((grid_size, grid_size), dtype=object)
        self.environment = RealTimeEnvironment(grid_size, r_mean, r_noise, env_sigma)
        self.individuals = []
        self.step_count = 0
        
        # 初始化个体
        self.initialize_individuals()
        
        # 历史数据
        self.history = {
            'step': [],
            'alive_count': [],
            'alive_ratio': [],
            'c_mean': [],
            'p_mean_serial': [],
            'p_mean_env': [],
            'pc_serial': [],
            'pc_env': [],
            'energy_mean': [],
            'age_mean': []
        }
        
        # 可视化设置
        self.fig = None
        self.axes = None
        self.animation = None
        self.is_running = True
        self.animation_speed = 100  # 毫秒
        
    def initialize_individuals(self):
        """初始化个体"""
        target_count = int(self.grid_size * self.grid_size * self.initial_density)
        positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        random.shuffle(positions)
        
        for i in range(target_count):
            x, y = positions[i]
            individual = RealTimeIndividual(x, y, self.initial_complexity, self.initial_energy)
            self.grid[x, y] = individual
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
            if self.grid[nx, ny] is None:
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
        dead_positions = []
        
        for individual in self.individuals[:]:
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
                dead_positions.append((x, y))
                continue
            
            # 4. 环境敏感性生存判定
            death_prob = individual.get_environment_death_prob(self.base_death, self.beta, delta_e)
            if random.random() < death_prob:
                individual.alive = False
                dead_positions.append((x, y))
                continue
            
            # 5. 能量检查
            if individual.energy <= 0:
                individual.alive = False
                dead_positions.append((x, y))
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
                    new_individual = RealTimeIndividual(nx, ny, new_complexity, self.initial_energy)
                    new_individuals.append(new_individual)
                    self.grid[nx, ny] = new_individual
                    
                    # 母体消耗能量
                    individual.energy -= self.birth_energy_threshold
            
            # 更新年龄
            individual.age += 1
        
        # 清理死亡的个体
        for pos in dead_positions:
            self.grid[pos] = None
        
        # 添加新个体
        self.individuals.extend(new_individuals)
        self.individuals = [ind for ind in self.individuals if ind.alive]
        
        # 更新颜色
        for individual in self.individuals:
            individual.color = individual.get_complexity_color()
        
        # 记录统计数据
        self.record_statistics()
    
    def record_statistics(self):
        """记录统计数据"""
        if not self.individuals:
            self.history['step'].append(self.step_count)
            self.history['alive_count'].append(0)
            self.history['alive_ratio'].append(0.0)
            self.history['c_mean'].append(0.0)
            self.history['p_mean_serial'].append(0.0)
            self.history['p_mean_env'].append(0.0)
            self.history['pc_serial'].append(0.0)
            self.history['pc_env'].append(0.0)
            self.history['energy_mean'].append(0.0)
            self.history['age_mean'].append(0.0)
            return
        
        total_cells = self.grid_size * self.grid_size
        alive_count = len(self.individuals)
        alive_ratio = alive_count / total_cells
        
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
        
        # 计算平均能量和年龄
        energy_mean = np.mean([ind.energy for ind in self.individuals])
        age_mean = np.mean([ind.age for ind in self.individuals])
        
        # 记录历史
        self.history['step'].append(self.step_count)
        self.history['alive_count'].append(alive_count)
        self.history['alive_ratio'].append(alive_ratio)
        self.history['c_mean'].append(c_mean)
        self.history['p_mean_serial'].append(p_mean_serial)
        self.history['p_mean_env'].append(p_mean_env)
        self.history['pc_serial'].append(pc_serial)
        self.history['pc_env'].append(pc_env)
        self.history['energy_mean'].append(energy_mean)
        self.history['age_mean'].append(age_mean)
    
    def setup_visualization(self):
        """设置可视化"""
        plt.style.use('seaborn-v0_8')
        self.fig = plt.figure(figsize=(16, 10))
        
        # 使用GridSpec创建复杂的布局
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # 主网格显示
        self.ax_grid = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_grid.set_title('递弱代偿实时模拟 - 细胞状态', fontsize=14, fontweight='bold')
        self.ax_grid.set_xlabel('X坐标')
        self.ax_grid.set_ylabel('Y坐标')
        
        # 初始化网格显示
        self.grid_display = self.ax_grid.imshow(
            np.zeros((self.grid_size, self.grid_size, 3)), 
            interpolation='nearest', 
            vmin=0, vmax=1
        )
        
        # 存活率
        self.ax_alive = self.fig.add_subplot(gs[0, 2])
        self.line_alive, = self.ax_alive.plot([], [], 'b-', linewidth=2, label='存活率')
        self.ax_alive.set_xlabel('步骤')
        self.ax_alive.set_ylabel('存活率')
        self.ax_alive.set_title('群体存活率')
        self.ax_alive.grid(True, alpha=0.3)
        self.ax_alive.legend()
        
        # 平均复杂度
        self.ax_complexity = self.fig.add_subplot(gs[0, 3])
        self.line_complexity, = self.ax_complexity.plot([], [], 'r-', linewidth=2, label='平均复杂度')
        self.ax_complexity.set_xlabel('步骤')
        self.ax_complexity.set_ylabel('平均复杂度')
        self.ax_complexity.set_title('群体平均复杂度')
        self.ax_complexity.grid(True, alpha=0.3)
        self.ax_complexity.legend()
        
        # 鲁棒性对比
        self.ax_robustness = self.fig.add_subplot(gs[1, 2])
        self.line_serial, = self.ax_robustness.plot([], [], 'g-', linewidth=2, label='可靠性鲁棒性')
        self.line_env, = self.ax_robustness.plot([], [], 'orange', linewidth=2, label='环境鲁棒性')
        self.ax_robustness.set_xlabel('步骤')
        self.ax_robustness.set_ylabel('平均鲁棒性')
        self.ax_robustness.set_title('群体鲁棒性对比')
        self.ax_robustness.grid(True, alpha=0.3)
        self.ax_robustness.legend()
        
        # P·C乘积
        self.ax_pc = self.fig.add_subplot(gs[1, 3])
        self.line_pc_serial, = self.ax_pc.plot([], [], 'purple', linewidth=2, label='P·C (可靠性)')
        self.line_pc_env, = self.ax_pc.plot([], [], 'brown', linewidth=2, label='P·C (环境)')
        self.ax_pc.set_xlabel('步骤')
        self.ax_pc.set_ylabel('P·C 乘积')
        self.ax_pc.set_title('P·C 乘积守恒性')
        self.ax_pc.grid(True, alpha=0.3)
        self.ax_pc.legend()
        
        # 统计信息面板
        self.ax_stats = self.fig.add_subplot(gs[2, :])
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.02, 0.95, '', transform=self.ax_stats.transAxes, 
                                           fontsize=11, verticalalignment='top',
                                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 设置整体标题
        self.fig.suptitle('递弱代偿实时可视化实验 (siyan方案)', fontsize=16, fontweight='bold')
    
    def update_visualization(self, frame):
        """更新可视化"""
        if not self.is_running:
            return
        
        # 执行一步模拟
        self.simulation_step()
        
        # 更新网格显示
        grid_colors = np.zeros((self.grid_size, self.grid_size, 3))
        for individual in self.individuals:
            x, y = individual.x, individual.y
            # 将颜色字符串转换为RGB值
            color_hex = individual.color.lstrip('#')
            rgb = tuple(int(color_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            grid_colors[x, y] = rgb
        
        self.grid_display.set_array(grid_colors)
        
        # 更新历史数据
        steps = self.history['step']
        
        # 更新存活率图
        self.line_alive.set_data(steps, self.history['alive_ratio'])
        self.ax_alive.relim()
        self.ax_alive.autoscale_view()
        
        # 更新复杂度图
        self.line_complexity.set_data(steps, self.history['c_mean'])
        self.ax_complexity.relim()
        self.ax_complexity.autoscale_view()
        
        # 更新鲁棒性图
        self.line_serial.set_data(steps, self.history['p_mean_serial'])
        self.line_env.set_data(steps, self.history['p_mean_env'])
        self.ax_robustness.relim()
        self.ax_robustness.autoscale_view()
        
        # 更新P·C图
        self.line_pc_serial.set_data(steps, self.history['pc_serial'])
        self.line_pc_env.set_data(steps, self.history['pc_env'])
        self.ax_pc.relim()
        self.ax_pc.autoscale_view()
        
        # 更新统计信息
        self.update_stats_text()
        
        # 检查是否应该停止（比如所有个体都死亡）
        if not self.individuals:
            print("所有个体都已死亡，模拟结束")
            self.is_running = False
    
    def update_stats_text(self):
        """更新统计信息文本"""
        if not self.history['step']:
            return
        
        current_step = self.history['step'][-1]
        alive_count = self.history['alive_count'][-1]
        alive_ratio = self.history['alive_ratio'][-1]
        c_mean = self.history['c_mean'][-1]
        p_serial = self.history['p_mean_serial'][-1]
        p_env = self.history['p_mean_env'][-1]
        pc_serial = self.history['pc_serial'][-1]
        pc_env = self.history['pc_env'][-1]
        energy_mean = self.history['energy_mean'][-1]
        age_mean = self.history['age_mean'][-1]
        
        # 计算P·C变异系数
        if len(self.history['pc_serial']) >= 10:
            pc_serial_cv = np.std(self.history['pc_serial'][-10:]) / np.mean(self.history['pc_serial'][-10:]) if np.mean(self.history['pc_serial'][-10:]) > 0 else 0
            pc_env_cv = np.std(self.history['pc_env'][-10:]) / np.mean(self.history['pc_env'][-10:]) if np.mean(self.history['pc_env'][-10:]) > 0 else 0
        else:
            pc_serial_cv = 0
            pc_env_cv = 0
        
        stats_text = f"""
        📊 实时统计信息 (步骤 {current_step}):
        
        👥 群体状态:
          存活个体数: {alive_count}
          存活率: {alive_ratio:.3f}
          平均复杂度: {c_mean:.3f}
          平均年龄: {age_mean:.1f}
          平均能量: {energy_mean:.2f}
        
        🔬 鲁棒性分析:
          可靠性鲁棒性: {p_serial:.3f}
          环境鲁棒性: {p_env:.3f}
          复杂度-鲁棒性相关性: {np.corrcoef(self.history['c_mean'][-min(50, len(self.history['c_mean'])):], 
                                         self.history['p_mean_serial'][-min(50, len(self.history['p_mean_serial'])):])[0,1]:.3f}
        
        ⚖️ P·C守恒性:
          P·C (可靠性): {pc_serial:.3f} (变异系数: {pc_serial_cv:.3f})
          P·C (环境): {pc_env:.3f} (变异系数: {pc_env_cv:.3f})
        
        🎯 递弱代偿验证:
          {self.get_validation_status(pc_serial_cv, pc_env_cv, np.corrcoef(self.history['c_mean'][-min(50, len(self.history['c_mean'])):], 
                                                                          self.history['p_mean_serial'][-min(50, len(self.history['p_mean_serial'])):])[0,1])}
        """
        
        self.stats_text.set_text(stats_text.strip())
    
    def get_validation_status(self, pc_serial_cv: float, pc_env_cv: float, c_p_corr: float) -> str:
        """获取验证状态"""
        status = []
        
        if pc_serial_cv < 0.2:
            status.append("✅ P·C (可靠性) 高度守恒")
        elif pc_serial_cv < 0.5:
            status.append("🟡 P·C (可靠性) 中度守恒")
        else:
            status.append("❌ P·C (可靠性) 守恒性差")
            
        if pc_env_cv < 0.2:
            status.append("✅ P·C (环境) 高度守恒")
        elif pc_env_cv < 0.5:
            status.append("🟡 P·C (环境) 中度守恒")
        else:
            status.append("❌ P·C (环境) 守恒性差")
            
        if c_p_corr < -0.3:
            status.append("✅ 复杂度-鲁棒性负相关显著")
        elif c_p_corr < -0.1:
            status.append("🟡 复杂度-鲁棒性负相关较弱")
        else:
            status.append("❌ 复杂度-鲁棒性相关性不显著")
            
        return " | ".join(status)
    
    def run_realtime_simulation(self, max_steps: int = 2000):
        """运行实时模拟"""
        self.setup_visualization()
        
        def animate(frame):
            self.update_visualization(frame)
            return [self.grid_display, self.line_alive, self.line_complexity, 
                   self.line_serial, self.line_env, self.line_pc_serial, 
                   self.line_pc_env, self.stats_text]
        
        # 添加键盘控制
        def on_key(event):
            if event.key == ' ':
                self.is_running = not self.is_running
                print(f"模拟 {'继续' if self.is_running else '暂停'}")
            elif event.key == 'r':
                print("重置模拟...")
                self.__init__(**self.get_current_params())
            elif event.key == 'q':
                print("退出模拟")
                plt.close(self.fig)
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # 显示操作说明
        print("""
        🎮 操作说明:
          空格键: 暂停/继续
          R键: 重置模拟
          Q键: 退出
        """)
        
        # 开始动画
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=max_steps, 
            interval=self.animation_speed, blit=True, 
            repeat=False
        )
        
        plt.show()
    
    def get_current_params(self) -> Dict:
        """获取当前参数"""
        return {
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
            'env_sigma': self.env_sigma
        }
    
    def save_results(self, filename: str):
        """保存结果"""
        # 保存历史数据
        with open(f"{filename}_history.json", 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        
        # 保存参数
        params = self.get_current_params()
        params['total_steps'] = self.step_count
        params['final_alive_count'] = len(self.individuals)
        
        with open(f"{filename}_params.json", 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到 {filename}_history.json 和 {filename}_params.json")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='递弱代偿实时可视化实验 (siyan方案)')
    parser.add_argument('--grid', type=int, default=50, help='网格大小')
    parser.add_argument('--steps', type=int, default=2000, help='最大模拟步数')
    parser.add_argument('--speed', type=int, default=100, help='动画速度（毫秒）')
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
    parser.add_argument('--output', type=str, default='realtime_siyan', help='输出文件名前缀')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print("🧬 启动递弱代偿实时可视化实验 (siyan方案)")
    print("=" * 60)
    print(f"网格大小: {args.grid}x{args.grid}")
    print(f"最大步数: {args.steps}")
    print(f"动画速度: {args.speed}ms")
    print(f"随机种子: {args.seed}")
    
    # 创建模拟器
    simulator = RealTimeSiyanSimulator(
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
    
    try:
        # 运行实时模拟
        simulator.run_realtime_simulation(max_steps=args.steps)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        simulator.save_results(f"{args.output}_{timestamp}")
        
    except KeyboardInterrupt:
        print("\n用户中断模拟")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        simulator.save_results(f"{args.output}_interrupted_{timestamp}")
    
    except Exception as e:
        print(f"模拟出错: {str(e)}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        simulator.save_results(f"{args.output}_error_{timestamp}")


if __name__ == "__main__":
    main()