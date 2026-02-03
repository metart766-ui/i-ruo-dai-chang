import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import time
from datetime import datetime, timedelta

class DiRuoDaiChangCell:
    """元胞自动机中的单个细胞"""
    
    def __init__(self, x: int, y: int, complexity: int = 1):
        self.x = x
        self.y = y
        self.complexity = complexity  # 代偿度 C
        self.energy = 100.0
        self.age = 0
        self.alive = True
        self.color = self.get_color()
        
        # 根据复杂度计算存在度 P
        self.update_existence_degree()
        
    def get_color(self):
        """根据复杂度返回颜色"""
        # 复杂度越高，颜色越红
        intensity = min(1.0, self.complexity / 10.0)
        return (intensity, 0.2, 1.0 - intensity)
        
    def update_existence_degree(self):
        """存在度 P：系统的稳定性，与复杂度成反比"""
        alpha = 0.1
        beta = 1.5
        self.existence_degree = 1.0 / (1.0 + alpha * (self.complexity ** beta))
        self.color = self.get_color()
        
    def energy_consumption_rate(self) -> float:
        """能量消耗率：复杂度越高，维持生存所需的能量越多"""
        base_rate = 0.5
        complexity_factor = 1.0 + 0.15 * self.complexity
        return base_rate * complexity_factor
        
    def survival_probability(self, environment_stress: float) -> float:
        """生存概率：存在度越高，在环境压力下的生存概率越大"""
        base_survival = self.existence_degree
        stress_factor = 1.0 / (1.0 + environment_stress)
        complexity_vulnerability = 1.0 / (1.0 + 0.1 * self.complexity)
        return base_survival * stress_factor * complexity_vulnerability
        
    def reproduce(self, grid_size: int) -> 'DiRuoDaiChangCell':
        """繁殖：有一定概率产生更复杂的后代"""
        if random.random() < 0.08:  # 8%的突变概率
            new_complexity = self.complexity + 1
        else:
            new_complexity = self.complexity
            
        # 在相邻位置创建新细胞
        directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        dx, dy = random.choice(directions)
        new_x = (self.x + dx) % grid_size
        new_y = (self.y + dy) % grid_size
        
        return DiRuoDaiChangCell(new_x, new_y, new_complexity)
        
    def update(self, environment_stress: float):
        """更新细胞状态"""
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

class OptimizedRealTimeSimulator:
    """优化的实时模拟器 - 更好的性能"""
    
    def __init__(self, grid_size: int = 50, initial_cells: int = 100):
        self.grid_size = grid_size
        self.grid = {}
        self.time_step = 0
        self.running = False
        self.start_time = None
        self.simulation_speed = 1.0
        
        # 优化的数据结构
        self.alive_cells = set()
        self.cell_array = []  # 用于快速迭代的细胞列表
        
        # 统计信息
        self.stats_history = []
        self.max_history_length = 500  # 减少历史记录长度
        
        # 环境参数
        self.base_environment_stress = 0.03
        self.environment_variability = 0.01
        
        # 初始化细胞
        self.initialize_cells(initial_cells)
        
        # 设置图形界面
        self.setup_gui()
        
    def initialize_cells(self, count: int):
        """初始化指定数量的细胞"""
        positions = set()
        while len(positions) < count:
            x = random.randint(0, self.grid_size-1)
            y = random.randint(0, self.grid_size-1)
            positions.add((x, y))
        
        for x, y in positions:
            cell = DiRuoDaiChangCell(x, y, complexity=random.randint(1, 3))
            self.grid[(x, y)] = cell
            self.alive_cells.add((x, y))
            self.cell_array.append(cell)
            
    def get_environment_stress(self) -> float:
        """获取当前环境压力"""
        time_variation = self.environment_variability * np.sin(self.time_step * 0.02)
        random_noise = random.gauss(0, 0.005)
        return self.base_environment_stress + time_variation + random_noise
        
    def calculate_statistics(self) -> Dict:
        """计算当前统计信息 - 优化版本"""
        if not self.alive_cells:
            return {
                'total_cells': 0,
                'avg_complexity': 0,
                'avg_existence_degree': 0,
                'total_energy': 0,
                'p_times_c': 0,
                'environment_stress': self.get_environment_stress(),
                'max_complexity': 0,
                'min_existence': 1.0
            }
        
        # 使用缓存的活细胞列表进行快速计算
        alive_list = [self.grid[pos] for pos in self.alive_cells if self.grid[pos].alive]
        
        total_complexity = sum(cell.complexity for cell in alive_list)
        total_existence = sum(cell.existence_degree for cell in alive_list)
        total_energy = sum(cell.energy for cell in alive_list)
        
        n_cells = len(alive_list)
        avg_complexity = total_complexity / n_cells
        avg_existence = total_existence / n_cells
        max_complexity = max(cell.complexity for cell in alive_list)
        min_existence = min(cell.existence_degree for cell in alive_list)
        
        p_times_c = avg_existence * avg_complexity
        
        return {
            'total_cells': n_cells,
            'avg_complexity': avg_complexity,
            'avg_existence_degree': avg_existence,
            'total_energy': total_energy,
            'p_times_c': p_times_c,
            'environment_stress': self.get_environment_stress(),
            'max_complexity': max_complexity,
            'min_existence': min_existence
        }
        
    def simulation_step(self):
        """执行一个模拟步骤 - 优化版本"""
        if not self.running:
            return
            
        self.time_step += 1
        environment_stress = self.get_environment_stress()
        
        # 批量处理细胞更新
        new_positions = []
        dead_positions = []
        
        # 更新现有活细胞
        current_alive = list(self.alive_cells)  # 创建快照避免修改时迭代
        
        for pos in current_alive:
            cell = self.grid[pos]
            if not cell.alive:
                continue
                
            cell.update(environment_stress)
            
            if cell.alive:
                # 繁殖机会
                if random.random() < 0.15:
                    new_cell = cell.reproduce(self.grid_size)
                    new_pos = (new_cell.x, new_cell.y)
                    
                    if new_pos not in self.grid:
                        self.grid[new_pos] = new_cell
                        new_positions.append(new_pos)
                    elif not self.grid[new_pos].alive:
                        # 替换死亡细胞
                        self.grid[new_pos] = new_cell
                        new_positions.append(new_pos)
            else:
                dead_positions.append(pos)
        
        # 批量更新活细胞集合
        for pos in dead_positions:
            self.alive_cells.discard(pos)
        
        for pos in new_positions:
            self.alive_cells.add(pos)
        
        # 更新细胞数组
        self.cell_array = [self.grid[pos] for pos in self.alive_cells]
        
        # 记录统计信息（降低频率）
        if self.time_step % 5 == 0:  # 每5步记录一次
            stats = self.calculate_statistics()
            self.stats_history.append(stats)
            
            if len(self.stats_history) > self.max_history_length:
                self.stats_history.pop(0)
                
    def setup_gui(self):
        """设置图形用户界面 - 简化版"""
        plt.ion()  # 开启交互模式
        self.fig = plt.figure(figsize=(14, 8))
        
        # 主网格显示
        self.ax_main = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_main.set_xlim(0, self.grid_size)
        self.ax_main.set_ylim(0, self.grid_size)
        self.ax_main.set_aspect('equal')
        
        # 统计信息
        self.ax_stats = plt.subplot2grid((2, 3), (0, 2))
        self.ax_trend = plt.subplot2grid((2, 3), (1, 2))
        
        # 控制面板
        control_text = """Controls:
        Space: Start/Pause
        R: Reset
        +/-: Speed
        Q: Quit
        
        Status: Ready"""
        
        self.ax_control = plt.subplot2grid((2, 3), (0, 2))
        self.ax_control.text(0.1, 0.9, control_text, transform=self.ax_control.transAxes, 
                            fontsize=10, verticalalignment='top', fontfamily='monospace')
        self.ax_control.set_xlim(0, 1)
        self.ax_control.set_ylim(0, 1)
        self.ax_control.axis('off')
        
        # 设置键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
        
    def on_key_press(self, event):
        """处理键盘事件"""
        if event.key == ' ':
            self.running = not self.running
            status = "Running" if self.running else "Paused"
            self.update_control_panel(status)
            print(f"Simulation {status}")
            
        elif event.key.lower() == 'r':
            self.reset_simulation()
            
        elif event.key == '+' or event.key == '=':
            self.simulation_speed = min(10.0, self.simulation_speed + 0.5)
            self.update_control_panel(f"Speed: {self.simulation_speed:.1f}x")
            
        elif event.key == '-' or event.key == '_':
            self.simulation_speed = max(0.1, self.simulation_speed - 0.5)
            self.update_control_panel(f"Speed: {self.simulation_speed:.1f}x")
            
        elif event.key.lower() == 'q':
            print("Quitting simulation...")
            self.running = False
            plt.close(self.fig)
            
    def update_control_panel(self, status_text):
        """更新控制面板"""
        self.ax_control.clear()
        control_text = f"""Controls:
        Space: Start/Pause
        R: Reset
        +/-: Speed
        Q: Quit
        
        Status: {status_text}"""
        
        self.ax_control.text(0.1, 0.9, control_text, transform=self.ax_control.transAxes, 
                            fontsize=10, verticalalignment='top', fontfamily='monospace')
        self.ax_control.set_xlim(0, 1)
        self.ax_control.set_ylim(0, 1)
        self.ax_control.axis('off')
        
    def reset_simulation(self):
        """重置模拟"""
        self.running = False
        self.time_step = 0
        self.grid.clear()
        self.alive_cells.clear()
        self.cell_array.clear()
        self.stats_history.clear()
        self.start_time = None
        self.initialize_cells(100)
        self.update_control_panel("Reset Complete")
        print("Simulation reset")
        
    def update_display(self):
        """更新显示 - 优化版本"""
        if not plt.fignum_exists(self.fig.number):
            return False
            
        # 执行模拟步骤
        steps_per_update = max(1, int(self.simulation_speed))
        for _ in range(steps_per_update):
            self.simulation_step()
        
        # 清除之前的显示
        self.ax_main.clear()
        self.ax_stats.clear()
        self.ax_trend.clear()
        
        # 绘制细胞 - 使用散点图提高性能
        if self.alive_cells:
            positions = list(self.alive_cells)
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # 获取颜色和大小
            colors = []
            sizes = []
            for pos in positions:
                cell = self.grid[pos]
                colors.append(cell.color)
                sizes.append(max(20, 100 * cell.existence_degree))
            
            self.ax_main.scatter(x_coords, y_coords, c=colors, s=sizes, 
                               alpha=0.8, edgecolors='black', linewidth=0.3)
        
        # 设置主图属性
        self.ax_main.set_xlim(0, self.grid_size)
        self.ax_main.set_ylim(0, self.grid_size)
        self.ax_main.set_aspect('equal')
        
        # 计算运行时间
        elapsed_time = "00:00:00"
        if self.start_time and self.running:
            elapsed = datetime.now() - self.start_time
            elapsed_time = str(elapsed).split('.')[0]
            
            # 检查24小时限制
            if elapsed >= timedelta(hours=24):
                print("24小时模拟完成！")
                self.running = False
        
        self.ax_main.set_title(f'DiRuoDaiChang Simulation\nStep: {self.time_step} | Time: {elapsed_time}', 
                              fontsize=12, fontweight='bold')
        
        # 显示统计信息
        stats = self.calculate_statistics()
        
        # 文本统计
        stats_text = f"""Live Cells: {stats['total_cells']}
Avg Complexity: {stats['avg_complexity']:.2f}
Avg Existence: {stats['avg_existence_degree']:.3f}
P×C Product: {stats['p_times_c']:.3f}
Max Complexity: {stats['max_complexity']}
Min Existence: {stats['min_existence']:.3f}
Environment: {stats['environment_stress']:.3f}"""
        
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes, 
                          fontsize=10, verticalalignment='top', fontfamily='monospace')
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        
        # 趋势图（简化）
        if len(self.stats_history) > 10:
            time_steps = range(len(self.stats_history))
            cell_counts = [h['total_cells'] for h in self.stats_history]
            
            self.ax_trend.plot(time_steps, cell_counts, 'b-', linewidth=2, label='Cells')
            self.ax_trend.set_title('Cell Count Trend')
            self.ax_trend.set_ylabel('Count')
            self.ax_trend.grid(True, alpha=0.3)
            self.ax_trend.legend()
        else:
            self.ax_trend.text(0.5, 0.5, 'Collecting data...', 
                               transform=self.ax_trend.transAxes, 
                               ha='center', va='center', fontsize=12)
            self.ax_trend.axis('off')
        
        plt.tight_layout()
        plt.pause(0.001)  # 短暂暂停以更新显示
        
        return True
        
    def run(self):
        """运行实时模拟"""
        print("=== DiRuoDaiChang 24-Hour Real-time Simulation ===")
        print("Controls:")
        print("  Space: Start/Pause simulation")
        print("  R: Reset simulation")
        print("  +/-: Adjust speed")
        print("  Q: Quit")
        print("\nStarting with 100 cells...")
        
        self.start_time = datetime.now()
        
        try:
            while plt.fignum_exists(self.fig.number):
                if not self.update_display():
                    break
                    
                # 控制更新频率
                time.sleep(0.05)  # 20 FPS
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            print("Simulation ended")
            plt.close('all')

if __name__ == "__main__":
    # 创建并运行优化的实时模拟器
    simulator = OptimizedRealTimeSimulator(grid_size=50, initial_cells=100)
    simulator.run()