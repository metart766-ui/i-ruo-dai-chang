import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import random
from datetime import datetime, timedelta
import time
import threading
from collections import defaultdict

class RealTimeDiRuoDaiChangCell:
    """实时递弱代偿理论中的单个细胞"""
    
    def __init__(self, x: int, y: int, complexity: int = 1):
        self.x = x
        self.y = y
        self.complexity = complexity  # 代偿度 C
        self.energy = 100.0
        self.age = 0
        self.alive = True
        self.color = self.get_complexity_color()
        self.existence_degree = 1.0  # 初始存在度
        self.update_existence_degree()  # 根据复杂度更新存在度
        
    def get_complexity_color(self):
        """根据复杂度返回颜色"""
        # 复杂度越高，颜色越红
        intensity = min(1.0, self.complexity / 10.0)
        return (intensity, 0.2, 1.0 - intensity)
        
    def update_existence_degree(self):
        """更新存在度 P = 1 / (1 + α * C^β)"""
        alpha = 0.08  # 稍微降低alpha值，让系统更稳定
        beta = 1.3
        self.existence_degree = 1.0 / (1.0 + alpha * (self.complexity ** beta))
        self.color = self.get_complexity_color()
        
    def energy_consumption_rate(self) -> float:
        """能量消耗率"""
        base_rate = 0.3  # 降低基础消耗率
        complexity_factor = 1.0 + 0.15 * self.complexity
        return base_rate * complexity_factor
        
    def survival_probability(self, environment_stress: float) -> float:
        """生存概率"""
        base_survival = self.existence_degree
        stress_factor = 1.0 / (1.0 + environment_stress)
        complexity_vulnerability = 1.0 / (1.0 + 0.08 * self.complexity)
        return base_survival * stress_factor * complexity_vulnerability
        
    def reproduce(self, grid_size: int) -> 'RealTimeDiRuoDaiChangCell':
        """繁殖"""
        if random.random() < 0.08:  # 降低突变概率到8%
            new_complexity = self.complexity + 1
        else:
            new_complexity = self.complexity
            
        # 在相邻位置创建新细胞
        directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        dx, dy = random.choice(directions)
        new_x = (self.x + dx) % grid_size
        new_y = (self.y + dy) % grid_size
        
        return RealTimeDiRuoDaiChangCell(new_x, new_y, new_complexity)
        
    def update(self, environment_stress: float) -> bool:
        """更新细胞状态，返回是否存活"""
        if not self.alive:
            return False
            
        self.age += 1
        self.energy -= self.energy_consumption_rate()
        
        survival_prob = self.survival_probability(environment_stress)
        
        if random.random() > survival_prob or self.energy <= 0:
            self.alive = False
            return False
            
        self.update_existence_degree()
        return True

class RealTimeDiRuoDaiChangSimulator:
    """实时递弱代偿模拟器"""
    
    def __init__(self, grid_size: int = 50, initial_cells: int = 100):
        self.grid_size = grid_size
        self.grid = {}
        self.time_step = 0
        self.history = []
        self.max_history_length = 1000
        
        # 环境参数
        self.base_environment_stress = 0.03  # 进一步降低基础压力
        self.environment_variability = 0.015
        self.catastrophe_probability = 0.0005  # 降低灾难概率
        
        # 统计信息
        self.total_births = 0
        self.total_deaths = 0
        self.max_complexity_reached = 1
        
        # 时间控制
        self.start_time = datetime.now()
        self.target_duration = timedelta(hours=24)  # 24小时
        self.is_running = True
        self.paused = False
        
        # 初始化细胞
        self.initialize_cells(initial_cells)
        
    def initialize_cells(self, count: int):
        """初始化细胞"""
        positions = set()
        while len(positions) < count:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            positions.add((x, y))
            
        for x, y in positions:
            cell = RealTimeDiRuoDaiChangCell(x, y, complexity=1)
            self.grid[(x, y)] = cell
            
    def get_environment_stress(self) -> float:
        """获取环境压力"""
        periodic = self.environment_variability * np.sin(self.time_step * 0.03)
        chaotic = 0.01 * random.gauss(0, 1)
        
        catastrophe = 0
        if random.random() < self.catastrophe_probability:
            catastrophe = random.uniform(0.3, 1.0)
            print(f"🌪️  灾难事件！时间步 {self.time_step}: 压力增加 {catastrophe:.2f}")
            
        return self.base_environment_stress + periodic + chaotic + catastrophe
        
    def calculate_statistics(self) -> dict:
        """计算统计信息"""
        alive_cells = [cell for cell in self.grid.values() if cell.alive]
        
        if not alive_cells:
            return {
                'alive_count': 0,
                'avg_complexity': 0,
                'avg_existence': 0,
                'total_energy': 0,
                'pc_product': 0,
                'environment_stress': self.get_environment_stress()
            }
            
        total_complexity = sum(cell.complexity for cell in alive_cells)
        total_existence = sum(cell.existence_degree for cell in alive_cells)
        total_energy = sum(cell.energy for cell in alive_cells)
        
        avg_complexity = total_complexity / len(alive_cells)
        avg_existence = total_existence / len(alive_cells)
        pc_product = avg_existence * avg_complexity
        
        # 更新最大复杂度
        max_complexity = max(cell.complexity for cell in alive_cells)
        self.max_complexity_reached = max(self.max_complexity_reached, max_complexity)
        
        return {
            'alive_count': len(alive_cells),
            'avg_complexity': avg_complexity,
            'avg_existence': avg_existence,
            'total_energy': total_energy,
            'pc_product': pc_product,
            'environment_stress': self.get_environment_stress(),
            'max_complexity': max_complexity
        }
        
    def simulation_step(self):
        """执行一个模拟步"""
        if not self.is_running or self.paused:
            return
            
        self.time_step += 1
        environment_stress = self.get_environment_stress()
        
        # 更新所有细胞
        dead_positions = []
        new_cells = []
        
        for pos, cell in list(self.grid.items()):
            if cell.alive:
                survived = cell.update(environment_stress)
                if not survived:
                    dead_positions.append(pos)
                    self.total_deaths += 1
                else:
                    # 繁殖机会
                    if random.random() < 0.15:  # 15%繁殖概率
                        new_cell = cell.reproduce(self.grid_size)
                        if new_cell:
                            new_pos = (new_cell.x, new_cell.y)
                            if new_pos not in self.grid:
                                new_cells.append((new_pos, new_cell))
                                self.total_births += 1
                                
        # 移除死亡细胞
        for pos in dead_positions:
            if pos in self.grid:
                del self.grid[pos]
                
        # 添加新细胞
        for pos, cell in new_cells:
            self.grid[pos] = cell
            
        # 记录统计
        stats = self.calculate_statistics()
        self.history.append(stats)
        
        # 限制历史长度
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]
            
        # 检查是否达到24小时
        elapsed = datetime.now() - self.start_time
        if elapsed >= self.target_duration:
            print(f"\n🎉 24小时模拟完成！")
            print(f"总时间步: {self.time_step:,}")
            print(f"总出生: {self.total_births:,}")
            print(f"总死亡: {self.total_deaths:,}")
            print(f"达到的最大复杂度: {self.max_complexity_reached}")
            self.is_running = False
            self.print_final_report()
            
    def print_final_report(self):
        """打印最终报告"""
        if not self.history:
            return
            
        final_stats = self.history[-1]
        initial_stats = self.history[0] if len(self.history) > 1 else final_stats
        
        print("\n" + "="*60)
        print("🏁 24小时递弱代偿模拟最终报告")
        print("="*60)
        print(f"总运行时间: {datetime.now() - self.start_time}")
        print(f"总模拟步数: {self.time_step:,}")
        print(f"最终存活细胞: {final_stats['alive_count']:,}")
        print(f"总出生细胞: {self.total_births:,}")
        print(f"总死亡细胞: {self.total_deaths:,}")
        print(f"达到的最大复杂度: {self.max_complexity_reached}")
        
        if len(self.history) > 1:
            complexity_change = final_stats['avg_complexity'] - initial_stats['avg_complexity']
            existence_change = final_stats['avg_existence'] - initial_stats['avg_existence']
            
            print(f"\n复杂度变化: {complexity_change:+.3f}")
            print(f"存在度变化: {existence_change:+.3f}")
            
            if complexity_change > 0 and existence_change < 0:
                print("✅ 观察到递弱代偿模式：复杂度增加，存在度降低")
            elif complexity_change > 0:
                print("△ 复杂度增加趋势")
            elif existence_change < 0:
                print("▽ 存在度降低趋势")
                
        # P×C守恒分析
        pc_values = [h['pc_product'] for h in self.history if h['alive_count'] > 0]
        if pc_values:
            pc_mean = np.mean(pc_values)
            pc_std = np.std(pc_values)
            pc_cv = pc_std / pc_mean if pc_mean > 0 else 0
            
            print(f"\nP×C守恒分析:")
            print(f"  平均值: {pc_mean:.3f}")
            print(f"  标准差: {pc_std:.3f}")
            print(f"  变异系数: {pc_cv:.3f}")
            
            if pc_cv < 0.1:
                print("✅ P×C乘积高度稳定，支持守恒假设")
            elif pc_cv < 0.2:
                print("△ P×C乘积相对稳定")
            else:
                print("▽ P×C乘积波动较大")
                
        print("="*60)

class RealTimeVisualizer:
    """实时可视化器"""
    
    def __init__(self, simulator: RealTimeDiRuoDaiChangSimulator):
        self.simulator = simulator
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('递弱代偿理论 - 24小时实时模拟')
        
        # 创建子图
        self.grid_ax = plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=2)
        self.complexity_ax = plt.subplot2grid((3, 4), (0, 2), colspan=2)
        self.existence_ax = plt.subplot2grid((3, 4), (1, 2), colspan=2)
        self.pc_ax = plt.subplot2grid((3, 4), (2, 0), colspan=2)
        self.info_ax = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        
        self.setup_plots()
        
    def setup_plots(self):
        """设置图形"""
        # 网格图
        self.grid_ax.set_xlim(0, self.simulator.grid_size)
        self.grid_ax.set_ylim(0, self.simulator.grid_size)
        self.grid_ax.set_aspect('equal')
        self.grid_ax.set_title('细胞活动实时画面', fontsize=14, fontweight='bold')
        self.grid_ax.set_xticks([])
        self.grid_ax.set_yticks([])
        
        # 复杂度图
        self.complexity_ax.set_title('平均复杂度 (代偿度 C)', fontsize=12, fontweight='bold')
        self.complexity_ax.set_xlabel('时间步')
        self.complexity_ax.set_ylabel('复杂度')
        self.complexity_ax.grid(True, alpha=0.3)
        
        # 存在度图
        self.existence_ax.set_title('平均存在度 (P)', fontsize=12, fontweight='bold')
        self.existence_ax.set_xlabel('时间步')
        self.existence_ax.set_ylabel('存在度')
        self.existence_ax.grid(True, alpha=0.3)
        
        # P×C图
        self.pc_ax.set_title('P×C 乘积守恒', fontsize=12, fontweight='bold')
        self.pc_ax.set_xlabel('时间步')
        self.pc_ax.set_ylabel('P×C')
        self.pc_ax.grid(True, alpha=0.3)
        
        # 信息面板
        self.info_ax.axis('off')
        self.info_ax.set_title('系统信息', fontsize=12, fontweight='bold')
        
    def update_display(self, frame):
        """更新显示"""
        if not self.simulator.is_running:
            return
            
        # 清除之前的图形
        self.grid_ax.clear()
        self.complexity_ax.clear()
        self.existence_ax.clear()
        self.pc_ax.clear()
        self.info_ax.clear()
        
        # 重新设置图形
        self.setup_plots()
        
        # 绘制细胞网格
        cell_size = 1.0
        for pos, cell in self.simulator.grid.items():
            if cell.alive:
                rect = Rectangle((pos[0], pos[1]), cell_size, cell_size, 
                               facecolor=cell.color, edgecolor='black', linewidth=0.1)
                self.grid_ax.add_patch(rect)
                
        # 绘制统计图表
        if self.simulator.history:
            time_steps = range(len(self.simulator.history))
            
            # 复杂度趋势
            complexities = [h['avg_complexity'] for h in self.simulator.history]
            self.complexity_ax.plot(time_steps, complexities, 'r-', linewidth=2)
            
            # 存在度趋势
            existences = [h['avg_existence'] for h in self.simulator.history]
            self.existence_ax.plot(time_steps, existences, 'g-', linewidth=2)
            
            # P×C趋势
            pc_products = [h['pc_product'] for h in self.simulator.history]
            self.pc_ax.plot(time_steps, pc_products, 'm-', linewidth=2)
            
            # 添加P×C平均线
            if pc_products:
                mean_pc = np.mean(pc_products)
                self.pc_ax.axhline(y=mean_pc, color='k', linestyle='--', alpha=0.7, 
                                 label=f'平均值: {mean_pc:.3f}')
                self.pc_ax.legend()
            
        # 更新信息面板
        current_stats = self.simulator.calculate_statistics()
        elapsed = datetime.now() - self.simulator.start_time
        remaining = self.simulator.target_duration - elapsed
        
        info_text = f"""
系统状态:
• 运行时间: {elapsed}
• 剩余时间: {remaining}
• 时间步: {self.simulator.time_step:,}
• 存活细胞: {current_stats['alive_count']:,}
• 总出生: {self.simulator.total_births:,}
• 总死亡: {self.simulator.total_deaths:,}
• 最大复杂度: {self.simulator.max_complexity_reached}
• 环境压力: {current_stats['environment_stress']:.3f}

递弱代偿指标:
• 平均复杂度: {current_stats['avg_complexity']:.3f}
• 平均存在度: {current_stats['avg_existence']:.3f}
• P×C乘积: {current_stats['pc_product']:.3f}
        """
        
        self.info_ax.text(0.05, 0.95, info_text, transform=self.info_ax.transAxes, 
                         fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 执行一步模拟
        self.simulator.simulation_step()
        
    def show_control_info(self):
        """显示控制信息"""
        print("\n" + "="*60)
        print("� 实时模拟控制说明")
        print("="*60)
        print("空格键: 暂停/继续")
        print("R键: 重置模拟")
        print("S键: 保存当前状态")
        print("Q键: 退出模拟")
        print("="*60)
        print("模拟正在运行，请观察细胞活动...")

def on_key_press(event, simulator, visualizer):
    """键盘事件处理"""
    if event.key == ' ':
        simulator.paused = not simulator.paused
        status = "暂停" if simulator.paused else "继续"
        print(f"模拟已{status}")
    elif event.key.lower() == 'r':
        print("正在重置模拟...")
        simulator.__init__(simulator.grid_size, 100)  # 重新初始化
        visualizer.simulator = simulator
        print("模拟已重置")
    elif event.key.lower() == 's':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"di_ruo_dai_chang_snapshot_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"状态已保存到: {filename}")
    elif event.key.lower() == 'q':
        print("正在退出模拟...")
        simulator.is_running = False
        plt.close('all')

def main():
    """主函数"""
    print("🚀 启动24小时递弱代偿实时模拟...")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标持续时间: 24小时")
    print(f"初始细胞数量: 100")
    print(f"网格大小: 50×50")
    
    # 创建模拟器
    simulator = RealTimeDiRuoDaiChangSimulator(grid_size=50, initial_cells=100)
    
    # 创建可视化器
    visualizer = RealTimeVisualizer(simulator)
    visualizer.show_control_info()
    
    # 设置键盘事件
    visualizer.fig.canvas.mpl_connect('key_press_event', 
                                    lambda event: on_key_press(event, simulator, visualizer))
    
    # 创建动画
    anim = animation.FuncAnimation(visualizer.fig, visualizer.update_display, 
                                 interval=100, blit=False, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n用户中断模拟")
        simulator.is_running = False
    finally:
        if simulator.history:
            simulator.print_final_report()
        print("\n模拟结束")

if __name__ == "__main__":
    main()