#!/usr/bin/env python3
"""
超长时间尺度递弱代偿实验
观察系统在2000-5000步的长期演化行为
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from siyan_experiment import SiyanSimulator
import warnings
warnings.filterwarnings('ignore')


class LongTermExperiment:
    """长时间尺度实验类"""
    
    def __init__(self, params: Dict, steps: int = 5000, checkpoint_interval: int = 500):
        """
        初始化长时间实验
        
        Args:
            params: 实验参数
            steps: 总步数（2000-5000）
            checkpoint_interval: 检查点间隔，用于保存中间结果
        """
        self.params = params
        self.steps = steps
        self.checkpoint_interval = checkpoint_interval
        self.history = {
            'step': [],
            'alive_ratio': [],
            'c_mean': [],
            'c_variance': [],
            'p_mean_serial': [],
            'p_variance': [],
            'pc_serial': [],
            'pc_env': [],
            'robustness_mean': [],
            'robustness_variance': [],
            'energy_mean': [],
            'energy_variance': [],
            'birth_rate': [],
            'death_rate': [],
            'mutation_events': []
        }
        
    def run_long_term_simulation(self):
        """运行长时间模拟"""
        print(f"开始长时间尺度实验，总步数: {self.steps}")
        
        np.random.seed(42)
        simulator = SiyanSimulator(**self.params)
        
        for step in range(self.steps):
            # 执行一步模拟
            simulator.simulation_step()
            
            # 记录详细指标
            current_state = simulator.get_current_state()
            
            self.history['step'].append(step)
            self.history['alive_ratio'].append(current_state['alive_ratio'])
            self.history['c_mean'].append(current_state['c_mean'])
            self.history['c_variance'].append(current_state['c_variance'])
            self.history['p_mean_serial'].append(current_state['p_mean_serial'])
            self.history['p_variance'].append(current_state['p_variance'])
            self.history['pc_serial'].append(current_state['pc_serial'])
            self.history['pc_env'].append(current_state['pc_env'])
            self.history['robustness_mean'].append(current_state['robustness_mean'])
            self.history['robustness_variance'].append(current_state['robustness_variance'])
            self.history['energy_mean'].append(current_state['energy_mean'])
            self.history['energy_variance'].append(current_state['energy_variance'])
            self.history['birth_rate'].append(current_state['birth_rate'])
            self.history['death_rate'].append(current_state['death_rate'])
            self.history['mutation_events'].append(current_state['mutation_events'])
            
            # 检查点保存
            if (step + 1) % self.checkpoint_interval == 0:
                print(f"检查点: 第 {step + 1} 步完成")
                print(f"  存活率: {current_state['alive_ratio']:.4f}")
                print(f"  平均复杂度: {current_state['c_mean']:.4f}")
                print(f"  P·C 乘积: {current_state['pc_serial']:.4f}")
                
                # 检测是否发生系统性崩溃
                if self.detect_system_collapse(current_state):
                    print(f"警告：检测到系统性崩溃，在第 {step + 1} 步终止")
                    break
        
        print("长时间实验完成！")
        return self.history
    
    def detect_system_collapse(self, state: Dict) -> bool:
        """检测系统性崩溃"""
        # 多种崩溃检测标准
        if state['alive_ratio'] < 0.01:  # 存活率低于1%
            return True
        if state['c_mean'] > 10:  # 平均复杂度过高
            return True
        if state['pc_serial'] > 15 or state['pc_serial'] < 0.1:  # P·C乘积异常
            return True
        return False
    
    def analyze_long_term_patterns(self) -> Dict:
        """分析长期演化模式"""
        if not self.history['step']:
            return {}
        
        analysis = {
            'total_steps': len(self.history['step']),
            'final_state': {
                'alive_ratio': self.history['alive_ratio'][-1],
                'c_mean': self.history['c_mean'][-1],
                'pc_serial': self.history['pc_serial'][-1]
            },
            'stability_metrics': self.calculate_stability_metrics(),
            'phase_transitions': self.detect_phase_transitions(),
            'equilibrium_analysis': self.analyze_equilibrium(),
            'cyclic_patterns': self.detect_cycles(),
            'critical_events': self.identify_critical_events()
        }
        
        return analysis
    
    def calculate_stability_metrics(self) -> Dict:
        """计算稳定性指标"""
        alive_ratios = self.history['alive_ratio']
        c_means = self.history['c_mean']
        pc_serials = self.history['pc_serial']
        
        # 分段计算稳定性（每500步一段）
        segment_size = 500
        segments = len(alive_ratios) // segment_size
        
        stability_metrics = {
            'overall_stability': {
                'alive_ratio_cv': np.std(alive_ratios) / np.mean(alive_ratios) if np.mean(alive_ratios) > 0 else 0,
                'c_mean_cv': np.std(c_means) / np.mean(c_means) if np.mean(c_means) > 0 else 0,
                'pc_serial_cv': np.std(pc_serials) / np.mean(pc_serials) if np.mean(pc_serials) > 0 else 0
            },
            'temporal_stability': [],
            'trend_analysis': {}
        }
        
        # 时间分段稳定性
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, len(alive_ratios))
            
            segment_alive = alive_ratios[start_idx:end_idx]
            segment_c = c_means[start_idx:end_idx]
            segment_pc = pc_serials[start_idx:end_idx]
            
            if len(segment_alive) > 1:
                stability_metrics['temporal_stability'].append({
                    'segment': i + 1,
                    'time_range': f"{start_idx}-{end_idx}",
                    'alive_ratio_mean': np.mean(segment_alive),
                    'alive_ratio_std': np.std(segment_alive),
                    'c_mean_mean': np.mean(segment_c),
                    'c_mean_std': np.std(segment_c),
                    'pc_serial_mean': np.mean(segment_pc),
                    'pc_serial_std': np.std(segment_pc)
                })
        
        # 趋势分析
        steps = self.history['step']
        stability_metrics['trend_analysis']['alive_ratio_trend'] = self.calculate_trend(steps, alive_ratios)
        stability_metrics['trend_analysis']['c_mean_trend'] = self.calculate_trend(steps, c_means)
        stability_metrics['trend_analysis']['pc_serial_trend'] = self.calculate_trend(steps, pc_serials)
        
        return stability_metrics
    
    def calculate_trend(self, x: List, y: List) -> float:
        """计算趋势"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            coeffs = np.polyfit(x, y, 1)
            return coeffs[0]
        except:
            return 0.0
    
    def detect_phase_transitions(self) -> List[Dict]:
        """检测相变点"""
        alive_ratios = self.history['alive_ratio']
        c_means = self.history['c_mean']
        steps = self.history['step']
        
        transitions = []
        window_size = 100  # 滑动窗口大小
        
        for i in range(window_size, len(alive_ratios) - window_size):
            # 检查存活率的急剧变化
            prev_window = alive_ratios[i-window_size:i]
            next_window = alive_ratios[i:i+window_size]
            
            if len(prev_window) > 0 and len(next_window) > 0:
                prev_mean = np.mean(prev_window)
                next_mean = np.mean(next_window)
                
                # 如果变化超过阈值，认为是相变
                if abs(next_mean - prev_mean) > 0.2:
                    transitions.append({
                        'step': steps[i],
                        'type': 'survival_crash' if next_mean < prev_mean else 'survival_recovery',
                        'magnitude': abs(next_mean - prev_mean),
                        'from_value': prev_mean,
                        'to_value': next_mean
                    })
            
            # 检查复杂度的急剧变化
            prev_c_window = c_means[i-window_size:i]
            next_c_window = c_means[i:i+window_size]
            
            if len(prev_c_window) > 0 and len(next_c_window) > 0:
                prev_c_mean = np.mean(prev_c_window)
                next_c_mean = np.mean(next_c_window)
                
                if abs(next_c_mean - prev_c_mean) > 1.0:
                    transitions.append({
                        'step': steps[i],
                        'type': 'complexity_surge' if next_c_mean > prev_c_mean else 'complexity_collapse',
                        'magnitude': abs(next_c_mean - prev_c_mean),
                        'from_value': prev_c_mean,
                        'to_value': next_c_mean
                    })
        
        return transitions
    
    def analyze_equilibrium(self) -> Dict:
        """分析平衡态特征"""
        alive_ratios = self.history['alive_ratio']
        c_means = self.history['c_mean']
        pc_serials = self.history['pc_serial']
        
        # 只分析后一半数据（假设已达到平衡）
        half_point = len(alive_ratios) // 2
        
        equilibrium_data = {
            'equilibrium_alive_ratio': np.mean(alive_ratios[half_point:]),
            'equilibrium_c_mean': np.mean(c_means[half_point:]),
            'equilibrium_pc_serial': np.mean(pc_serials[half_point:]),
            'fluctuation_alive_ratio': np.std(alive_ratios[half_point:]),
            'fluctuation_c_mean': np.std(c_means[half_point:]),
            'fluctuation_pc_serial': np.std(pc_serials[half_point:])
        }
        
        return equilibrium_data
    
    def detect_cycles(self) -> List[Dict]:
        """检测周期性模式"""
        # 简化版周期检测，基于自相关
        alive_ratios = self.history['alive_ratio']
        
        cycles = []
        min_cycle_length = 100
        max_cycle_length = 1000
        
        for cycle_length in range(min_cycle_length, min(max_cycle_length, len(alive_ratios)//2)):
            # 计算自相关
            if len(alive_ratios) > cycle_length:
                correlation = np.corrcoef(alive_ratios[:-cycle_length], alive_ratios[cycle_length:])[0, 1]
                
                if correlation > 0.7:  # 强相关性表示可能的周期
                    cycles.append({
                        'cycle_length': cycle_length,
                        'correlation': correlation,
                        'confidence': correlation ** 2
                    })
        
        # 按相关性排序，取前5个
        cycles = sorted(cycles, key=lambda x: x['correlation'], reverse=True)[:5]
        
        return cycles
    
    def identify_critical_events(self) -> List[Dict]:
        """识别关键事件"""
        alive_ratios = self.history['alive_ratio']
        c_means = self.history['c_mean']
        steps = self.history['step']
        
        events = []
        
        # 识别极端事件
        alive_threshold_high = np.percentile(alive_ratios, 90)
        alive_threshold_low = np.percentile(alive_ratios, 10)
        c_threshold_high = np.percentile(c_means, 90)
        c_threshold_low = np.percentile(c_means, 10)
        
        for i, (step, alive, c_mean) in enumerate(zip(steps, alive_ratios, c_means)):
            if alive > alive_threshold_high:
                events.append({
                    'step': step,
                    'type': 'population_peak',
                    'value': alive,
                    'magnitude': 'high'
                })
            elif alive < alive_threshold_low:
                events.append({
                    'step': step,
                    'type': 'population_crash',
                    'value': alive,
                    'magnitude': 'low'
                })
            
            if c_mean > c_threshold_high:
                events.append({
                    'step': step,
                    'type': 'complexity_surge',
                    'value': c_mean,
                    'magnitude': 'high'
                })
            elif c_mean < c_threshold_low:
                events.append({
                    'step': step,
                    'type': 'complexity_crash',
                    'value': c_mean,
                    'magnitude': 'low'
                })
        
        return events
    
    def plot_long_term_evolution(self):
        """绘制长期演化图"""
        if not self.history['step']:
            print("没有数据可绘制")
            return
        
        steps = self.history['step']
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        
        # 1. 存活率演化
        axes[0, 0].plot(steps, self.history['alive_ratio'], 'b-', linewidth=1, alpha=0.8)
        axes[0, 0].set_xlabel('步数')
        axes[0, 0].set_ylabel('存活率')
        axes[0, 0].set_title('长期存活率演化')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # 2. 复杂度演化
        axes[0, 1].plot(steps, self.history['c_mean'], 'r-', linewidth=1, alpha=0.8)
        axes[0, 1].set_xlabel('步数')
        axes[0, 1].set_ylabel('平均复杂度')
        axes[0, 1].set_title('长期复杂度演化')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. P·C 乘积演化
        axes[1, 0].plot(steps, self.history['pc_serial'], 'g-', linewidth=1, alpha=0.8)
        axes[1, 0].set_xlabel('步数')
        axes[1, 0].set_ylabel('P·C 乘积')
        axes[1, 0].set_title('P·C 乘积长期演化')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 复杂度方差
        axes[1, 1].plot(steps, self.history['c_variance'], 'm-', linewidth=1, alpha=0.8)
        axes[1, 1].set_xlabel('步数')
        axes[1, 1].set_ylabel('复杂度方差')
        axes[1, 1].set_title('复杂度方差演化')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 出生率与死亡率
        axes[2, 0].plot(steps, self.history['birth_rate'], 'c-', linewidth=1, alpha=0.8, label='出生率')
        axes[2, 0].plot(steps, self.history['death_rate'], 'y-', linewidth=1, alpha=0.8, label='死亡率')
        axes[2, 0].set_xlabel('步数')
        axes[2, 0].set_ylabel('率')
        axes[2, 0].set_title('出生率与死亡率演化')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 平均能量
        axes[2, 1].plot(steps, self.history['energy_mean'], 'orange', linewidth=1, alpha=0.8)
        axes[2, 1].set_xlabel('步数')
        axes[2, 1].set_ylabel('平均能量')
        axes[2, 1].set_title('平均能量演化')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 7. 鲁棒性演化
        axes[3, 0].plot(steps, self.history['robustness_mean'], 'purple', linewidth=1, alpha=0.8)
        axes[3, 0].set_xlabel('步数')
        axes[3, 0].set_ylabel('平均鲁棒性')
        axes[3, 0].set_title('鲁棒性长期演化')
        axes[3, 0].grid(True, alpha=0.3)
        
        # 8. 变异事件累积
        cumulative_mutations = np.cumsum(self.history['mutation_events'])
        axes[3, 1].plot(steps, cumulative_mutations, 'brown', linewidth=1, alpha=0.8)
        axes[3, 1].set_xlabel('步数')
        axes[3, 1].set_ylabel('累积变异事件')
        axes[3, 1].set_title('变异事件累积')
        axes[3, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('long_term_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_phase_transitions(self, transitions: List[Dict]):
        """绘制相变点"""
        if not transitions:
            print("没有检测到相变")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        steps = self.history['step']
        alive_ratios = self.history['alive_ratio']
        c_means = self.history['c_mean']
        
        # 存活率相变
        axes[0].plot(steps, alive_ratios, 'b-', linewidth=1, alpha=0.8, label='存活率')
        
        # 标记相变点
        survival_transitions = [t for t in transitions if 'survival' in t['type']]
        for transition in survival_transitions:
            color = 'red' if 'crash' in transition['type'] else 'green'
            axes[0].axvline(x=transition['step'], color=color, linestyle='--', alpha=0.7)
            axes[0].text(transition['step'], transition['from_value'], 
                        f"{transition['type'][:4]}\n{transition['magnitude']:.2f}", 
                        rotation=90, fontsize=8)
        
        axes[0].set_xlabel('步数')
        axes[0].set_ylabel('存活率')
        axes[0].set_title('存活率相变点')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 复杂度相变
        axes[1].plot(steps, c_means, 'r-', linewidth=1, alpha=0.8, label='平均复杂度')
        
        # 标记相变点
        complexity_transitions = [t for t in transitions if 'complexity' in t['type']]
        for transition in complexity_transitions:
            color = 'orange' if 'surge' in transition['type'] else 'purple'
            axes[1].axvline(x=transition['step'], color=color, linestyle='--', alpha=0.7)
            axes[1].text(transition['step'], transition['from_value'], 
                        f"{transition['type'][:4]}\n{transition['magnitude']:.2f}", 
                        rotation=90, fontsize=8)
        
        axes[1].set_xlabel('步数')
        axes[1].set_ylabel('平均复杂度')
        axes[1].set_title('复杂度相变点')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('phase_transitions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename: str):
        """保存结果"""
        # 保存历史数据
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(f"{filename}_history.csv", index=False)
        
        # 保存分析结果
        analysis = self.analyze_long_term_patterns()
        with open(f"{filename}_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存参数
        with open(f"{filename}_params.json", 'w', encoding='utf-8') as f:
            json.dump(self.params, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到 {filename}_history.csv, {filename}_analysis.json, {filename}_params.json")


def main():
    """主函数：运行长时间尺度实验"""
    
    # 实验参数 - 选择之前表现较好的参数组合
    params = {
        'grid_size': 40,
        'initial_density': 0.4,
        'initial_complexity': 1,
        'initial_energy': 5.0,
        'gamma': 1.4,      # 从超细扫描中得到的较优值
        'beta': 0.4,
        'r_mean': 1.0,
        'r_noise': 0.2,
        'env_sigma': 0.05,
        'p_up': 0.04,
        'p_down': 0.03,
        'birth_energy_threshold': 3.0,
        'r': 0.98
    }
    
    print("超长时间尺度递弱代偿实验")
    print("=" * 50)
    print(f"实验参数: {params}")
    print(f"总步数: 5000步")
    print(f"预计实验时间: 约 15-20 分钟")
    
    # 创建实验
    experiment = LongTermExperiment(params, steps=5000, checkpoint_interval=500)
    
    # 运行实验
    history = experiment.run_long_term_simulation()
    
    # 分析结果
    analysis = experiment.analyze_long_term_patterns()
    
    print("\n长期演化分析结果:")
    print("-" * 40)
    print(f"总步数: {analysis['total_steps']}")
    print(f"最终状态:")
    print(f"  存活率: {analysis['final_state']['alive_ratio']:.4f}")
    print(f"  平均复杂度: {analysis['final_state']['c_mean']:.4f}")
    print(f"  P·C 乘积: {analysis['final_state']['pc_serial']:.4f}")
    
    print(f"\n稳定性指标:")
    stability = analysis['stability_metrics']['overall_stability']
    print(f"  存活率变异系数: {stability['alive_ratio_cv']:.4f}")
    print(f"  复杂度变异系数: {stability['c_mean_cv']:.4f}")
    print(f"  P·C 变异系数: {stability['pc_serial_cv']:.4f}")
    
    print(f"\n相变点数量: {len(analysis['phase_transitions'])}")
    print(f"周期性模式数量: {len(analysis['cyclic_patterns'])}")
    print(f"关键事件数量: {len(analysis['critical_events'])}")
    
    # 绘制长期演化图
    experiment.plot_long_term_evolution()
    
    # 绘制相变点
    experiment.plot_phase_transitions(analysis['phase_transitions'])
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment.save_results(f"long_term_experiment_{timestamp}")
    
    # 打印平衡态分析
    equilibrium = analysis['equilibrium_analysis']
    print(f"\n平衡态分析（后一半数据）:")
    print(f"  平衡态存活率: {equilibrium['equilibrium_alive_ratio']:.4f} ± {equilibrium['fluctuation_alive_ratio']:.4f}")
    print(f"  平衡态复杂度: {equilibrium['equilibrium_c_mean']:.4f} ± {equilibrium['fluctuation_c_mean']:.4f}")
    print(f"  平衡态P·C: {equilibrium['equilibrium_pc_serial']:.4f} ± {equilibrium['fluctuation_pc_serial']:.4f}")


if __name__ == "__main__":
    main()