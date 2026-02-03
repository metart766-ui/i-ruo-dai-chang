#!/usr/bin/env python3
"""
超细颗粒度参数扫描实验
更密集的参数组合，寻找临界条件和相变点
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import itertools
import json
from datetime import datetime
from siyan_experiment import SiyanSimulator
import warnings
warnings.filterwarnings('ignore')


class UltraFineParameterScan:
    """超细颗粒度参数扫描实验类"""
    
    def __init__(self, base_params: Dict, scan_params: Dict, steps: int = 2000):
        """
        初始化超细颗粒度参数扫描实验
        
        Args:
            base_params: 基础参数（固定不变）
            scan_params: 扫描参数（字典，键为参数名，值为要测试的密集列表）
            steps: 每轮模拟的步数（增加到2000步观察长期演化）
        """
        self.base_params = base_params
        self.scan_params = scan_params
        self.steps = steps
        self.results = []
        
    def generate_parameter_combinations(self) -> List[Dict]:
        """生成所有参数组合"""
        param_names = list(self.scan_params.keys())
        param_values = list(self.scan_params.values())
        
        combinations = []
        for values in itertools.product(*param_values):
            combo = self.base_params.copy()
            for name, value in zip(param_names, values):
                combo[name] = value
            combinations.append(combo)
            
        return combinations
    
    def run_single_experiment(self, params: Dict) -> Dict:
        """运行单个实验，增加更多指标监测"""
        np.random.seed(42)
        
        simulator = SiyanSimulator(**params)
        simulator.run_simulation(self.steps)
        
        history = simulator.history
        
        # 计算更多详细指标
        result = {
            'params': params,
            'final_alive_ratio': history['alive_ratio'][-1] if history['alive_ratio'] else 0,
            'final_c_mean': history['c_mean'][-1] if history['c_mean'] else 0,
            'final_pc_serial': history['pc_serial'][-1] if history['pc_serial'] else 0,
            'final_pc_env': history['pc_env'][-1] if history['pc_env'] else 0,
            'collapse_detected': simulator.detect_collapse(),
            'max_c_mean': max(history['c_mean']) if history['c_mean'] else 0,
            'min_alive_ratio': min(history['alive_ratio']) if history['alive_ratio'] else 0,
            'pc_serial_correlation': self.calculate_correlation(history['step'], history['pc_serial']),
            'pc_env_correlation': self.calculate_correlation(history['step'], history['pc_env']),
            'c_p_correlation': self.calculate_correlation(history['c_mean'], history['p_mean_serial']),
            'c_variance': np.var(history['c_mean']) if history['c_mean'] else 0,
            'p_variance': np.var(history['p_mean_serial']) if history['p_mean_serial'] else 0,
            'alive_ratio_trend': self.calculate_trend(history['step'], history['alive_ratio']),
            'c_mean_trend': self.calculate_trend(history['step'], history['c_mean']),
            'early_survival_rate': np.mean(history['alive_ratio'][:100]) if len(history['alive_ratio']) >= 100 else 0,
            'late_survival_rate': np.mean(history['alive_ratio'][-100:]) if len(history['alive_ratio']) >= 100 else 0,
            'complexity_peak_step': history['c_mean'].index(max(history['c_mean'])) if history['c_mean'] else 0,
            'survival_half_life': self.calculate_half_life(history['step'], history['alive_ratio'])
        }
        
        return result
    
    def calculate_correlation(self, x: List, y: List) -> float:
        """计算相关系数"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            return np.corrcoef(x, y)[0, 1]
        except:
            return 0.0
    
    def calculate_trend(self, x: List, y: List) -> float:
        """计算趋势（线性回归斜率）"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            coeffs = np.polyfit(x, y, 1)
            return coeffs[0]  # 斜率
        except:
            return 0.0
    
    def calculate_half_life(self, steps: List, survival_rates: List) -> int:
        """计算半衰期（存活率降到初始值一半所需的步数）"""
        if not survival_rates or len(survival_rates) < 2:
            return self.steps
        
        initial_rate = survival_rates[0]
        half_rate = initial_rate / 2
        
        for i, rate in enumerate(survival_rates):
            if rate <= half_rate:
                return steps[i]
        
        return self.steps
    
    def run_all_experiments(self, progress_interval: int = 50):
        """运行所有实验"""
        combinations = self.generate_parameter_combinations()
        total = len(combinations)
        
        print(f"开始超细颗粒度参数扫描实验，共 {total} 组参数...")
        print(f"每轮模拟步数: {self.steps}")
        
        for i, params in enumerate(combinations):
            try:
                result = self.run_single_experiment(params)
                self.results.append(result)
                
                if (i + 1) % progress_interval == 0:
                    print(f"进度: {i + 1}/{total} ({(i + 1)/total*100:.1f}%)")
                    
            except Exception as e:
                print(f"实验 {i + 1} 失败: {str(e)}")
                continue
        
        print(f"实验完成！成功完成 {len(self.results)}/{total} 组实验")
    
    def analyze_critical_transitions(self) -> Dict:
        """分析临界相变"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        # 寻找临界参数值
        critical_analysis = {
            'collapse_boundary': {},
            'survival_boundary': {},
            'phase_transitions': []
        }
        
        # 对每个扫描参数分析临界值
        for param_name in self.scan_params.keys():
            param_values = sorted(self.scan_params[param_name])
            
            collapse_rates = []
            survival_rates = []
            
            for value in param_values:
                value_results = df[df['params'].apply(lambda x: x.get(param_name) == value)]
                if not value_results.empty:
                    collapse_rate = value_results['collapse_detected'].mean()
                    avg_survival = value_results['final_alive_ratio'].mean()
                    collapse_rates.append(collapse_rate)
                    survival_rates.append(avg_survival)
                else:
                    collapse_rates.append(0)
                    survival_rates.append(0)
            
            # 寻找相变点（崩盘率急剧变化的参数值）
            if len(collapse_rates) >= 3:
                for i in range(1, len(collapse_rates)-1):
                    # 检查是否有急剧变化
                    if abs(collapse_rates[i+1] - collapse_rates[i-1]) > 0.3:
                        critical_analysis['phase_transitions'].append({
                            'parameter': param_name,
                            'critical_value': param_values[i],
                            'collapse_rate_change': collapse_rates[i+1] - collapse_rates[i-1]
                        })
            
            critical_analysis['collapse_boundary'][param_name] = {
                'param_values': param_values,
                'collapse_rates': collapse_rates
            }
        
        return critical_analysis
    
    def plot_phase_diagram(self):
        """绘制相图"""
        if not self.results or len(self.scan_params) < 2:
            print("需要至少2个扫描参数才能绘制相图")
            return
        
        df = pd.DataFrame(self.results)
        
        # 获取前两个扫描参数
        param1, param2 = list(self.scan_params.keys())[:2]
        
        # 创建参数网格
        param1_values = sorted(self.scan_params[param1])
        param2_values = sorted(self.scan_params[param2])
        
        # 创建相图数据
        phase_map = np.zeros((len(param2_values), len(param1_values)))
        survival_map = np.zeros((len(param2_values), len(param1_values)))
        
        for i, p2_val in enumerate(param2_values):
            for j, p1_val in enumerate(param1_values):
                # 找到对应参数组合的结果
                mask = (df['params'].apply(lambda x: x.get(param1) == p1_val)) & \
                       (df['params'].apply(lambda x: x.get(param2) == p2_val))
                
                selected_results = df[mask]
                
                if not selected_results.empty:
                    # 相图：崩盘率
                    phase_map[i, j] = selected_results['collapse_detected'].mean()
                    # 存活率图
                    survival_map[i, j] = selected_results['final_alive_ratio'].mean()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 相图
        im1 = axes[0].imshow(phase_map, cmap='RdYlBu_r', aspect='auto', origin='lower')
        axes[0].set_xlabel(param1)
        axes[0].set_ylabel(param2)
        axes[0].set_title('相图：崩盘率')
        axes[0].set_xticks(range(len(param1_values)))
        axes[0].set_xticklabels([f'{v:.2f}' for v in param1_values])
        axes[0].set_yticks(range(len(param2_values)))
        axes[0].set_yticklabels([f'{v:.2f}' for v in param2_values])
        
        # 添加数值标注
        for i in range(len(param2_values)):
            for j in range(len(param1_values)):
                if not np.isnan(phase_map[i, j]):
                    axes[0].text(j, i, f'{phase_map[i, j]:.2f}', 
                               ha='center', va='center', color='white', fontsize=8)
        
        plt.colorbar(im1, ax=axes[0], label='崩盘率')
        
        # 存活率图
        im2 = axes[1].imshow(survival_map, cmap='RdYlGn', aspect='auto', origin='lower')
        axes[1].set_xlabel(param1)
        axes[1].set_ylabel(param2)
        axes[1].set_title('存活率分布')
        axes[1].set_xticks(range(len(param1_values)))
        axes[1].set_xticklabels([f'{v:.2f}' for v in param1_values])
        axes[1].set_yticks(range(len(param2_values)))
        axes[1].set_yticklabels([f'{v:.2f}' for v in param2_values])
        
        # 添加数值标注
        for i in range(len(param2_values)):
            for j in range(len(param1_values)):
                if not np.isnan(survival_map[i, j]):
                    axes[1].text(j, i, f'{survival_map[i, j]:.2f}', 
                               ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im2, ax=axes[1], label='存活率')
        
        plt.tight_layout()
        plt.savefig('ultra_fine_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_detailed_analysis(self):
        """绘制详细分析图"""
        if not self.results:
            print("没有实验结果可绘制")
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # 1. 崩盘率 vs 复杂度趋势
        axes[0, 0].scatter(df['final_c_mean'], df['collapse_detected'], alpha=0.6)
        axes[0, 0].set_xlabel('最终平均复杂度')
        axes[0, 0].set_ylabel('崩盘状态')
        axes[0, 0].set_title('复杂度与崩盘关系')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 存活率分布
        axes[0, 1].hist(df['final_alive_ratio'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('最终存活率')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('存活率分布')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 复杂度方差 vs 崩盘
        axes[0, 2].scatter(df['c_variance'], df['collapse_detected'], alpha=0.6, color='red')
        axes[0, 2].set_xlabel('复杂度方差')
        axes[0, 2].set_ylabel('崩盘状态')
        axes[0, 2].set_title('复杂度波动与崩盘')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. P·C 相关性分布
        axes[1, 0].hist(df['pc_serial_correlation'], bins=30, alpha=0.7, color='purple')
        axes[1, 0].set_xlabel('P·C 相关系数')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('P·C 相关性分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 早期 vs 晚期存活率
        axes[1, 1].scatter(df['early_survival_rate'], df['late_survival_rate'], alpha=0.6)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1, 1].set_xlabel('早期存活率')
        axes[1, 1].set_ylabel('晚期存活率')
        axes[1, 1].set_title('早期 vs 晚期存活率')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 复杂度峰值分布
        axes[1, 2].hist(df['complexity_peak_step'], bins=30, alpha=0.7, color='orange')
        axes[1, 2].set_xlabel('复杂度峰值步数')
        axes[1, 2].set_ylabel('频次')
        axes[1, 2].set_title('复杂度峰值时间分布')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. 存活率趋势
        axes[2, 0].hist(df['alive_ratio_trend'], bins=30, alpha=0.7, color='blue')
        axes[2, 0].set_xlabel('存活率趋势（斜率）')
        axes[2, 0].set_ylabel('频次')
        axes[2, 0].set_title('存活率趋势分布')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. 复杂度趋势
        axes[2, 1].hist(df['c_mean_trend'], bins=30, alpha=0.7, color='brown')
        axes[2, 1].set_xlabel('复杂度趋势（斜率）')
        axes[2, 1].set_ylabel('频次')
        axes[2, 1].set_title('复杂度趋势分布')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. 半衰期分布
        axes[2, 2].hist(df['survival_half_life'], bins=30, alpha=0.7, color='pink')
        axes[2, 2].set_xlabel('存活半衰期')
        axes[2, 2].set_ylabel('频次')
        axes[2, 2].set_title('存活半衰期分布')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ultra_fine_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename: str):
        """保存实验结果"""
        # 保存详细结果
        with open(f"{filename}_detailed.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存分析结果
        analysis = self.analyze_critical_transitions()
        with open(f"{filename}_critical_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # 保存为CSV格式
        df = pd.DataFrame(self.results)
        param_df = pd.json_normalize(df['params'])
        result_df = df.drop('params', axis=1)
        final_df = pd.concat([param_df, result_df], axis=1)
        
        final_df.to_csv(f"{filename}.csv", index=False)
        
        print(f"结果已保存到 {filename}.csv, {filename}_detailed.json, {filename}_critical_analysis.json")


def main():
    """主函数：运行超细颗粒度参数扫描实验"""
    
    # 基础参数（固定不变）
    base_params = {
        'grid_size': 40,  # 增加网格大小
        'initial_density': 0.4,  # 增加初始密度
        'initial_complexity': 1,
        'initial_energy': 5.0,
        'r_mean': 1.0,
        'r_noise': 0.2,
        'env_sigma': 0.05,
        'birth_energy_threshold': 3.0,
        'p_down': 0.03
    }
    
    # 超细颗粒度扫描参数
    scan_params = {
        'gamma': np.arange(1.1, 2.1, 0.1).tolist(),  # 0.1步长，11个值
        'beta': np.arange(0.2, 0.9, 0.1).tolist(),   # 0.1步长，7个值
        'p_up': np.arange(0.02, 0.11, 0.01).tolist(), # 0.01步长，9个值
        'r': [0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.998]  # 8个值
    }
    
    total_combinations = len(list(itertools.product(*scan_params.values())))
    
    print("超细颗粒度递弱代偿参数扫描实验")
    print("=" * 60)
    print(f"基础参数: {base_params}")
    print(f"扫描参数: {scan_params}")
    print(f"总实验数: {total_combinations}")
    print(f"每轮模拟步数: 2000步")
    print(f"预计实验时间: {total_combinations * 2 / 60:.1f} 分钟")
    
    # 创建实验
    experiment = UltraFineParameterScan(base_params, scan_params, steps=2000)
    
    # 运行所有实验
    experiment.run_all_experiments(progress_interval=100)
    
    # 分析临界相变
    critical_analysis = experiment.analyze_critical_transitions()
    print("\n临界相变分析:")
    print("-" * 40)
    for transition in critical_analysis.get('phase_transitions', []):
        print(f"参数 {transition['parameter']} 在值 {transition['critical_value']:.3f} 处发生相变，"
              f"崩盘率变化: {transition['collapse_rate_change']:.3f}")
    
    # 绘制相图
    experiment.plot_phase_diagram()
    
    # 绘制详细分析
    experiment.plot_detailed_analysis()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment.save_results(f"ultra_fine_scan_{timestamp}")
    
    # 寻找关键发现
    if experiment.results:
        # 最容易崩盘的参数组合
        most_unstable = max(experiment.results, key=lambda x: x['collapse_detected'])
        
        # 最稳定的参数组合
        most_stable = min(experiment.results, key=lambda x: abs(x['pc_serial_correlation']))
        
        # 最高存活率
        best_survival = max(experiment.results, key=lambda x: x['final_alive_ratio'])
        
        print(f"\n关键发现:")
        print("-" * 30)
        print(f"最不稳定参数组合:")
        print(f"  崩盘状态: {most_unstable['collapse_detected']}")
        print(f"  参数: {most_unstable['params']}")
        
        print(f"\n最稳定P·C参数组合:")
        print(f"  P·C相关系数: {most_stable['pc_serial_correlation']:.6f}")
        print(f"  参数: {most_stable['params']}")
        
        print(f"\n最高存活率参数组合:")
        print(f"  存活率: {best_survival['final_alive_ratio']:.4f}")
        print(f"  参数: {best_survival['params']}")


if __name__ == "__main__":
    main()