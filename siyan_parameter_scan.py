#!/usr/bin/env python3
"""
递弱代偿参数扫描实验
系统地测试不同参数组合，寻找"过度代偿→集体崩盘"现象
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


class ParameterScanExperiment:
    """参数扫描实验类"""
    
    def __init__(self, base_params: Dict, scan_params: Dict, steps: int = 1000):
        """
        初始化参数扫描实验
        
        Args:
            base_params: 基础参数（固定不变）
            scan_params: 扫描参数（字典，键为参数名，值为要测试的列表）
            steps: 每轮模拟的步数
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
        """运行单个实验"""
        # 设置随机种子以确保可重复性
        np.random.seed(42)
        
        simulator = SiyanSimulator(**params)
        simulator.run_simulation(self.steps)
        
        # 提取关键指标
        history = simulator.history
        
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
            'c_p_correlation': self.calculate_correlation(history['c_mean'], history['p_mean_serial'])
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
    
    def run_all_experiments(self, progress_interval: int = 10):
        """运行所有实验"""
        combinations = self.generate_parameter_combinations()
        total = len(combinations)
        
        print(f"开始参数扫描实验，共 {total} 组参数...")
        
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
    
    def analyze_results(self) -> Dict:
        """分析实验结果"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        analysis = {
            'total_experiments': len(self.results),
            'collapse_rate': df['collapse_detected'].mean(),
            'avg_final_alive_ratio': df['final_alive_ratio'].mean(),
            'avg_final_c_mean': df['final_c_mean'].mean(),
            'avg_pc_serial_correlation': df['pc_serial_correlation'].mean(),
            'avg_pc_env_correlation': df['pc_env_correlation'].mean(),
            'avg_c_p_correlation': df['c_p_correlation'].mean()
        }
        
        # 寻找极端情况
        collapse_cases = df[df['collapse_detected'] == True]
        if not collapse_cases.empty:
            analysis['collapse_avg_c_mean'] = collapse_cases['final_c_mean'].mean()
            analysis['collapse_avg_alive_ratio'] = collapse_cases['final_alive_ratio'].mean()
        
        survival_cases = df[df['collapse_detected'] == False]
        if not survival_cases.empty:
            analysis['survival_avg_c_mean'] = survival_cases['final_c_mean'].mean()
            analysis['survival_avg_alive_ratio'] = survival_cases['final_alive_ratio'].mean()
        
        return analysis
    
    def plot_parameter_effects(self):
        """绘制参数影响图"""
        if not self.results:
            print("没有实验结果可绘制")
            return
        
        # 提取扫描参数
        scan_param_names = list(self.scan_params.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 参数对崩盘率的影响
        if len(scan_param_names) >= 1:
            param1 = scan_param_names[0]
            param1_values = [r['params'][param1] for r in self.results]
            collapse_rates = []
            
            for value in self.scan_params[param1]:
                value_results = [r for r in self.results if r['params'][param1] == value]
                if value_results:
                    collapse_rate = sum(r['collapse_detected'] for r in value_results) / len(value_results)
                    collapse_rates.append(collapse_rate)
                else:
                    collapse_rates.append(0)
            
            axes[0, 0].plot(self.scan_params[param1], collapse_rates, 'bo-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel(param1)
            axes[0, 0].set_ylabel('崩盘率')
            axes[0, 0].set_title(f'{param1} 对崩盘率的影响')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 参数对最终存活率的影响
        if len(scan_param_names) >= 1:
            param1 = scan_param_names[0]
            param1_values = [r['params'][param1] for r in self.results]
            final_ratios = []
            
            for value in self.scan_params[param1]:
                value_results = [r for r in self.results if r['params'][param1] == value]
                if value_results:
                    avg_ratio = np.mean([r['final_alive_ratio'] for r in value_results])
                    final_ratios.append(avg_ratio)
                else:
                    final_ratios.append(0)
            
            axes[0, 1].plot(self.scan_params[param1], final_ratios, 'ro-', linewidth=2, markersize=8)
            axes[0, 1].set_xlabel(param1)
            axes[0, 1].set_ylabel('最终存活率')
            axes[0, 1].set_title(f'{param1} 对最终存活率的影响')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 参数对平均复杂度的影响
        if len(scan_param_names) >= 1:
            param1 = scan_param_names[0]
            param1_values = [r['params'][param1] for r in self.results]
            final_c_means = []
            
            for value in self.scan_params[param1]:
                value_results = [r for r in self.results if r['params'][param1] == value]
                if value_results:
                    avg_c = np.mean([r['final_c_mean'] for r in value_results])
                    final_c_means.append(avg_c)
                else:
                    final_c_means.append(0)
            
            axes[1, 0].plot(self.scan_params[param1], final_c_means, 'go-', linewidth=2, markersize=8)
            axes[1, 0].set_xlabel(param1)
            axes[1, 0].set_ylabel('最终平均复杂度')
            axes[1, 0].set_title(f'{param1} 对最终平均复杂度的影响')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. P·C 相关性分布
        pc_correlations = [r['pc_serial_correlation'] for r in self.results]
        axes[1, 1].hist(pc_correlations, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('P·C 相关系数')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('P·C 相关系数分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('parameter_scan_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename: str):
        """保存实验结果"""
        # 保存详细结果
        with open(f"{filename}_detailed.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存分析结果
        analysis = self.analyze_results()
        with open(f"{filename}_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # 保存为CSV格式
        df = pd.DataFrame(self.results)
        
        # 展开参数字段
        param_df = pd.json_normalize(df['params'])
        result_df = df.drop('params', axis=1)
        final_df = pd.concat([param_df, result_df], axis=1)
        
        final_df.to_csv(f"{filename}.csv", index=False)
        
        print(f"结果已保存到 {filename}.csv, {filename}_detailed.json, {filename}_analysis.json")


def main():
    """主函数：运行参数扫描实验"""
    
    # 基础参数（固定不变）
    base_params = {
        'grid_size': 50,
        'initial_density': 0.3,
        'initial_complexity': 1,
        'initial_energy': 5.0,
        'r_mean': 1.0,
        'r_noise': 0.2,
        'env_sigma': 0.05,
        'birth_energy_threshold': 3.0,
        'p_down': 0.03
    }
    
    # 扫描参数（要测试的参数组合）
    scan_params = {
        'gamma': [1.2, 1.5, 1.8, 2.0],  # 维护成本超线性系数
        'beta': [0.3, 0.5, 0.7, 1.0],    # 环境敏感性系数
        'p_up': [0.03, 0.05, 0.08, 0.1], # 复杂度上调概率
        'r': [0.95, 0.98, 0.99, 0.995]   # 环节可靠性
    }
    
    print("递弱代偿参数扫描实验")
    print("=" * 50)
    print(f"基础参数: {base_params}")
    print(f"扫描参数: {scan_params}")
    print(f"总实验数: {len(list(itertools.product(*scan_params.values())))}")
    
    # 创建实验
    experiment = ParameterScanExperiment(base_params, scan_params, steps=1000)
    
    # 运行所有实验
    experiment.run_all_experiments(progress_interval=20)
    
    # 分析结果
    analysis = experiment.analyze_results()
    print("\n实验分析结果:")
    print("-" * 30)
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 绘制结果
    experiment.plot_parameter_effects()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment.save_results(f"parameter_scan_{timestamp}")
    
    # 寻找最佳参数组合
    if experiment.results:
        best_survival = max(experiment.results, key=lambda x: x['final_alive_ratio'])
        best_pc_stability = min(experiment.results, key=lambda x: abs(x['pc_serial_correlation']))
        
        print(f"\n最佳存活参数组合:")
        print(f"存活率: {best_survival['final_alive_ratio']:.4f}")
        print(f"参数: {best_survival['params']}")
        
        print(f"\n最稳定P·C参数组合:")
        print(f"P·C相关系数: {best_pc_stability['pc_serial_correlation']:.4f}")
        print(f"参数: {best_pc_stability['params']}")


if __name__ == "__main__":
    main()