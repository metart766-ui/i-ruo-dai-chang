#!/usr/bin/env python3
"""
超细颗粒度参数扫描 - 详细版本
用于寻找临界条件和相变点
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import concurrent.futures
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

from siyan_experiment import SiyanSimulator

@dataclass
class DetailedExperimentConfig:
    """详细实验配置"""
    grid_size: int = 35
    steps: int = 3000
    gamma_range: Tuple[float, float] = (1.0, 2.2)
    beta_range: Tuple[float, float] = (0.1, 0.9)
    p_up_range: Tuple[float, float] = (0.01, 0.12)
    r_range: Tuple[float, float] = (0.92, 0.999)
    
    gamma_step: float = 0.05  # 0.05步长
    beta_step: float = 0.02   # 0.02步长  
    p_up_step: float = 0.005  # 0.005步长
    r_step: float = 0.002     # 0.002步长
    
    n_replicates: int = 5     # 每个参数组合重复5次
    checkpoint_interval: int = 100
    
class UltraFineDetailedScanner:
    """超细颗粒度参数扫描器"""
    
    def __init__(self, config: DetailedExperimentConfig):
        self.config = config
        self.results = []
        self.critical_points = []
        self.phase_transitions = []
        
    def generate_parameter_space(self) -> List[Dict]:
        """生成超细参数空间"""
        gamma_values = np.arange(
            self.config.gamma_range[0], 
            self.config.gamma_range[1] + self.config.gamma_step/2, 
            self.config.gamma_step
        )
        beta_values = np.arange(
            self.config.beta_range[0], 
            self.config.beta_range[1] + self.config.beta_step/2, 
            self.config.beta_step
        )
        p_up_values = np.arange(
            self.config.p_up_range[0], 
            self.config.p_up_range[1] + self.config.p_up_step/2, 
            self.config.p_up_step
        )
        r_values = np.arange(
            self.config.r_range[0], 
            self.config.r_range[1] + self.config.r_step/2, 
            self.config.r_step
        )
        
        print(f"参数空间大小:")
        print(f"gamma: {len(gamma_values)} 个值 ({gamma_values[0]:.2f} - {gamma_values[-1]:.2f})")
        print(f"beta: {len(beta_values)} 个值 ({beta_values[0]:.2f} - {beta_values[-1]:.2f})")
        print(f"p_up: {len(p_up_values)} 个值 ({p_up_values[0]:.3f} - {p_up_values[-1]:.3f})")
        print(f"r: {len(r_values)} 个值 ({r_values[0]:.3f} - {r_values[-1]:.3f})")
        print(f"总参数组合数: {len(gamma_values) * len(beta_values) * len(p_up_values) * len(r_values)}")
        print(f"总实验次数: {len(gamma_values) * len(beta_values) * len(p_up_values) * len(r_values) * self.config.n_replicates}")
        
        parameter_combinations = []
        for gamma in gamma_values:
            for beta in beta_values:
                for p_up in p_up_values:
                    for r in r_values:
                        parameter_combinations.append({
                            'gamma': gamma,
                            'beta': beta,
                            'p_up': p_up,
                            'r': r
                        })
        
        return parameter_combinations
    
    def run_single_detailed_experiment(self, params: Dict, replicate_id: int) -> Dict:
        """运行单个详细实验"""
        try:
            simulator = SiyanSimulator(
                grid_size=self.config.grid_size,
                gamma=params['gamma'],
                beta=params['beta'],
                p_up=params['p_up'],
                r=params['r']
            )
            
            # 详细记录
            detailed_history = {
                'steps': [],
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
                'mutation_events': [],
                'collapse_events': [],
                'recovery_events': []
            }
            
            collapse_detected = False
            recovery_detected = False
            collapse_step = None
            recovery_step = None
            
            for step in range(self.config.steps):
                simulator.simulation_step()
                
                if step % self.config.checkpoint_interval == 0:
                    current_state = simulator.get_current_state()
                    
                    detailed_history['steps'].append(step)
                    detailed_history['alive_ratio'].append(current_state['alive_ratio'])
                    detailed_history['c_mean'].append(current_state['c_mean'])
                    detailed_history['c_variance'].append(current_state['c_variance'])
                    detailed_history['p_mean_serial'].append(current_state['p_mean_serial'])
                    detailed_history['p_variance'].append(current_state['p_variance'])
                    detailed_history['pc_serial'].append(current_state['pc_serial'])
                    detailed_history['pc_env'].append(current_state['pc_env'])
                    detailed_history['robustness_mean'].append(current_state['robustness_mean'])
                    detailed_history['robustness_variance'].append(current_state['robustness_variance'])
                    detailed_history['energy_mean'].append(current_state['energy_mean'])
                    detailed_history['energy_variance'].append(current_state['energy_variance'])
                    detailed_history['birth_rate'].append(current_state['birth_rate'])
                    detailed_history['death_rate'].append(current_state['death_rate'])
                    detailed_history['mutation_events'].append(current_state['mutation_events'])
                    
                    # 检测崩溃和恢复事件
                    if step > 0:
                        prev_alive = detailed_history['alive_ratio'][-2] if len(detailed_history['alive_ratio']) > 1 else 1.0
                        curr_alive = current_state['alive_ratio']
                        
                        # 崩溃检测：存活率急剧下降
                        if prev_alive > 0.8 and curr_alive < 0.3 and not collapse_detected:
                            collapse_detected = True
                            collapse_step = step
                            detailed_history['collapse_events'].append(step)
                        
                        # 恢复检测：从崩溃状态恢复
                        if collapse_detected and curr_alive > 0.7 and not recovery_detected:
                            recovery_detected = True
                            recovery_step = step
                            detailed_history['recovery_events'].append(step)
            
            # 计算详细统计
            final_alive_ratio = detailed_history['alive_ratio'][-1] if detailed_history['alive_ratio'] else 0
            max_c_mean = max(detailed_history['c_mean']) if detailed_history['c_mean'] else 0
            min_pc_serial = min(detailed_history['pc_serial']) if detailed_history['pc_serial'] else 0
            max_pc_serial = max(detailed_history['pc_serial']) if detailed_history['pc_serial'] else 0
            
            # 计算波动性指标
            c_volatility = np.std(detailed_history['c_mean']) if detailed_history['c_mean'] else 0
            alive_volatility = np.std(detailed_history['alive_ratio']) if detailed_history['alive_ratio'] else 0
            
            return {
                'params': params,
                'replicate_id': replicate_id,
                'final_alive_ratio': final_alive_ratio,
                'max_c_mean': max_c_mean,
                'min_pc_serial': min_pc_serial,
                'max_pc_serial': max_pc_serial,
                'c_volatility': c_volatility,
                'alive_volatility': alive_volatility,
                'collapse_detected': collapse_detected,
                'collapse_step': collapse_step,
                'recovery_detected': recovery_detected,
                'recovery_step': recovery_step,
                'detailed_history': detailed_history,
                'success': True
            }
            
        except Exception as e:
            return {
                'params': params,
                'replicate_id': replicate_id,
                'error': str(e),
                'success': False
            }
    
    def run_parallel_scan(self, max_workers: int = 4) -> pd.DataFrame:
        """并行运行超细扫描"""
        parameter_combinations = self.generate_parameter_space()
        total_combinations = len(parameter_combinations)
        
        print(f"开始超细颗粒度参数扫描...")
        print(f"总参数组合: {total_combinations}")
        print(f"每个组合重复: {self.config.n_replicates} 次")
        print(f"总实验次数: {total_combinations * self.config.n_replicates}")
        print(f"并行工作进程: {max_workers}")
        
        all_tasks = []
        for params in parameter_combinations:
            for replicate_id in range(self.config.n_replicates):
                all_tasks.append((params, replicate_id))
        
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.run_single_detailed_experiment, params, rep_id) 
                      for params, rep_id in all_tasks]
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), desc="运行实验"):
                result = future.result()
                results.append(result)
        
        # 转换为DataFrame
        df_data = []
        for result in results:
            if result['success']:
                row = {
                    'gamma': result['params']['gamma'],
                    'beta': result['params']['beta'],
                    'p_up': result['params']['p_up'],
                    'r': result['params']['r'],
                    'replicate_id': result['replicate_id'],
                    'final_alive_ratio': result['final_alive_ratio'],
                    'max_c_mean': result['max_c_mean'],
                    'min_pc_serial': result['min_pc_serial'],
                    'max_pc_serial': result['max_pc_serial'],
                    'c_volatility': result['c_volatility'],
                    'alive_volatility': result['alive_volatility'],
                    'collapse_detected': result['collapse_detected'],
                    'collapse_step': result['collapse_step'],
                    'recovery_detected': result['recovery_detected'],
                    'recovery_step': result['recovery_step']
                }
                df_data.append(row)
        
        self.results_df = pd.DataFrame(df_data)
        return self.results_df
    
    def identify_critical_transitions(self) -> Dict:
        """识别临界相变点"""
        if self.results_df is None:
            raise ValueError("需要先运行参数扫描")
        
        critical_points = {}
        
        # 按参数分组分析
        for param in ['gamma', 'beta', 'p_up', 'r']:
            param_values = sorted(self.results_df[param].unique())
            
            # 计算每个参数值的平均指标
            avg_metrics = {}
            for value in param_values:
                subset = self.results_df[self.results_df[param] == value]
                avg_metrics[value] = {
                    'final_alive_ratio': subset['final_alive_ratio'].mean(),
                    'max_c_mean': subset['max_c_mean'].mean(),
                    'c_volatility': subset['c_volatility'].mean(),
                    'alive_volatility': subset['alive_volatility'].mean(),
                    'collapse_rate': subset['collapse_detected'].mean()
                }
            
            # 寻找急剧变化的点
            critical_points[param] = self._find_critical_points(avg_metrics, param_values)
        
        return critical_points
    
    def _find_critical_points(self, avg_metrics: Dict, param_values: List[float]) -> List[Dict]:
        """寻找临界点"""
        critical_points = []
        
        metrics_to_check = ['final_alive_ratio', 'max_c_mean', 'c_volatility', 'collapse_rate']
        
        for i in range(1, len(param_values)):
            prev_value = param_values[i-1]
            curr_value = param_values[i]
            
            for metric in metrics_to_check:
                prev_metric = avg_metrics[prev_value][metric]
                curr_metric = avg_metrics[curr_value][metric]
                
                # 计算相对变化
                if prev_metric != 0:
                    relative_change = abs(curr_metric - prev_metric) / prev_metric
                    
                    # 如果变化超过阈值，认为是临界点
                    if relative_change > 0.3:  # 30%变化阈值
                        critical_points.append({
                            'parameter_value': curr_value,
                            'metric': metric,
                            'change_magnitude': relative_change,
                            'before_value': prev_metric,
                            'after_value': curr_metric
                        })
        
        return critical_points
    
    def plot_phase_diagrams(self, output_dir: str = "ultra_fine_phase_diagrams"):
        """绘制相图"""
        if self.results_df is None:
            raise ValueError("需要先运行参数扫描")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建平均结果（跨重复实验）
        avg_results = self.results_df.groupby(['gamma', 'beta', 'p_up', 'r']).agg({
            'final_alive_ratio': 'mean',
            'max_c_mean': 'mean',
            'c_volatility': 'mean',
            'alive_volatility': 'mean',
            'collapse_rate': 'mean'
        }).reset_index()
        
        # 固定p_up和r，绘制gamma-beta相图
        unique_p_up = sorted(avg_results['p_up'].unique())
        unique_r = sorted(avg_results['r'].unique())
        
        for p_up in unique_p_up[:3]:  # 只画前3个p_up值
            for r in unique_r[:3]:     # 只画前3个r值
                subset = avg_results[
                    (avg_results['p_up'] == p_up) & 
                    (avg_results['r'] == r)
                ]
                
                if len(subset) > 0:
                    self._plot_2d_phase_diagram(
                        subset, 'gamma', 'beta', 'final_alive_ratio',
                        f"存活率相图 (p_up={p_up:.3f}, r={r:.3f})",
                        os.path.join(output_dir, f"phase_diagram_alive_pup{p_up:.3f}_r{r:.3f}.png")
                    )
                    
                    self._plot_2d_phase_diagram(
                        subset, 'gamma', 'beta', 'collapse_rate',
                        f"崩溃率相图 (p_up={p_up:.3f}, r={r:.3f})",
                        os.path.join(output_dir, f"phase_diagram_collapse_pup{p_up:.3f}_r{r:.3f}.png")
                    )
    
    def _plot_2d_phase_diagram(self, data: pd.DataFrame, x_param: str, y_param: str, 
                               z_metric: str, title: str, output_file: str):
        """绘制2D相图"""
        pivot_data = data.pivot_table(
            values=z_metric, 
            index=y_param, 
            columns=x_param, 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=False, cmap='viridis', cbar_kws={'label': z_metric})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(x_param, fontsize=12)
        plt.ylabel(y_param, fontsize=12)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"相图已保存: {output_file}")
    
    def save_results(self, output_dir: str = "ultra_fine_scan_results"):
        """保存详细结果"""
        if self.results_df is None:
            raise ValueError("需要先运行参数扫描")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存主要结果
        results_file = os.path.join(output_dir, f"ultra_fine_results_{timestamp}.csv")
        self.results_df.to_csv(results_file, index=False)
        
        # 保存配置
        config_dict = {
            'grid_size': self.config.grid_size,
            'steps': self.config.steps,
            'gamma_range': self.config.gamma_range,
            'beta_range': self.config.beta_range,
            'p_up_range': self.config.p_up_range,
            'r_range': self.config.r_range,
            'gamma_step': self.config.gamma_step,
            'beta_step': self.config.beta_step,
            'p_up_step': self.config.p_up_step,
            'r_step': self.config.r_step,
            'n_replicates': self.config.n_replicates,
            'total_experiments': len(self.results_df)
        }
        
        config_file = os.path.join(output_dir, f"ultra_fine_config_{timestamp}.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        # 识别临界转换
        try:
            critical_points = self.identify_critical_transitions()
            critical_file = os.path.join(output_dir, f"critical_points_{timestamp}.json")
            with open(critical_file, 'w', encoding='utf-8') as f:
                json.dump(critical_points, f, indent=2, ensure_ascii=False)
            print(f"临界转换点已保存: {critical_file}")
        except Exception as e:
            print(f"识别临界转换点时出错: {e}")
        
        print(f"结果已保存到: {output_dir}")
        print(f"主要结果文件: {results_file}")
        print(f"配置文件: {config_file}")
        
        return results_file, config_file

def main():
    """主函数"""
    # 创建详细配置
    config = DetailedExperimentConfig(
        grid_size=35,
        steps=3000,
        gamma_range=(1.0, 2.2),
        beta_range=(0.1, 0.9),
        p_up_range=(0.01, 0.12),
        r_range=(0.92, 0.999),
        gamma_step=0.05,
        beta_step=0.02,
        p_up_step=0.005,
        r_step=0.002,
        n_replicates=5,
        checkpoint_interval=100
    )
    
    # 创建扫描器
    scanner = UltraFineDetailedScanner(config)
    
    # 运行扫描
    print("开始超细颗粒度参数扫描...")
    results_df = scanner.run_parallel_scan(max_workers=6)
    
    print(f"扫描完成！共收集 {len(results_df)} 个数据点")
    
    # 保存结果
    results_file, config_file = scanner.save_results()
    
    # 绘制相图
    print("开始绘制相图...")
    scanner.plot_phase_diagrams()
    
    print("所有任务完成！")
    print(f"结果文件: {results_file}")
    print(f"配置文件: {config_file}")

if __name__ == "__main__":
    main()