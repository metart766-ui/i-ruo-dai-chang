#!/usr/bin/env python3
"""
实时参数敏感性分析器 - 轻量级版本
用于快速分析参数空间的敏感性和相变行为
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import griddata
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置出版级质量
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

class QuickParameterSensitivityAnalyzer:
    """快速参数敏感性分析器"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        self.sensitivity_metrics = {}
        self.phase_transitions = {}
        
    def quick_sensitivity_analysis(self) -> Dict:
        """快速敏感性分析"""
        print("开始快速敏感性分析...")
        
        parameters = ['gamma', 'beta', 'p_up', 'r']
        metrics = ['final_alive_ratio', 'collapse_rate', 'max_c_mean', 'c_volatility']
        
        sensitivity_summary = {}
        
        for metric in metrics:
            sensitivity_summary[metric] = {}
            print(f"分析指标: {metric}")
            
            for param in parameters:
                # 获取参数值和对应的指标值
                param_values = self.results_df[param].unique()
                param_values = sorted(param_values)
                
                if len(param_values) < 3:
                    continue
                
                # 计算每个参数值的平均指标
                mean_values = []
                for value in param_values:
                    subset = self.results_df[self.results_df[param] == value][metric]
                    mean_values.append(subset.mean())
                
                mean_values = np.array(mean_values)
                
                # 计算相关系数敏感性
                correlation, _ = stats.pearsonr(self.results_df[param], self.results_df[metric])
                correlation_sensitivity = abs(correlation)
                
                # 计算梯度敏感性
                gradients = np.gradient(mean_values, param_values)
                gradient_sensitivity = np.mean(np.abs(gradients))
                
                sensitivity_summary[metric][param] = {
                    'correlation_sensitivity': correlation_sensitivity,
                    'gradient_sensitivity': gradient_sensitivity,
                    'mean_values': mean_values.tolist(),
                    'param_values': param_values
                }
        
        self.sensitivity_metrics = sensitivity_summary
        return sensitivity_summary
    
    def quick_phase_transition_detection(self) -> Dict:
        """快速相变检测"""
        print("开始快速相变检测...")
        
        parameters = ['gamma', 'beta']
        metrics = ['final_alive_ratio', 'collapse_rate']
        
        transition_summary = {}
        
        for param in parameters:
            transition_summary[param] = {}
            
            for metric in metrics:
                # 获取参数值和对应的指标值
                param_values = sorted(self.results_df[param].unique())
                
                if len(param_values) < 5:
                    continue
                
                # 计算每个参数值的平均指标
                mean_values = []
                for value in param_values:
                    subset = self.results_df[self.results_df[param] == value][metric]
                    mean_values.append(subset.mean())
                
                mean_values = np.array(mean_values)
                
                # 检测相变点
                transitions = []
                
                # 方法1: 存活率的急剧下降
                if metric == 'final_alive_ratio':
                    for i in range(1, len(param_values)):
                        if mean_values[i-1] > 0.7 and mean_values[i] < 0.3:
                            drop_magnitude = mean_values[i-1] - mean_values[i]
                            transitions.append({
                                'type': 'sharp_drop',
                                'param_value': param_values[i],
                                'drop_magnitude': drop_magnitude
                            })
                
                # 方法2: 崩溃率的急剧上升
                if metric == 'collapse_rate':
                    for i in range(1, len(param_values)):
                        if mean_values[i-1] < 0.2 and mean_values[i] > 0.5:
                            rise_magnitude = mean_values[i] - mean_values[i-1]
                            transitions.append({
                                'type': 'sharp_rise',
                                'param_value': param_values[i],
                                'rise_magnitude': rise_magnitude
                            })
                
                transition_summary[param][metric] = {
                    'transitions': transitions,
                    'mean_values': mean_values.tolist(),
                    'param_values': param_values
                }
        
        self.phase_transitions = transition_summary
        return transition_summary
    
    def create_quick_heatmaps(self, output_dir: str = "quick_sensitivity_heatmaps"):
        """创建快速敏感性热力图"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算敏感性指标
        sensitivity_data = self.quick_sensitivity_analysis()
        
        # 创建敏感性矩阵
        parameters = ['gamma', 'beta', 'p_up', 'r']
        metrics = ['final_alive_ratio', 'collapse_rate', 'max_c_mean', 'c_volatility']
        
        # 相关系数敏感性热力图
        sensitivity_matrix = np.zeros((len(metrics), len(parameters)))
        
        for i, metric in enumerate(metrics):
            for j, param in enumerate(parameters):
                if param in sensitivity_data[metric]:
                    sensitivity_matrix[i, j] = sensitivity_data[metric][param]['correlation_sensitivity']
                else:
                    sensitivity_matrix[i, j] = 0
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sensitivity_matrix, 
                   xticklabels=parameters,
                   yticklabels=[m.replace('_', ' ').title() for m in metrics],
                   annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Correlation Sensitivity'})
        
        plt.title('Parameter Sensitivity Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Parameters', fontsize=14)
        plt.ylabel('Metrics', fontsize=14)
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 'quick_sensitivity_heatmap.png')
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"快速敏感性热力图已保存: {output_file}")
        return output_file
    
    def create_quick_phase_plots(self, output_dir: str = "quick_phase_plots"):
        """创建快速相变图"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 识别相变
        phase_data = self.quick_phase_transition_detection()
        
        for param in phase_data.keys():
            for metric in phase_data[param].keys():
                if not phase_data[param][metric]['transitions']:
                    continue
                
                param_values = phase_data[param][metric]['param_values']
                mean_values = phase_data[param][metric]['mean_values']
                transitions = phase_data[param][metric]['transitions']
                
                plt.figure(figsize=(12, 8))
                
                # 主图：参数响应曲线
                plt.plot(param_values, mean_values, 'o-', linewidth=3, markersize=8, 
                        color='blue', label='Mean Value')
                
                # 标记相变点
                for transition in transitions:
                    if transition['type'] == 'sharp_drop':
                        plt.axvline(x=transition['param_value'], color='red', 
                                   linestyle='--', linewidth=2, alpha=0.8,
                                   label=f"Sharp Drop: {transition['drop_magnitude']:.3f}")
                    elif transition['type'] == 'sharp_rise':
                        plt.axvline(x=transition['param_value'], color='orange', 
                                   linestyle='--', linewidth=2, alpha=0.8,
                                   label=f"Sharp Rise: {transition['rise_magnitude']:.3f}")
                
                plt.xlabel(param, fontsize=14)
                plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
                plt.title(f'Quick Phase Transition: {param} vs {metric.replace("_", " ").title()}',
                         fontsize=16, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                
                output_file = os.path.join(output_dir, f'quick_phase_{param}_{metric}.png')
                plt.savefig(output_file, dpi=600, bbox_inches='tight')
                plt.close()
                
                print(f"快速相变图已保存: {output_file}")
                return output_file
    
    def create_stability_map(self, output_dir: str = "quick_stability_maps"):
        """创建稳定性地图"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建稳定性地图
        gamma_range = np.linspace(self.results_df['gamma'].min(), 
                                 self.results_df['gamma'].max(), 30)
        beta_range = np.linspace(self.results_df['beta'].min(), 
                                self.results_df['beta'].max(), 30)
        
        gamma_grid, beta_grid = np.meshgrid(gamma_range, beta_range)
        
        # 插值得到稳定性指标
        points = self.results_df[['gamma', 'beta']].values
        values = self.results_df['final_alive_ratio'].values
        
        stability_grid = griddata(points, values, (gamma_grid, beta_grid), 
                                   method='linear')
        
        plt.figure(figsize=(12, 10))
        
        # 绘制等高线图
        contour = plt.contourf(gamma_grid, beta_grid, stability_grid, 
                             levels=15, cmap='RdYlGn', alpha=0.8)
        
        # 标记稳定性区域
        plt.contour(gamma_grid, beta_grid, stability_grid, 
                  levels=[0.3, 0.7], colors=['red', 'yellow'], 
                  linewidths=2, alpha=0.8)
        
        # 添加颜色条
        cbar = plt.colorbar(contour)
        cbar.set_label('Survival Rate')
        
        plt.xlabel('γ (Maintenance Cost Exponent)', fontsize=14)
        plt.ylabel('β (Environmental Sensitivity)', fontsize=14)
        plt.title('Quick System Stability Map', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 添加区域标签
        plt.text(0.05, 0.95, 'Stable\nRegion', transform=plt.gca().transAxes,
                fontsize=12, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.text(0.05, 0.05, 'Unstable\nRegion', transform=plt.gca().transAxes,
                fontsize=12, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        output_file = os.path.join(output_dir, 'quick_stability_map.png')
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"快速稳定性地图已保存: {output_file}")
        return output_file
    
    def print_analysis_summary(self):
        """打印分析摘要"""
        print("\n" + "="*60)
        print("快速参数敏感性分析摘要")
        print("="*60)
        
        if self.sensitivity_metrics:
            print("\n1. 敏感性分析结果:")
            for metric in ['final_alive_ratio', 'collapse_rate']:
                print(f"\n   {metric}:")
                for param in ['gamma', 'beta', 'p_up', 'r']:
                    if param in self.sensitivity_metrics[metric]:
                        corr_sens = self.sensitivity_metrics[metric][param]['correlation_sensitivity']
                        grad_sens = self.sensitivity_metrics[metric][param]['gradient_sensitivity']
                        print(f"     {param}: 相关系数={corr_sens:.3f}, 梯度={grad_sens:.3f}")
        
        if self.phase_transitions:
            print("\n2. 相变检测结果:")
            for param in ['gamma', 'beta']:
                for metric in ['final_alive_ratio', 'collapse_rate']:
                    if param in self.phase_transitions and metric in self.phase_transitions[param]:
                        transitions = self.phase_transitions[param][metric]['transitions']
                        if transitions:
                            print(f"\n   {param} - {metric}:")
                            for transition in transitions:
                                if transition['type'] == 'sharp_drop':
                                    print(f"     急剧下降点: {param}={transition['param_value']:.3f}, "
                                          f"下降幅度={transition['drop_magnitude']:.3f}")
                                elif transition['type'] == 'sharp_rise':
                                    print(f"     急剧上升点: {param}={transition['param_value']:.3f}, "
                                          f"上升幅度={transition['rise_magnitude']:.3f}")
                        else:
                            print(f"   {param} - {metric}: 未检测到明显相变")
        
        print("\n" + "="*60)

def main():
    """主函数"""
    # 加载数据
    import glob
    result_files = glob.glob("ultra_fine_scan_results/ultra_fine_results_*.csv")
    
    if not result_files:
        print("未找到超细扫描结果文件，尝试加载其他结果文件...")
        result_files = glob.glob("siyan_results.csv")
        
        if not result_files:
            print("未找到任何结果文件")
            return
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"加载数据: {latest_file}")
    
    results_df = pd.read_csv(latest_file)
    print(f"数据形状: {results_df.shape}")
    print(f"可用列: {results_df.columns.tolist()}")
    
    # 创建快速分析器
    analyzer = QuickParameterSensitivityAnalyzer(results_df)
    
    # 运行快速分析
    print("开始快速参数敏感性分析...")
    
    # 1. 快速敏感性分析
    print("\n1. 快速敏感性分析...")
    sensitivity_data = analyzer.quick_sensitivity_analysis()
    
    # 2. 快速相变检测
    print("\n2. 快速相变检测...")
    phase_data = analyzer.quick_phase_transition_detection()
    
    # 3. 创建可视化
    print("\n3. 创建快速可视化...")
    
    print("创建敏感性热力图...")
    analyzer.create_quick_heatmaps()
    
    print("创建相变图...")
    analyzer.create_quick_phase_plots()
    
    print("创建稳定性地图...")
    analyzer.create_stability_map()
    
    # 4. 打印摘要
    analyzer.print_analysis_summary()
    
    print("\n快速参数敏感性分析完成！")

if __name__ == "__main__":
    main()