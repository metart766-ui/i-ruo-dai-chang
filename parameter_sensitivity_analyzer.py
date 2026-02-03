#!/usr/bin/env python3
"""
参数敏感性热力图分析器
专门用于分析参数空间的敏感性和相变行为
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.interpolate import griddata
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import json
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

class ParameterSensitivityAnalyzer:
    """参数敏感性分析器"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        self.sensitivity_metrics = {}
        self.phase_transitions = {}
        self.critical_exponents = {}
        
    def calculate_sensitivity_indices(self) -> Dict:
        """计算敏感性指标"""
        parameters = ['gamma', 'beta', 'p_up', 'r']
        metrics = ['final_alive_ratio', 'collapse_rate', 'max_c_mean', 'c_volatility']
        
        sensitivity_results = {}
        
        for metric in metrics:
            sensitivity_results[metric] = {}
            
            for param in parameters:
                # 获取参数值和对应的指标值
                param_values = self.results_df[param].unique()
                param_values = sorted(param_values)
                
                if len(param_values) < 3:
                    continue
                
                # 计算每个参数值的平均指标
                mean_values = []
                std_values = []
                
                for value in param_values:
                    subset = self.results_df[self.results_df[param] == value][metric]
                    mean_values.append(subset.mean())
                    std_values.append(subset.std())
                
                mean_values = np.array(mean_values)
                std_values = np.array(std_values)
                
                # 计算多种敏感性指标
                
                # 1. 基于梯度的敏感性
                gradients = np.gradient(mean_values, param_values)
                gradient_sensitivity = np.mean(np.abs(gradients))
                
                # 2. 基于方差的敏感性
                variance_sensitivity = np.var(mean_values) / np.mean(mean_values) if np.mean(mean_values) != 0 else 0
                
                # 3. 基于相关系数的敏感性
                correlation, _ = stats.pearsonr(self.results_df[param], self.results_df[metric])
                correlation_sensitivity = abs(correlation)
                
                # 4. 基于 Sobol 指数的近似
                # 计算一阶 Sobol 指数的近似值
                total_variance = np.var(self.results_df[metric])
                param_variance = np.var(mean_values)
                sobol_first_order = param_variance / total_variance if total_variance != 0 else 0
                
                # 5. 基于导数的最大值（寻找最敏感的区域）
                max_gradient = np.max(np.abs(gradients))
                
                sensitivity_results[metric][param] = {
                    'gradient_sensitivity': gradient_sensitivity,
                    'variance_sensitivity': variance_sensitivity,
                    'correlation_sensitivity': correlation_sensitivity,
                    'sobol_first_order': sobol_first_order,
                    'max_gradient': max_gradient,
                    'mean_values': mean_values.tolist(),
                    'param_values': param_values,
                    'std_values': std_values.tolist()
                }
        
        self.sensitivity_metrics = sensitivity_results
        return sensitivity_results
    
    def identify_phase_transitions(self) -> Dict:
        """识别相变点"""
        parameters = ['gamma', 'beta', 'p_up', 'r']
        metrics = ['final_alive_ratio', 'collapse_rate']
        
        phase_transitions = {}
        
        for param in parameters:
            phase_transitions[param] = {}
            
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
                
                # 寻找相变点
                transitions = self._detect_phase_transitions(param_values, mean_values, metric)
                
                phase_transitions[param][metric] = {
                    'transitions': transitions,
                    'mean_values': mean_values.tolist(),
                    'param_values': param_values
                }
        
        self.phase_transitions = phase_transitions
        return phase_transitions
    
    def _detect_phase_transitions(self, param_values: List[float], mean_values: List[float], 
                                 metric: str) -> List[Dict]:
        """检测相变点"""
        transitions = []
        
        # 方法1: 基于二阶导数的突变检测
        if len(param_values) >= 5:
            # 计算二阶导数
            second_derivative = np.gradient(np.gradient(mean_values, param_values), param_values)
            
            # 寻找二阶导数的极值点
            threshold = np.percentile(np.abs(second_derivative), 90)
            
            for i, (param_val, second_deriv) in enumerate(zip(param_values[1:-1], second_derivative[1:-1])):
                if abs(second_deriv) > threshold:
                    transitions.append({
                        'type': 'curvature_change',
                        'param_value': param_val,
                        'second_derivative': second_deriv,
                        'magnitude': abs(second_deriv),
                        'index': i + 1
                    })
        
        # 方法2: 基于方差的变化检测
        if metric == 'final_alive_ratio':
            # 寻找存活率的急剧下降点
            for i in range(1, len(param_values)):
                if mean_values[i-1] > 0.7 and mean_values[i] < 0.3:
                    # 急剧下降
                    drop_magnitude = mean_values[i-1] - mean_values[i]
                    transitions.append({
                        'type': 'sharp_drop',
                        'param_value': param_values[i],
                        'drop_magnitude': drop_magnitude,
                        'from_value': mean_values[i-1],
                        'to_value': mean_values[i]
                    })
        
        # 方法3: 基于崩溃率的急剧上升点
        if metric == 'collapse_rate':
            for i in range(1, len(param_values)):
                if mean_values[i-1] < 0.2 and mean_values[i] > 0.5:
                    # 急剧上升
                    rise_magnitude = mean_values[i] - mean_values[i-1]
                    transitions.append({
                        'type': 'sharp_rise',
                        'param_value': param_values[i],
                        'rise_magnitude': rise_magnitude,
                        'from_value': mean_values[i-1],
                        'to_value': mean_values[i]
                    })
        
        return transitions
    
    def calculate_critical_exponents(self) -> Dict:
        """计算临界指数"""
        critical_exponents = {}
        
        for param in ['gamma', 'beta']:
            critical_exponents[param] = {}
            
            # 获取参数值和存活率
            param_values = sorted(self.results_df[param].unique())
            survival_rates = []
            
            for value in param_values:
                subset = self.results_df[self.results_df[param] == value]['final_alive_ratio']
                survival_rates.append(subset.mean())
            
            param_values = np.array(param_values)
            survival_rates = np.array(survival_rates)
            
            # 寻找临界点附近的数据
            # 假设相变点在存活率急剧变化的区域
            if len(param_values) >= 5:
                # 寻找存活率最接近0.5的点作为临界点
                critical_idx = np.argmin(np.abs(survival_rates - 0.5))
                critical_param = param_values[critical_idx]
                
                # 拟合临界指数
                exponents = self._fit_critical_exponents(param_values, survival_rates, critical_param)
                
                critical_exponents[param] = {
                    'critical_value': critical_param,
                    'critical_index': critical_idx,
                    'exponents': exponents,
                    'param_values': param_values.tolist(),
                    'survival_rates': survival_rates.tolist()
                }
        
        self.critical_exponents = critical_exponents
        return critical_exponents
    
    def _fit_critical_exponents(self, param_values: np.ndarray, survival_rates: np.ndarray, 
                               critical_param: float) -> Dict:
        """拟合临界指数"""
        exponents = {}
        
        # 分离临界点两侧的数据
        left_mask = param_values < critical_param
        right_mask = param_values > critical_param
        
        # 左侧数据（存活相）
        if np.sum(left_mask) >= 3:
            left_params = param_values[left_mask]
            left_rates = survival_rates[left_mask]
            
            # 计算距离临界点的距离
            distances = critical_param - left_params
            
            # 拟合幂律：survival_rate ~ (critical_param - param)^beta
            try:
                # 只使用距离临界点不太远的数据
                valid_mask = distances > 0
                if np.sum(valid_mask) >= 3:
                    log_distances = np.log(distances[valid_mask])
                    log_rates = np.log(left_rates[valid_mask])
                    
                    # 线性回归拟合
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_distances, log_rates)
                    
                    exponents['beta_left'] = {
                        'value': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'std_error': std_err
                    }
            except:
                pass
        
        # 右侧数据（崩溃相）
        if np.sum(right_mask) >= 3:
            right_params = param_values[right_mask]
            right_rates = survival_rates[right_mask]
            
            # 计算距离临界点的距离
            distances = right_params - critical_param
            
            # 拟合幂律：survival_rate ~ (param - critical_param)^beta
            try:
                valid_mask = distances > 0
                if np.sum(valid_mask) >= 3:
                    log_distances = np.log(distances[valid_mask])
                    log_rates = np.log(right_rates[valid_mask])
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_distances, log_rates)
                    
                    exponents['beta_right'] = {
                        'value': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'std_error': std_err
                    }
            except:
                pass
        
        return exponents
    
    def create_sensitivity_heatmaps(self, output_dir: str = "sensitivity_heatmaps"):
        """创建敏感性热力图"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算敏感性指标
        sensitivity_data = self.calculate_sensitivity_indices()
        
        # 创建敏感性矩阵
        parameters = ['gamma', 'beta', 'p_up', 'r']
        metrics = ['final_alive_ratio', 'collapse_rate', 'max_c_mean', 'c_volatility']
        
        for sensitivity_type in ['gradient_sensitivity', 'variance_sensitivity', 
                               'correlation_sensitivity', 'sobol_first_order']:
            
            sensitivity_matrix = np.zeros((len(metrics), len(parameters)))
            
            for i, metric in enumerate(metrics):
                for j, param in enumerate(parameters):
                    if param in sensitivity_data[metric]:
                        sensitivity_matrix[i, j] = sensitivity_data[metric][param][sensitivity_type]
                    else:
                        sensitivity_matrix[i, j] = 0
            
            # 创建热力图
            plt.figure(figsize=(12, 10))
            sns.heatmap(sensitivity_matrix, 
                       xticklabels=parameters,
                       yticklabels=[m.replace('_', ' ').title() for m in metrics],
                       annot=True, fmt='.3f', cmap='YlOrRd',
                       cbar_kws={'label': sensitivity_type.replace('_', ' ').title()})
            
            plt.title(f'Parameter Sensitivity Heatmap: {sensitivity_type.replace("_", " ").title()}',
                     fontsize=16, fontweight='bold')
            plt.xlabel('Parameters', fontsize=14)
            plt.ylabel('Metrics', fontsize=14)
            plt.tight_layout()
            
            output_file = os.path.join(output_dir, f'sensitivity_{sensitivity_type}.png')
            plt.savefig(output_file, dpi=600, bbox_inches='tight')
            plt.close()
            
            print(f"敏感性热力图已保存: {output_file}")
    
    def create_phase_transition_plots(self, output_dir: str = "phase_transitions"):
        """创建相变图"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 识别相变
        phase_data = self.identify_phase_transitions()
        
        for param in phase_data.keys():
            for metric in phase_data[param].keys():
                if not phase_data[param][metric]['transitions']:
                    continue
                
                param_values = phase_data[param][metric]['param_values']
                mean_values = phase_data[param][metric]['mean_values']
                transitions = phase_data[param][metric]['transitions']
                
                plt.figure(figsize=(14, 10))
                
                # 主图：参数响应曲线
                plt.subplot(2, 1, 1)
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
                    elif transition['type'] == 'curvature_change':
                        plt.axvline(x=transition['param_value'], color='green', 
                                   linestyle=':', linewidth=2, alpha=0.8,
                                   label=f"Curvature Change: {transition['magnitude']:.3f}")
                
                plt.xlabel(param, fontsize=14)
                plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
                plt.title(f'Phase Transition Analysis: {param} vs {metric.replace("_", " ").title()}',
                         fontsize=16, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                
                # 子图：一阶导数
                plt.subplot(2, 1, 2)
                if len(param_values) >= 3:
                    gradients = np.gradient(mean_values, param_values)
                    plt.plot(param_values, gradients, 's-', color='purple', 
                            linewidth=2, markersize=6, label='First Derivative')
                    
                    # 标记导数极值点
                    max_grad_idx = np.argmax(np.abs(gradients))
                    plt.axvline(x=param_values[max_grad_idx], color='red', 
                               linestyle=':', linewidth=2, alpha=0.8,
                               label=f'Max Gradient: {gradients[max_grad_idx]:.3f}')
                    
                    plt.xlabel(param, fontsize=14)
                    plt.ylabel('First Derivative', fontsize=14)
                    plt.title('First Derivative Analysis', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                
                plt.tight_layout()
                
                output_file = os.path.join(output_dir, f'phase_transition_{param}_{metric}.png')
                plt.savefig(output_file, dpi=600, bbox_inches='tight')
                plt.close()
                
                print(f"相变图已保存: {output_file}")
    
    def create_critical_exponent_plots(self, output_dir: str = "critical_exponents"):
        """创建临界指数图"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算临界指数
        exponent_data = self.calculate_critical_exponents()
        
        for param in exponent_data.keys():
            if not exponent_data[param]:
                continue
            
            param_values = np.array(exponent_data[param]['param_values'])
            survival_rates = np.array(exponent_data[param]['survival_rates'])
            critical_value = exponent_data[param]['critical_value']
            exponents = exponent_data[param]['exponents']
            
            plt.figure(figsize=(15, 10))
            
            # 主图：原始数据和对数图
            plt.subplot(2, 2, 1)
            plt.plot(param_values, survival_rates, 'o-', linewidth=2, markersize=8,
                    color='blue', label='Original Data')
            plt.axvline(x=critical_value, color='red', linestyle='--', 
                       linewidth=2, alpha=0.8, label=f'Critical Point: {critical_value:.3f}')
            plt.xlabel(param, fontsize=12)
            plt.ylabel('Survival Rate', fontsize=12)
            plt.title('Original Data with Critical Point', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 对数图 - 左侧
            if 'beta_left' in exponents:
                plt.subplot(2, 2, 2)
                left_mask = param_values < critical_value
                if np.sum(left_mask) >= 3:
                    left_params = param_values[left_mask]
                    left_rates = survival_rates[left_mask]
                    distances = critical_value - left_params
                    
                    valid_mask = distances > 0
                    if np.sum(valid_mask) >= 3:
                        log_distances = np.log(distances[valid_mask])
                        log_rates = np.log(left_rates[valid_mask])
                        
                        plt.plot(log_distances, log_rates, 'o', color='green', 
                                markersize=8, label='Data (Left Side)')
                        
                        # 拟合线
                        slope = exponents['beta_left']['value']
                        intercept = np.mean(log_rates - slope * log_distances)
                        fit_line = slope * log_distances + intercept
                        
                        plt.plot(log_distances, fit_line, 'r-', linewidth=2,
                                label=f'Power Law Fit: β={slope:.3f}')
                        
                        plt.xlabel(f'ln({critical_value:.3f} - {param})', fontsize=12)
                        plt.ylabel('ln(Survival Rate)', fontsize=12)
                        plt.title(f'Log-Log Plot (Left Side): R²={exponents["beta_left"]["r_squared"]:.3f}',
                                 fontsize=14)
                        plt.grid(True, alpha=0.3)
                        plt.legend()
            
            # 对数图 - 右侧
            if 'beta_right' in exponents:
                plt.subplot(2, 2, 3)
                right_mask = param_values > critical_value
                if np.sum(right_mask) >= 3:
                    right_params = param_values[right_mask]
                    right_rates = survival_rates[right_mask]
                    distances = right_params - critical_value
                    
                    valid_mask = distances > 0
                    if np.sum(valid_mask) >= 3:
                        log_distances = np.log(distances[valid_mask])
                        log_rates = np.log(right_rates[valid_mask])
                        
                        plt.plot(log_distances, log_rates, 'o', color='orange', 
                                markersize=8, label='Data (Right Side)')
                        
                        # 拟合线
                        slope = exponents['beta_right']['value']
                        intercept = np.mean(log_rates - slope * log_distances)
                        fit_line = slope * log_distances + intercept
                        
                        plt.plot(log_distances, fit_line, 'r-', linewidth=2,
                                label=f'Power Law Fit: β={slope:.3f}')
                        
                        plt.xlabel(f'ln({param} - {critical_value:.3f})', fontsize=12)
                        plt.ylabel('ln(Survival Rate)', fontsize=12)
                        plt.title(f'Log-Log Plot (Right Side): R²={exponents["beta_right"]["r_squared"]:.3f}',
                                 fontsize=14)
                        plt.grid(True, alpha=0.3)
                        plt.legend()
            
            # 残差分析
            plt.subplot(2, 2, 4)
            # 简单的多项式拟合残差
            try:
                z = np.polyfit(param_values, survival_rates, 3)
                p = np.poly1d(z)
                fitted_values = p(param_values)
                residuals = survival_rates - fitted_values
                
                plt.plot(param_values, residuals, 'o-', color='purple', 
                        markersize=6, linewidth=1)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                plt.xlabel(param, fontsize=12)
                plt.ylabel('Residuals', fontsize=12)
                plt.title('Residuals from Cubic Fit', fontsize=14)
                plt.grid(True, alpha=0.3)
            except:
                pass
            
            plt.tight_layout()
            
            output_file = os.path.join(output_dir, f'critical_exponents_{param}.png')
            plt.savefig(output_file, dpi=600, bbox_inches='tight')
            plt.close()
            
            print(f"临界指数图已保存: {output_file}")
    
    def create_comprehensive_sensitivity_report(self, output_dir: str = "sensitivity_report"):
        """创建综合敏感性报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建大综合图
        fig = plt.figure(figsize=(24, 20))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 敏感性热力图汇总
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_sensitivity_summary(ax1)
        
        # 2. 相变点汇总
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_phase_transition_summary(ax2)
        
        # 3. 参数响应曲线
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_parameter_response_curves(ax3)
        
        # 4. 临界指数汇总
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_critical_exponent_summary(ax4)
        
        # 5. 敏感性排名
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_sensitivity_ranking(ax5)
        
        # 6. 不确定性分析
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_uncertainty_analysis(ax6)
        
        # 7. 稳定性区域图
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_stability_regions(ax7)
        
        plt.suptitle('Comprehensive Parameter Sensitivity Analysis Report',
                    fontsize=20, fontweight='bold', y=0.98)
        
        output_file = os.path.join(output_dir, 'comprehensive_sensitivity_report.png')
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"综合敏感性报告已保存: {output_file}")
        return output_file
    
    def _plot_sensitivity_summary(self, ax: plt.Axes):
        """绘制敏感性汇总"""
        if not self.sensitivity_metrics:
            self.calculate_sensitivity_indices()
        
        # 创建敏感性矩阵
        parameters = ['gamma', 'beta', 'p_up', 'r']
        metrics = ['final_alive_ratio', 'collapse_rate', 'max_c_mean', 'c_volatility']
        
        sensitivity_matrix = np.zeros((len(metrics), len(parameters)))
        
        for i, metric in enumerate(metrics):
            for j, param in enumerate(parameters):
                if param in self.sensitivity_metrics[metric]:
                    sensitivity_matrix[i, j] = self.sensitivity_metrics[metric][param]['correlation_sensitivity']
        
        im = ax.imshow(sensitivity_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(parameters)))
        ax.set_yticks(range(len(metrics)))
        ax.set_xticklabels(parameters)
        ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
        
        # 添加数值标签
        for i in range(len(metrics)):
            for j in range(len(parameters)):
                text = ax.text(j, i, f'{sensitivity_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Correlation Sensitivity')
        ax.set_title('Parameter Sensitivity Matrix', fontsize=14, fontweight='bold')
    
    def _plot_phase_transition_summary(self, ax: plt.Axes):
        """绘制相变汇总"""
        if not self.phase_transitions:
            self.identify_phase_transitions()
        
        # 统计相变点
        transition_stats = {}
        for param in self.phase_transitions.keys():
            for metric in self.phase_transitions[param].keys():
                transitions = self.phase_transitions[param][metric]['transitions']
                if transitions:
                    key = f"{param}_{metric}"
                    transition_stats[key] = len(transitions)
        
        if transition_stats:
            keys = list(transition_stats.keys())
            values = list(transition_stats.values())
            
            bars = ax.bar(range(len(keys)), values, color='skyblue')
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=45, ha='right')
            ax.set_ylabel('Number of Phase Transitions')
            ax.set_title('Phase Transition Summary', fontsize=14, fontweight='bold')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       str(value), ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Phase Transitions Detected', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, fontweight='bold')
    
    def _plot_parameter_response_curves(self, ax: plt.Axes):
        """绘制参数响应曲线"""
        # 选择最关键的参数和指标
        param = 'gamma'
        metric = 'final_alive_ratio'
        
        if param in self.sensitivity_metrics and metric in self.sensitivity_metrics[param]:
            param_values = self.sensitivity_metrics[param][metric]['param_values']
            mean_values = self.sensitivity_metrics[param][metric]['mean_values']
            std_values = self.sensitivity_metrics[param][metric]['std_values']
            
            mean_values = np.array(mean_values)
            std_values = np.array(std_values)
            
            # 绘制均值曲线
            ax.plot(param_values, mean_values, 'o-', linewidth=3, markersize=8,
                   color='blue', label='Mean Response')
            
            # 绘制标准差阴影
            ax.fill_between(param_values, mean_values - std_values, 
                           mean_values + std_values, alpha=0.3, color='blue')
            
            # 标记可能的相变区域
            gradients = np.gradient(mean_values, param_values)
            high_gradient_indices = np.where(np.abs(gradients) > np.percentile(np.abs(gradients), 90))[0]
            
            for idx in high_gradient_indices:
                ax.axvline(x=param_values[idx], color='red', linestyle='--', 
                          alpha=0.7, linewidth=2)
            
            ax.set_xlabel(param, fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'Parameter Response: {param} vs {metric.replace("_", " ").title()}',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def _plot_critical_exponent_summary(self, ax: plt.Axes):
        """绘制临界指数汇总"""
        if not self.critical_exponents:
            self.calculate_critical_exponents()
        
        # 收集临界指数
        exponents_data = []
        for param in self.critical_exponents.keys():
            if self.critical_exponents[param] and 'exponents' in self.critical_exponents[param]:
                exponents = self.critical_exponents[param]['exponents']
                for exp_type, exp_data in exponents.items():
                    exponents_data.append({
                        'parameter': param,
                        'exponent_type': exp_type,
                        'value': exp_data['value'],
                        'r_squared': exp_data['r_squared']
                    })
        
        if exponents_data:
            # 创建条形图
            df_exp = pd.DataFrame(exponents_data)
            
            # 按参数分组
            unique_params = df_exp['parameter'].unique()
            x_pos = np.arange(len(unique_params))
            
            colors = ['blue', 'red', 'green', 'orange']
            
            for i, exp_type in enumerate(df_exp['exponent_type'].unique()):
                subset = df_exp[df_exp['exponent_type'] == exp_type]
                values = []
                for param in unique_params:
                    param_subset = subset[subset['parameter'] == param]
                    if len(param_subset) > 0:
                        values.append(param_subset['value'].iloc[0])
                    else:
                        values.append(0)
                
                ax.bar(x_pos + i*0.2, values, 0.2, label=exp_type, 
                      color=colors[i % len(colors)], alpha=0.8)
            
            ax.set_xticks(x_pos + 0.2)
            ax.set_xticklabels(unique_params)
            ax.set_ylabel('Critical Exponent Value')
            ax.set_title('Critical Exponents Summary', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No Critical Exponents Calculated',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, fontweight='bold')
    
    def _plot_sensitivity_ranking(self, ax: plt.Axes):
        """绘制敏感性排名"""
        if not self.sensitivity_metrics:
            self.calculate_sensitivity_indices()
        
        # 计算总体敏感性得分
        parameters = ['gamma', 'beta', 'p_up', 'r']
        overall_sensitivity = {}
        
        for param in parameters:
            scores = []
            for metric in ['final_alive_ratio', 'collapse_rate', 'max_c_mean', 'c_volatility']:
                if param in self.sensitivity_metrics[metric]:
                    scores.append(self.sensitivity_metrics[metric][param]['correlation_sensitivity'])
            
            if scores:
                overall_sensitivity[param] = np.mean(scores)
        
        if overall_sensitivity:
            sorted_params = sorted(overall_sensitivity.keys(), 
                                 key=lambda x: overall_sensitivity[x], reverse=True)
            sorted_scores = [overall_sensitivity[param] for param in sorted_params]
            
            bars = ax.barh(range(len(sorted_params)), sorted_scores, 
                          color=['red', 'orange', 'yellow', 'lightblue'])
            
            ax.set_yticks(range(len(sorted_params)))
            ax.set_yticklabels(sorted_params)
            ax.set_xlabel('Overall Sensitivity Score')
            ax.set_title('Parameter Sensitivity Ranking', fontsize=14, fontweight='bold')
            
            # 添加数值标签
            for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', ha='left', va='center', fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'No Sensitivity Data Available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, fontweight='bold')
    
    def _plot_uncertainty_analysis(self, ax: plt.Axes):
        """绘制不确定性分析"""
        # 计算预测的不确定性
        param = 'gamma'
        metric = 'final_alive_ratio'
        
        if param in self.sensitivity_metrics and metric in self.sensitivity_metrics[param]:
            param_values = self.sensitivity_metrics[param][metric]['param_values']
            mean_values = self.sensitivity_metrics[param][metric]['mean_values']
            std_values = self.sensitivity_metrics[param][metric]['std_values']
            
            # 计算变异系数作为不确定性指标
            cv_values = np.array(std_values) / np.array(mean_values)
            
            ax.plot(param_values, cv_values, 'o-', linewidth=2, markersize=6,
                   color='red', label='Coefficient of Variation')
            
            # 标记高不确定性区域
            high_uncertainty_threshold = np.percentile(cv_values, 75)
            high_uncertainty_indices = np.where(cv_values > high_uncertainty_threshold)[0]
            
            for idx in high_uncertainty_indices:
                ax.axvline(x=param_values[idx], color='orange', linestyle='--', 
                          alpha=0.7, linewidth=2)
            
            ax.set_xlabel(param, fontsize=12)
            ax.set_ylabel('Coefficient of Variation')
            ax.set_title('Uncertainty Analysis', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def _plot_stability_regions(self, ax: plt.Axes):
        """绘制稳定性区域"""
        # 创建稳定性地图
        gamma_range = np.linspace(self.results_df['gamma'].min(), 
                                 self.results_df['gamma'].max(), 50)
        beta_range = np.linspace(self.results_df['beta'].min(), 
                                self.results_df['beta'].max(), 50)
        
        gamma_grid, beta_grid = np.meshgrid(gamma_range, beta_range)
        
        # 插值得到稳定性指标
        points = self.results_df[['gamma', 'beta']].values
        values = self.results_df['final_alive_ratio'].values
        
        stability_grid = griddata(points, values, (gamma_grid, beta_grid), 
                                   method='linear')
        
        # 绘制等高线图
        contour = ax.contourf(gamma_grid, beta_grid, stability_grid, 
                             levels=20, cmap='RdYlGn', alpha=0.8)
        
        # 标记稳定性区域
        ax.contour(gamma_grid, beta_grid, stability_grid, 
                  levels=[0.3, 0.7], colors=['red', 'yellow'], 
                  linewidths=2, alpha=0.8)
        
        # 添加颜色条
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Survival Rate')
        
        ax.set_xlabel('γ (Maintenance Cost Exponent)', fontsize=12)
        ax.set_ylabel('β (Environmental Sensitivity)', fontsize=12)
        ax.set_title('System Stability Regions', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加区域标签
        ax.text(0.05, 0.95, 'Stable\nRegion', transform=ax.transAxes,
               fontsize=12, fontweight='bold', color='darkgreen',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax.text(0.05, 0.05, 'Unstable\nRegion', transform=ax.transAxes,
               fontsize=12, fontweight='bold', color='darkred',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

def main():
    """主函数"""
    # 加载数据
    import glob
    result_files = glob.glob("ultra_fine_scan_results/ultra_fine_results_*.csv")
    
    if not result_files:
        print("未找到超细扫描结果文件")
        return
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"加载数据: {latest_file}")
    
    results_df = pd.read_csv(latest_file)
    
    # 创建分析器
    analyzer = ParameterSensitivityAnalyzer(results_df)
    
    # 运行所有分析
    print("开始参数敏感性分析...")
    
    # 1. 计算敏感性指标
    print("1. 计算敏感性指标...")
    sensitivity_data = analyzer.calculate_sensitivity_indices()
    
    # 2. 识别相变
    print("2. 识别相变点...")
    phase_data = analyzer.identify_phase_transitions()
    
    # 3. 计算临界指数
    print("3. 计算临界指数...")
    exponent_data = analyzer.calculate_critical_exponents()
    
    # 4. 创建可视化
    print("4. 创建敏感性热力图...")
    analyzer.create_sensitivity_heatmaps()
    
    print("5. 创建相变图...")
    analyzer.create_phase_transition_plots()
    
    print("6. 创建临界指数图...")
    analyzer.create_critical_exponent_plots()
    
    print("7. 创建综合报告...")
    analyzer.create_comprehensive_sensitivity_report()
    
    print("参数敏感性分析完成！")
    
    # 保存分析结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存敏感性数据
    with open(f"sensitivity_analysis_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump({
            'sensitivity_metrics': sensitivity_data,
            'phase_transitions': phase_data,
            'critical_exponents': exponent_data
        }, f, indent=2, ensure_ascii=False)
    
    print(f"分析结果已保存: sensitivity_analysis_{timestamp}.json")

if __name__ == "__main__":
    main()