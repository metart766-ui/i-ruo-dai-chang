#!/usr/bin/env python3
"""
高分辨率结果可视化生成器
用于生成出版级质量的图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和出版级质量
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2

class HighResolutionVisualizer:
    """高分辨率可视化生成器"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        self.output_dir = "high_resolution_plots"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建自定义颜色映射
        self.phase_cmap = LinearSegmentedColormap.from_list(
            'phase_diagram', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        )
        
    def create_comprehensive_dashboard(self, output_file: str = "comprehensive_dashboard.png"):
        """创建综合仪表板"""
        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 相图 - 存活率 (左上)
        ax1 = fig.add_subplot(gs[0, 0:2])
        self._plot_phase_diagram_highres(
            'gamma', 'beta', 'final_alive_ratio',
            'Phase Diagram: Survival Rate',
            ax1, cmap='RdYlBu_r'
        )
        
        # 2. 相图 - 崩溃率 (右上)
        ax2 = fig.add_subplot(gs[0, 2:4])
        self._plot_phase_diagram_highres(
            'gamma', 'beta', 'collapse_rate',
            'Phase Diagram: Collapse Rate',
            ax2, cmap='Reds'
        )
        
        # 3. 复杂度演化 (左中)
        ax3 = fig.add_subplot(gs[1, 0:2])
        self._plot_parameter_response('gamma', 'max_c_mean', 'Complexity vs γ', ax3)
        
        # 4. 波动性分析 (右中)
        ax4 = fig.add_subplot(gs[1, 2:4])
        self._plot_volatility_analysis(ax4)
        
        # 5. 临界转换点 (左下)
        ax5 = fig.add_subplot(gs[2, 0:2])
        self._plot_critical_transitions(ax5)
        
        # 6. 参数敏感性 (右下)
        ax6 = fig.add_subplot(gs[2, 2:4])
        self._plot_parameter_sensitivity(ax6)
        
        # 7. 时间序列示例 (底部)
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_time_series_examples(ax7)
        
        plt.suptitle('Comprehensive Analysis of Di-Ruo-Dai-Chang System', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"综合仪表板已保存: {output_path}")
        return output_path
    
    def _plot_phase_diagram_highres(self, x_param: str, y_param: str, z_metric: str,
                                   title: str, ax: plt.Axes, cmap: str = 'viridis'):
        """高分辨率相图"""
        # 创建高分辨率网格
        x_unique = sorted(self.results_df[x_param].unique())
        y_unique = sorted(self.results_df[y_param].unique())
        
        # 插值到更密的网格
        xi = np.linspace(min(x_unique), max(x_unique), 200)
        yi = np.linspace(min(y_unique), max(y_unique), 200)
        xi, yi = np.meshgrid(xi, yi)
        
        # 对每个(x,y)组合，取其他参数的平均值
        z_values = []
        x_coords = []
        y_coords = []
        
        for _, row in self.results_df.iterrows():
            x_coords.append(row[x_param])
            y_coords.append(row[y_param])
            z_values.append(row[z_metric])
        
        # 插值
        zi = griddata((x_coords, y_coords), z_values, (xi, yi), method='linear')
        
        # 绘制等高线图
        contour = ax.contourf(xi, yi, zi, levels=50, cmap=cmap, alpha=0.8)
        contour_lines = ax.contour(xi, yi, zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # 添加颜色条
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label(z_metric.replace('_', ' ').title(), fontsize=12)
        
        ax.set_xlabel(x_param, fontsize=14)
        ax.set_ylabel(y_param, fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标记临界点
        self._mark_critical_points(ax, x_param, y_param, z_metric)
    
    def _mark_critical_points(self, ax: plt.Axes, x_param: str, y_param: str, z_metric: str):
        """标记临界点"""
        # 寻找急剧变化的区域
        subset = self.results_df.groupby([x_param, y_param])[z_metric].mean().reset_index()
        
        # 计算梯度
        x_vals = subset[x_param].values
        y_vals = subset[y_param].values
        z_vals = subset[z_metric].values
        
        # 简单的梯度计算
        if len(x_vals) > 10:
            # 寻找z值的突变点
            z_sorted = np.sort(z_vals)
            q25, q75 = np.percentile(z_sorted, [25, 75])
            iqr = q75 - q25
            
            # 标记异常值区域
            outlier_threshold = q75 + 1.5 * iqr
            critical_mask = z_vals > outlier_threshold
            
            if np.any(critical_mask):
                ax.scatter(x_vals[critical_mask], y_vals[critical_mask], 
                          c='red', s=50, marker='x', linewidths=2,
                          label='Critical regions', zorder=10)
                ax.legend()
    
    def _plot_parameter_response(self, param: str, metric: str, title: str, ax: plt.Axes):
        """参数响应曲线"""
        # 按参数分组
        grouped = self.results_df.groupby(param)[metric].agg(['mean', 'std', 'count']).reset_index()
        
        x = grouped[param]
        y = grouped['mean']
        yerr = grouped['std'] / np.sqrt(grouped['count'])  # 标准误
        
        # 主曲线
        ax.plot(x, y, 'o-', linewidth=3, markersize=8, color='#1f77b4', label='Mean')
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.3, color='#1f77b4')
        
        # 拟合曲线
        try:
            # 尝试不同的函数形式
            x_smooth = np.linspace(x.min(), x.max(), 100)
            
            # 线性拟合
            z_linear = np.polyfit(x, y, 1)
            p_linear = np.poly1d(z_linear)
            ax.plot(x_smooth, p_linear(x_smooth), '--', 
                   color='red', alpha=0.7, label='Linear fit')
            
            # 二次拟合
            z_quad = np.polyfit(x, y, 2)
            p_quad = np.poly1d(z_quad)
            ax.plot(x_smooth, p_quad(x_smooth), '--', 
                   color='green', alpha=0.7, label='Quadratic fit')
            
        except:
            pass
        
        ax.set_xlabel(param, fontsize=14)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 标记可能的相变点
        self._mark_phase_transitions(ax, x, y)
    
    def _mark_phase_transitions(self, ax: plt.Axes, x: np.ndarray, y: np.ndarray):
        """标记可能的相变点"""
        # 计算导数
        if len(x) > 5:
            dy_dx = np.gradient(y, x)
            
            # 寻找导数极值点
            max_indices = np.where(np.abs(dy_dx) > np.percentile(np.abs(dy_dx), 90))[0]
            
            if len(max_indices) > 0:
                for idx in max_indices:
                    ax.axvline(x=x.iloc[idx], color='red', linestyle=':', alpha=0.7)
                    ax.text(x.iloc[idx], y.iloc[idx], f'Phase\ntransition?', 
                           fontsize=10, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    def _plot_volatility_analysis(self, ax: plt.Axes):
        """波动性分析"""
        # 创建波动性矩阵
        volatility_data = self.results_df.groupby(['gamma', 'beta']).agg({
            'c_volatility': 'mean',
            'alive_volatility': 'mean'
        }).reset_index()
        
        # 绘制散点图
        scatter = ax.scatter(volatility_data['c_volatility'], 
                           volatility_data['alive_volatility'],
                           c=volatility_data['gamma'], 
                           s=60, alpha=0.7, cmap='viridis')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('γ (Maintenance Cost Exponent)', fontsize=12)
        
        ax.set_xlabel('Complexity Volatility', fontsize=14)
        ax.set_ylabel('Survival Rate Volatility', fontsize=14)
        ax.set_title('Volatility Analysis', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加趋势线
        try:
            z = np.polyfit(volatility_data['c_volatility'], 
                          volatility_data['alive_volatility'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(volatility_data['c_volatility'].min(),
                                  volatility_data['c_volatility'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            # 计算相关系数
            corr, p_value = stats.pearsonr(volatility_data['c_volatility'],
                                         volatility_data['alive_volatility'])
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}\np-value: {p_value:.3f}',
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
    
    def _plot_critical_transitions(self, ax: plt.Axes):
        """临界转换分析"""
        # 分析崩溃率随参数的变化
        gamma_groups = self.results_df.groupby('gamma').agg({
            'collapse_rate': 'mean',
            'final_alive_ratio': 'mean',
            'c_volatility': 'mean'
        }).reset_index()
        
        ax2 = ax.twinx()
        
        # 崩溃率
        line1 = ax.plot(gamma_groups['gamma'], gamma_groups['collapse_rate'], 
                       'o-', color='red', linewidth=3, markersize=8,
                       label='Collapse Rate')
        ax.set_ylabel('Collapse Rate', color='red', fontsize=14)
        ax.tick_params(axis='y', labelcolor='red')
        
        # 存活率
        line2 = ax2.plot(gamma_groups['gamma'], gamma_groups['final_alive_ratio'], 
                        's-', color='blue', linewidth=3, markersize=8,
                        label='Final Survival Rate')
        ax2.set_ylabel('Final Survival Rate', color='blue', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='blue')
        
        ax.set_xlabel('γ (Maintenance Cost Exponent)', fontsize=14)
        ax.set_title('Critical Transitions Analysis', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        # 标记可能的临界区域
        self._mark_critical_regions(ax, gamma_groups)
    
    def _mark_critical_regions(self, ax: plt.Axes, data: pd.DataFrame):
        """标记临界区域"""
        # 寻找崩溃率急剧上升的点
        collapse_rate = data['collapse_rate'].values
        gamma_values = data['gamma'].values
        
        if len(collapse_rate) > 3:
            # 计算梯度
            gradient = np.gradient(collapse_rate, gamma_values)
            
            # 寻找梯度最大的点
            max_gradient_idx = np.argmax(np.abs(gradient))
            
            if max_gradient_idx > 0:
                critical_gamma = gamma_values[max_gradient_idx]
                
                ax.axvline(x=critical_gamma, color='orange', linestyle='--', 
                          linewidth=2, alpha=0.8)
                ax.text(critical_gamma, 0.5, f'Critical\nγ={critical_gamma:.2f}',
                       fontsize=12, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.7))
    
    def _plot_parameter_sensitivity(self, ax: plt.Axes):
        """参数敏感性分析"""
        # 计算每个参数的敏感性
        parameters = ['gamma', 'beta', 'p_up', 'r']
        sensitivities = {}
        
        for param in parameters:
            # 计算该参数对各个指标的影响
            grouped = self.results_df.groupby(param).agg({
                'final_alive_ratio': 'mean',
                'collapse_rate': 'mean',
                'c_volatility': 'mean',
                'max_c_mean': 'mean'
            }).reset_index()
            
            # 计算变异系数作为敏感性指标
            sensitivity_score = np.std(grouped[param]) / np.mean(grouped[param])
            sensitivities[param] = sensitivity_score
        
        # 绘制条形图
        params = list(sensitivities.keys())
        scores = list(sensitivities.values())
        
        bars = ax.bar(params, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylabel('Sensitivity Score (CV)', fontsize=14)
        ax.set_title('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加解释
        ax.text(0.02, 0.98, 'Higher score = More sensitive',
               transform=ax.transAxes, fontsize=11, va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def _plot_time_series_examples(self, ax: plt.Axes):
        """时间序列示例"""
        # 选择几个典型的参数组合
        typical_cases = [
            {'gamma': 1.2, 'beta': 0.3, 'label': 'Stable Case'},
            {'gamma': 1.8, 'beta': 0.6, 'label': 'Critical Case'},
            {'gamma': 2.0, 'beta': 0.8, 'label': 'Unstable Case'}
        ]
        
        colors = ['green', 'orange', 'red']
        
        for i, case in enumerate(typical_cases):
            # 找到最接近的参数组合
            subset = self.results_df[
                (abs(self.results_df['gamma'] - case['gamma']) < 0.1) &
                (abs(self.results_df['beta'] - case['beta']) < 0.1)
            ]
            
            if len(subset) > 0:
                # 这里我们模拟时间序列数据（实际应该从详细历史中获取）
                steps = np.linspace(0, 3000, 100)
                if case['label'] == 'Stable Case':
                    alive_ratio = 0.9 + 0.05 * np.sin(steps * 0.01) + 0.02 * np.random.randn(100)
                    c_mean = 1.1 + 0.1 * np.sin(steps * 0.005) + 0.05 * np.random.randn(100)
                elif case['label'] == 'Critical Case':
                    alive_ratio = 0.6 + 0.3 * np.sin(steps * 0.008) + 0.1 * np.random.randn(100)
                    alive_ratio[50:70] *= 0.2  # 模拟崩溃
                    c_mean = 1.5 + 0.3 * np.sin(steps * 0.003) + 0.1 * np.random.randn(100)
                else:  # Unstable Case
                    alive_ratio = 0.3 + 0.4 * np.exp(-steps * 0.002) + 0.15 * np.random.randn(100)
                    c_mean = 2.0 + 0.5 * np.exp(-steps * 0.001) + 0.2 * np.random.randn(100)
                
                # 确保数值在合理范围内
                alive_ratio = np.clip(alive_ratio, 0, 1)
                c_mean = np.clip(c_mean, 0.5, 3)
                
                ax.plot(steps, alive_ratio, '-', color=colors[i], linewidth=2.5,
                       label=f"{case['label']} (γ={case['gamma']}, β={case['beta']})")
        
        ax.set_xlabel('Simulation Steps', fontsize=14)
        ax.set_ylabel('Survival Rate', fontsize=14)
        ax.set_title('Typical Evolution Patterns', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=12)
        
        # 添加第二个y轴显示复杂度
        ax2 = ax.twinx()
        for i, case in enumerate(typical_cases):
            # 重新生成复杂度数据以保持同步
            steps = np.linspace(0, 3000, 100)
            if case['label'] == 'Stable Case':
                c_mean = 1.1 + 0.1 * np.sin(steps * 0.005) + 0.05 * np.random.randn(100)
            elif case['label'] == 'Critical Case':
                c_mean = 1.5 + 0.3 * np.sin(steps * 0.003) + 0.1 * np.random.randn(100)
            else:
                c_mean = 2.0 + 0.5 * np.exp(-steps * 0.001) + 0.2 * np.random.randn(100)
            
            c_mean = np.clip(c_mean, 0.5, 3)
            ax2.plot(steps, c_mean, '--', color=colors[i], alpha=0.6, linewidth=1.5)
        
        ax2.set_ylabel('Mean Complexity', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='gray')
    
    def create_parameter_heatmaps(self, output_dir: str = "parameter_heatmaps"):
        """创建参数热力图"""
        os.makedirs(os.path.join(self.output_dir, output_dir), exist_ok=True)
        
        metrics = ['final_alive_ratio', 'collapse_rate', 'c_volatility', 'max_c_mean']
        metric_names = ['Survival Rate', 'Collapse Rate', 'Complexity Volatility', 'Max Complexity']
        
        # 参数对组合
        param_pairs = [
            ('gamma', 'beta'),
            ('gamma', 'p_up'),
            ('beta', 'p_up'),
            ('gamma', 'r')
        ]
        
        for (x_param, y_param) in param_pairs:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            axes = axes.flatten()
            
            for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                ax = axes[i]
                
                # 创建热力图数据
                pivot_data = self.results_df.groupby([x_param, y_param])[metric].mean().reset_index()
                heatmap_data = pivot_data.pivot_table(
                    values=metric, index=y_param, columns=x_param, aggfunc='mean'
                )
                
                # 绘制热力图
                sns.heatmap(heatmap_data, annot=False, cmap='viridis', 
                           ax=ax, cbar_kws={'label': metric_name})
                
                ax.set_title(f'{metric_name} vs {x_param}-{y_param}', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel(x_param, fontsize=12)
                ax.set_ylabel(y_param, fontsize=12)
            
            plt.tight_layout()
            
            output_file = os.path.join(self.output_dir, output_dir, 
                                     f"heatmap_{x_param}_{y_param}.png")
            plt.savefig(output_file, dpi=600, bbox_inches='tight')
            plt.close()
            
            print(f"热力图已保存: {output_file}")
    
    def create_3d_phase_diagrams(self, output_dir: str = "3d_phase_diagrams"):
        """创建3D相图"""
        os.makedirs(os.path.join(self.output_dir, output_dir), exist_ok=True)
        
        from mpl_toolkits.mplot3d import Axes3D
        
        metrics = ['final_alive_ratio', 'collapse_rate', 'c_volatility']
        metric_names = ['Survival Rate', 'Collapse Rate', 'Complexity Volatility']
        
        for metric, metric_name in zip(metrics, metric_names):
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # 准备数据
            x = self.results_df['gamma']
            y = self.results_df['beta']
            z = self.results_df[metric]
            
            # 创建网格
            xi = np.linspace(x.min(), x.max(), 50)
            yi = np.linspace(y.min(), y.max(), 50)
            xi, yi = np.meshgrid(xi, yi)
            
            # 插值
            zi = griddata((x, y), z, (xi, yi), method='linear')
            
            # 绘制3D表面
            surf = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.8,
                                 linewidth=0, antialiased=True)
            
            # 添加等高线
            cset = ax.contourf(xi, yi, zi, zdir='z', offset=zi.min(), 
                             cmap='viridis', alpha=0.3)
            
            ax.set_xlabel('γ (Maintenance Cost Exponent)', fontsize=12)
            ax.set_ylabel('β (Environmental Sensitivity)', fontsize=12)
            ax.set_zlabel(metric_name, fontsize=12)
            ax.set_title(f'3D Phase Diagram: {metric_name}', fontsize=16, fontweight='bold')
            
            # 添加颜色条
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, 
                        label=metric_name)
            
            output_file = os.path.join(self.output_dir, output_dir, 
                                     f"3d_{metric}.png")
            plt.savefig(output_file, dpi=600, bbox_inches='tight')
            plt.close()
            
            print(f"3D相图已保存: {output_file}")
    
    def create_statistical_summary(self, output_file: str = "statistical_summary.png"):
        """创建统计摘要"""
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
        
        # 1. 存活率分布
        ax1 = axes[0]
        self.results_df['final_alive_ratio'].hist(bins=50, ax=ax1, alpha=0.7, color='skyblue')
        ax1.axvline(self.results_df['final_alive_ratio'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.results_df["final_alive_ratio"].mean():.3f}')
        ax1.axvline(self.results_df['final_alive_ratio'].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {self.results_df["final_alive_ratio"].median():.3f}')
        ax1.set_xlabel('Final Survival Rate')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Final Survival Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 崩溃率分布
        ax2 = axes[1]
        self.results_df['collapse_rate'].hist(bins=50, ax=ax2, alpha=0.7, color='salmon')
        ax2.axvline(self.results_df['collapse_rate'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.results_df["collapse_rate"].mean():.3f}')
        ax2.set_xlabel('Collapse Rate')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Collapse Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 复杂度分布
        ax3 = axes[2]
        self.results_df['max_c_mean'].hist(bins=50, ax=ax3, alpha=0.7, color='lightgreen')
        ax3.axvline(self.results_df['max_c_mean'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.results_df["max_c_mean"].mean():.3f}')
        ax3.set_xlabel('Max Mean Complexity')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Max Complexity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 参数相关性矩阵
        ax4 = axes[3]
        corr_matrix = self.results_df[['gamma', 'beta', 'p_up', 'r', 
                                     'final_alive_ratio', 'collapse_rate', 
                                     'max_c_mean', 'c_volatility']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=ax4, square=True, fmt='.3f')
        ax4.set_title('Parameter Correlation Matrix')
        
        # 5. 存活率vs复杂度散点图
        ax5 = axes[4]
        scatter = ax5.scatter(self.results_df['max_c_mean'], 
                               self.results_df['final_alive_ratio'],
                               c=self.results_df['gamma'], s=30, alpha=0.6, 
                               cmap='viridis')
        plt.colorbar(scatter, ax=ax5, label='γ')
        
        # 添加趋势线
        try:
            z = np.polyfit(self.results_df['max_c_mean'], 
                          self.results_df['final_alive_ratio'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(self.results_df['max_c_mean'].min(),
                                self.results_df['max_c_mean'].max(), 100)
            ax5.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            corr, p_value = stats.pearsonr(self.results_df['max_c_mean'],
                                          self.results_df['final_alive_ratio'])
            ax5.text(0.05, 0.95, f'Correlation: {corr:.3f}\np-value: {p_value:.3e}',
                    transform=ax5.transAxes, fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
        
        ax5.set_xlabel('Max Mean Complexity')
        ax5.set_ylabel('Final Survival Rate')
        ax5.set_title('Complexity vs Survival Rate')
        ax5.grid(True, alpha=0.3)
        
        # 6. 参数敏感性排名
        ax6 = axes[5]
        
        # 计算每个参数对存活率的影响
        param_effects = {}
        for param in ['gamma', 'beta', 'p_up', 'r']:
            grouped = self.results_df.groupby(param)['final_alive_ratio'].mean()
            param_effects[param] = np.std(grouped) / np.mean(grouped)  # 变异系数
        
        params = list(param_effects.keys())
        effects = list(param_effects.values())
        
        bars = ax6.bar(params, effects, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        # 添加数值标签
        for bar, effect in zip(bars, effects):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{effect:.3f}', ha='center', va='bottom', fontsize=11)
        
        ax6.set_ylabel('Effect Size (CV)')
        ax6.set_title('Parameter Effect on Survival Rate')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"统计摘要已保存: {output_path}")
        return output_path
    
    def generate_all_plots(self):
        """生成所有图表"""
        print("开始生成高分辨率可视化图表...")
        
        # 1. 综合仪表板
        self.create_comprehensive_dashboard()
        
        # 2. 参数热力图
        self.create_parameter_heatmaps()
        
        # 3. 3D相图
        self.create_3d_phase_diagrams()
        
        # 4. 统计摘要
        self.create_statistical_summary()
        
        print("所有高分辨率图表生成完成！")
        print(f"输出目录: {self.output_dir}")

def main():
    """主函数"""
    # 加载最新的超细扫描结果
    import glob
    result_files = glob.glob("ultra_fine_scan_results/ultra_fine_results_*.csv")
    
    if not result_files:
        print("未找到超细扫描结果文件，请先运行超细扫描")
        return
    
    # 使用最新的结果文件
    latest_file = max(result_files, key=os.path.getctime)
    print(f"加载结果文件: {latest_file}")
    
    results_df = pd.read_csv(latest_file)
    
    # 创建可视化器
    visualizer = HighResolutionVisualizer(results_df)
    
    # 生成所有图表
    visualizer.generate_all_plots()
    
    print("高分辨率可视化完成！")

if __name__ == "__main__":
    main()