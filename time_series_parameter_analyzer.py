#!/usr/bin/env python3
"""
实时参数敏感性分析器 - 针对时间序列数据优化版本
专门处理siyan_results.csv这种时间序列数据
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

class TimeSeriesParameterAnalyzer:
    """时间序列参数分析器"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        self.time_series_metrics = {}
        self.phase_analysis = {}
        
    def analyze_time_series(self) -> Dict:
        """分析时间序列特征"""
        print("开始时间序列分析...")
        
        # 基本统计
        metrics = ['alive_ratio', 'c_mean', 'p_mean_serial', 'p_mean_env', 'pc_serial', 'pc_env']
        
        time_series_stats = {}
        
        for metric in metrics:
            values = self.results_df[metric].values
            
            # 基本统计
            stats_dict = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'final_value': values[-1],
                'initial_value': values[0],
                'trend': self._calculate_trend(values),
                'volatility': self._calculate_volatility(values),
                'stability': self._calculate_stability(values)
            }
            
            time_series_stats[metric] = stats_dict
        
        self.time_series_metrics = time_series_stats
        return time_series_stats
    
    def _calculate_trend(self, values: np.ndarray) -> Dict:
        """计算趋势"""
        steps = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(steps, values)
        
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        }
    
    def _calculate_volatility(self, values: np.ndarray) -> float:
        """计算波动性"""
        # 计算滚动标准差
        if len(values) >= 10:
            rolling_std = pd.Series(values).rolling(window=10).std().dropna()
            return np.mean(rolling_std)
        else:
            return np.std(values)
    
    def _calculate_stability(self, values: np.ndarray) -> float:
        """计算稳定性"""
        # 计算变异系数
        mean_val = np.mean(values)
        if mean_val != 0:
            return 1.0 / (1.0 + np.std(values) / abs(mean_val))  # 稳定性指数
        else:
            return 0.0
    
    def detect_critical_transitions(self) -> Dict:
        """检测临界转变"""
        print("开始检测临界转变...")
        
        transitions = {}
        
        # 分析存活率的临界转变
        alive_ratio = self.results_df['alive_ratio'].values
        steps = self.results_df['step'].values
        
        # 方法1: 检测急剧变化点
        transitions['sharp_changes'] = self._detect_sharp_changes(alive_ratio, steps)
        
        # 方法2: 检测趋势变化点
        transitions['trend_changes'] = self._detect_trend_changes(alive_ratio, steps)
        
        # 方法3: 检测方差变化点
        transitions['variance_changes'] = self._detect_variance_changes(alive_ratio, steps)
        
        self.phase_analysis = transitions
        return transitions
    
    def _detect_sharp_changes(self, values: np.ndarray, steps: np.ndarray) -> List[Dict]:
        """检测急剧变化点"""
        changes = []
        
        # 计算一阶差分
        diffs = np.diff(values)
        
        # 寻找异常大的变化
        threshold = np.percentile(np.abs(diffs), 95)
        
        for i, diff in enumerate(diffs):
            if abs(diff) > threshold:
                changes.append({
                    'step': steps[i+1],
                    'change_magnitude': diff,
                    'from_value': values[i],
                    'to_value': values[i+1],
                    'type': 'sharp_increase' if diff > 0 else 'sharp_decrease'
                })
        
        return changes
    
    def _detect_trend_changes(self, values: np.ndarray, steps: np.ndarray) -> List[Dict]:
        """检测趋势变化点"""
        changes = []
        
        # 使用滑动窗口分析趋势
        window_size = min(50, len(values) // 4)
        
        if window_size >= 10:
            previous_trend = None
            
            for i in range(window_size, len(values), window_size // 2):
                start_idx = max(0, i - window_size)
                end_idx = i
                
                window_values = values[start_idx:end_idx]
                window_steps = steps[start_idx:end_idx]
                
                # 计算窗口内的趋势
                slope, _, _, p_value, _ = stats.linregress(window_steps, window_values)
                
                current_trend = 'increasing' if slope > 0.001 else 'decreasing' if slope < -0.001 else 'stable'
                
                if previous_trend and previous_trend != current_trend:
                    changes.append({
                        'step': steps[end_idx-1],
                        'from_trend': previous_trend,
                        'to_trend': current_trend,
                        'slope': slope,
                        'p_value': p_value
                    })
                
                previous_trend = current_trend
        
        return changes
    
    def _detect_variance_changes(self, values: np.ndarray, steps: np.ndarray) -> List[Dict]:
        """检测方差变化点"""
        changes = []
        
        # 使用滑动窗口分析方差
        window_size = min(30, len(values) // 3)
        
        if window_size >= 10:
            previous_variance = None
            
            for i in range(window_size, len(values), window_size // 2):
                start_idx = max(0, i - window_size)
                end_idx = i
                
                window_values = values[start_idx:end_idx]
                current_variance = np.var(window_values)
                
                if previous_variance:
                    variance_ratio = current_variance / previous_variance
                    if variance_ratio > 2.0 or variance_ratio < 0.5:
                        changes.append({
                            'step': steps[end_idx-1],
                            'from_variance': previous_variance,
                            'to_variance': current_variance,
                            'variance_ratio': variance_ratio,
                            'type': 'variance_increase' if variance_ratio > 2.0 else 'variance_decrease'
                        })
                
                previous_variance = current_variance
        
        return changes
    
    def create_time_series_plots(self, output_dir: str = "time_series_analysis"):
        """创建时间序列图"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 基本时间序列图
        plt.figure(figsize=(15, 12))
        
        metrics = ['alive_ratio', 'c_mean', 'p_mean_serial', 'p_mean_env', 'pc_serial', 'pc_env']
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            plt.subplot(3, 2, i+1)
            
            steps = self.results_df['step'].values
            values = self.results_df[metric].values
            
            plt.plot(steps, values, color=color, linewidth=2, alpha=0.8)
            
            # 标记临界转变点
            if self.phase_analysis:
                for change in self.phase_analysis.get('sharp_changes', []):
                    if change['step'] in steps:
                        idx = np.where(steps == change['step'])[0]
                        if len(idx) > 0:
                            plt.axvline(x=change['step'], color='red', linestyle='--', alpha=0.7)
                
                for change in self.phase_analysis.get('trend_changes', []):
                    if change['step'] in steps:
                        plt.axvline(x=change['step'], color='orange', linestyle=':', alpha=0.7)
            
            plt.xlabel('Step', fontsize=12)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.title(f'{metric.replace("_", " ").title()} vs Time', fontsize=14)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 'time_series_overview.png')
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"时间序列图已保存: {output_file}")
        return output_file
    
    def create_correlation_analysis(self, output_dir: str = "correlation_analysis"):
        """创建相关性分析图"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算相关性矩阵
        metrics = ['alive_ratio', 'c_mean', 'p_mean_serial', 'p_mean_env', 'pc_serial', 'pc_env']
        
        # 计算皮尔逊相关系数
        correlation_matrix = self.results_df[metrics].corr()
        
        plt.figure(figsize=(12, 10))
        
        # 绘制热力图
        sns.heatmap(correlation_matrix, 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=[m.replace('_', ' ').title() for m in metrics],
                   annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   square=True, cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Parameter Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"相关性矩阵图已保存: {output_file}")
        
        # 创建散点图矩阵
        plt.figure(figsize=(15, 15))
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i != j:
                    plt.subplot(len(metrics), len(metrics), i*len(metrics)+j+1)
                    
                    x = self.results_df[metric1].values
                    y = self.results_df[metric2].values
                    
                    plt.scatter(x, y, alpha=0.6, s=20)
                    
                    # 添加趋势线
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
                    
                    # 计算相关系数
                    corr, _ = stats.pearsonr(x, y)
                    plt.text(0.05, 0.95, f'r={corr:.3f}', transform=plt.gca().transAxes,
                            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    plt.xlabel(metric1.replace('_', ' ').title(), fontsize=10)
                    plt.ylabel(metric2.replace('_', ' ').title(), fontsize=10)
                    plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 'scatter_matrix.png')
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"散点图矩阵已保存: {output_file}")
        return output_file
    
    def create_phase_transition_analysis(self, output_dir: str = "phase_transition_analysis"):
        """创建相变分析图"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.phase_analysis:
            self.detect_critical_transitions()
        
        # 存活率的详细分析
        steps = self.results_df['step'].values
        alive_ratio = self.results_df['alive_ratio'].values
        
        plt.figure(figsize=(16, 12))
        
        # 主图：存活率时间序列
        plt.subplot(3, 1, 1)
        plt.plot(steps, alive_ratio, 'b-', linewidth=2, alpha=0.8, label='Alive Ratio')
        
        # 标记临界转变点
        colors = {'sharp_increase': 'green', 'sharp_decrease': 'red',
                 'variance_increase': 'orange', 'variance_decrease': 'purple'}
        
        for change_type, changes in self.phase_analysis.items():
            for change in changes:
                if 'step' in change:
                    color = colors.get(change.get('type', ''), 'black')
                    plt.axvline(x=change['step'], color=color, linestyle='--', 
                               alpha=0.7, linewidth=2)
        
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Alive Ratio', fontsize=12)
        plt.title('Critical Transitions in System Survival', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 子图1: 一阶差分
        plt.subplot(3, 1, 2)
        diffs = np.diff(alive_ratio)
        plt.plot(steps[1:], diffs, 'r-', linewidth=1.5, alpha=0.8)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 标记急剧变化
        threshold = np.percentile(np.abs(diffs), 95)
        plt.axhline(y=threshold, color='red', linestyle=':', alpha=0.7)
        plt.axhline(y=-threshold, color='red', linestyle=':', alpha=0.7)
        
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('First Difference', fontsize=12)
        plt.title('First Order Differences (Change Detection)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 子图2: 滚动方差
        plt.subplot(3, 1, 3)
        window_size = min(30, len(alive_ratio) // 3)
        if window_size >= 10:
            rolling_var = pd.Series(alive_ratio).rolling(window=window_size).var().dropna()
            plt.plot(steps[window_size-1:], rolling_var.values, 'purple', 
                    linewidth=2, alpha=0.8)
            
            plt.xlabel('Step', fontsize=12)
            plt.ylabel('Rolling Variance', fontsize=12)
            plt.title(f'Rolling Variance (Window Size: {window_size})', fontsize=14)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 'phase_transition_analysis.png')
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"相变分析图已保存: {output_file}")
        return output_file
    
    def print_analysis_summary(self):
        """打印分析摘要"""
        print("\n" + "="*80)
        print("时间序列参数敏感性分析摘要")
        print("="*80)
        
        if self.time_series_metrics:
            print("\n1. 时间序列统计特征:")
            for metric, stats in self.time_series_metrics.items():
                print(f"\n   {metric}:")
                print(f"     平均值: {stats['mean']:.4f}")
                print(f"     标准差: {stats['std']:.4f}")
                print(f"     最终值: {stats['final_value']:.4f}")
                print(f"     趋势: {stats['trend']['trend_direction']} "
                      f"(斜率={stats['trend']['slope']:.6f}, R²={stats['trend']['r_squared']:.3f})")
                print(f"     稳定性: {stats['stability']:.4f}")
        
        if self.phase_analysis:
            print("\n2. 临界转变检测结果:")
            for change_type, changes in self.phase_analysis.items():
                if changes:
                    print(f"\n   {change_type}:")
                    for change in changes[:3]:  # 只显示前3个
                        if 'step' in change:
                            print(f"     步骤{change['step']}: {change.get('type', 'unknown')}")
                            if 'change_magnitude' in change:
                                print(f"       变化幅度: {change['change_magnitude']:.4f}")
                            if 'variance_ratio' in change:
                                print(f"       方差比: {change['variance_ratio']:.2f}")
        
        print("\n" + "="*80)

def main():
    """主函数"""
    # 加载数据
    import glob
    result_files = glob.glob("siyan_results.csv")
    
    if not result_files:
        print("未找到结果文件")
        return
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"加载数据: {latest_file}")
    
    results_df = pd.read_csv(latest_file)
    print(f"数据形状: {results_df.shape}")
    print(f"可用列: {results_df.columns.tolist()}")
    
    # 创建分析器
    analyzer = TimeSeriesParameterAnalyzer(results_df)
    
    # 运行分析
    print("开始时间序列参数敏感性分析...")
    
    # 1. 时间序列分析
    print("\n1. 分析时间序列特征...")
    time_series_data = analyzer.analyze_time_series()
    
    # 2. 检测临界转变
    print("\n2. 检测临界转变...")
    transition_data = analyzer.detect_critical_transitions()
    
    # 3. 创建可视化
    print("\n3. 创建可视化...")
    
    print("创建时间序列图...")
    analyzer.create_time_series_plots()
    
    print("创建相关性分析图...")
    analyzer.create_correlation_analysis()
    
    print("创建相变分析图...")
    analyzer.create_phase_transition_analysis()
    
    # 4. 打印摘要
    analyzer.print_analysis_summary()
    
    print("\n时间序列参数敏感性分析完成！")

if __name__ == "__main__":
    main()