#!/usr/bin/env python3
"""
参数空间临界条件与相变行为分析
专门分析递弱代偿系统中的临界现象和相变点
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CriticalConditionAnalyzer:
    """临界条件分析器"""
    
    def __init__(self, results_data: pd.DataFrame):
        """
        初始化分析器
        
        Args:
            results_data: 实验结果数据，包含参数和结果
        """
        self.results_data = results_data
        self.critical_points = {}
        self.phase_diagrams = {}
        
    def identify_critical_points(self, parameter: str, response_var: str, 
                               threshold_method: str = 'derivative') -> Dict:
        """
        识别临界点
        
        Args:
            parameter: 参数名称
            response_var: 响应变量名称
            threshold_method: 阈值检测方法 ('derivative', 'variance', 'correlation')
        
        Returns:
            临界点信息
        """
        # 获取参数值和对应的响应
        param_values = sorted(self.results_data[parameter].unique())
        responses = []
        
        for value in param_values:
            subset = self.results_data[self.results_data[parameter] == value]
            if not subset.empty:
                responses.append(subset[response_var].mean())
            else:
                responses.append(0)
        
        responses = np.array(responses)
        param_values = np.array(param_values)
        
        critical_info = {
            'parameter': parameter,
            'response_variable': response_var,
            'method': threshold_method,
            'critical_points': [],
            'transition_zones': []
        }
        
        if threshold_method == 'derivative':
            # 基于导数变化检测临界点
            derivatives = np.gradient(responses, param_values)
            second_derivatives = np.gradient(derivatives, param_values)
            
            # 寻找导数变化最大的点
            derivative_changes = np.abs(derivatives)
            critical_indices = np.where(derivative_changes > np.percentile(derivative_changes, 90))[0]
            
            for idx in critical_indices:
                critical_info['critical_points'].append({
                    'param_value': param_values[idx],
                    'response_value': responses[idx],
                    'derivative': derivatives[idx],
                    'second_derivative': second_derivatives[idx],
                    'significance': derivative_changes[idx] / np.max(derivative_changes)
                })
        
        elif threshold_method == 'variance':
            # 基于方差变化检测临界点
            window_size = max(3, len(param_values) // 10)
            variances = []
            
            for i in range(len(responses)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(responses), i + window_size // 2)
                window_responses = responses[start_idx:end_idx]
                variances.append(np.var(window_responses))
            
            variances = np.array(variances)
            variance_changes = np.gradient(variances, param_values)
            
            # 寻找方差变化最大的点
            critical_indices = np.where(np.abs(variance_changes) > np.percentile(np.abs(variance_changes), 85))[0]
            
            for idx in critical_indices:
                critical_info['critical_points'].append({
                    'param_value': param_values[idx],
                    'response_value': responses[idx],
                    'local_variance': variances[idx],
                    'variance_change': variance_changes[idx],
                    'significance': np.abs(variance_changes[idx]) / np.max(np.abs(variance_changes))
                })
        
        return critical_info
    
    def analyze_phase_transitions(self, param1: str, param2: str, 
                                response_var: str) -> Dict:
        """
        分析相变行为
        
        Args:
            param1: 第一个参数
            param2: 第二个参数
            response_var: 响应变量
        
        Returns:
            相变分析结果
        """
        # 创建参数网格
        param1_values = sorted(self.results_data[param1].unique())
        param2_values = sorted(self.results_data[param2].unique())
        
        # 创建响应矩阵
        response_matrix = np.zeros((len(param2_values), len(param1_values)))
        variance_matrix = np.zeros((len(param2_values), len(param1_values)))
        
        for i, p2_val in enumerate(param2_values):
            for j, p1_val in enumerate(param1_values):
                subset = self.results_data[
                    (self.results_data[param1] == p1_val) & 
                    (self.results_data[param2] == p2_val)
                ]
                
                if not subset.empty:
                    response_matrix[i, j] = subset[response_var].mean()
                    variance_matrix[i, j] = subset[response_var].var()
                else:
                    response_matrix[i, j] = np.nan
                    variance_matrix[i, j] = np.nan
        
        # 检测相变边界
        transitions = self._detect_phase_boundaries(response_matrix, param1_values, param2_values)
        
        phase_analysis = {
            'param1': param1,
            'param2': param2,
            'response_variable': response_var,
            'param1_values': param1_values,
            'param2_values': param2_values,
            'response_matrix': response_matrix,
            'variance_matrix': variance_matrix,
            'phase_boundaries': transitions,
            'phase_regions': self._identify_phase_regions(response_matrix, transitions)
        }
        
        return phase_analysis
    
    def _detect_phase_boundaries(self, response_matrix: np.ndarray, 
                                param1_values: List, param2_values: List) -> List[Dict]:
        """检测相边界"""
        boundaries = []
        
        # 计算梯度
        grad_p1, grad_p2 = np.gradient(response_matrix)
        gradient_magnitude = np.sqrt(grad_p1**2 + grad_p2**2)
        
        # 寻找梯度较大的区域作为相边界
        threshold = np.percentile(gradient_magnitude[~np.isnan(gradient_magnitude)], 80)
        
        boundary_points = np.where(gradient_magnitude > threshold)
        
        for i, j in zip(boundary_points[0], boundary_points[1]):
            if i < len(param2_values) and j < len(param1_values):
                boundaries.append({
                    'param1_value': param1_values[j],
                    'param2_value': param2_values[i],
                    'gradient_magnitude': gradient_magnitude[i, j],
                    'response_value': response_matrix[i, j],
                    'boundary_strength': gradient_magnitude[i, j] / np.max(gradient_magnitude)
                })
        
        return boundaries
    
    def _identify_phase_regions(self, response_matrix: np.ndarray, 
                                 boundaries: List[Dict]) -> Dict:
        """识别相区域"""
        # 基于响应值将空间分为不同相
        flat_responses = response_matrix[~np.isnan(response_matrix)].flatten()
        
        if len(flat_responses) < 3:
            return {'phases': [], 'transitions': []}
        
        # 使用聚类或分位数方法识别相
        q25, q50, q75 = np.percentile(flat_responses, [25, 50, 75])
        
        phases = [
            {'name': 'low_response', 'range': [np.min(flat_responses), q25], 'color': 'blue'},
            {'name': 'medium_low_response', 'range': [q25, q50], 'color': 'cyan'},
            {'name': 'medium_response', 'range': [q50, q75], 'color': 'yellow'},
            {'name': 'high_response', 'range': [q75, np.max(flat_responses)], 'color': 'red'}
        ]
        
        return {'phases': phases, 'transitions': boundaries}
    
    def fit_critical_scaling(self, parameter: str, response_var: str, 
                           critical_point: float) -> Dict:
        """
        拟合临界标度律
        
        Args:
            parameter: 参数名称
            response_var: 响应变量
            critical_point: 临界点
        
        Returns:
            标度律拟合结果
        """
        # 获取数据
        param_values = self.results_data[parameter].values
        response_values = self.results_data[response_var].values
        
        # 计算到临界点的距离
        distances = np.abs(param_values - critical_point)
        
        # 只考虑临界点附近的点
        near_critical = distances < np.max(distances) * 0.3
        
        if np.sum(near_critical) < 10:
            return {'error': 'Insufficient data near critical point'}
        
        x_data = distances[near_critical]
        y_data = response_values[near_critical]
        
        # 尝试不同的标度律
        scaling_results = {}
        
        # 幂律标度: R ~ |p - p_c|^β
        try:
            def power_law(x, beta, amplitude):
                return amplitude * (x ** beta)
            
            popt, pcov = curve_fit(power_law, x_data, y_data, 
                                  p0=[1.0, 1.0], bounds=([-5, 0], [5, np.inf]))
            
            scaling_results['power_law'] = {
                'beta': popt[0],
                'amplitude': popt[1],
                'r_squared': self._calculate_r_squared(y_data, power_law(x_data, *popt)),
                'fitted': True
            }
        except:
            scaling_results['power_law'] = {'fitted': False}
        
        # 对数标度: R ~ log|p - p_c|
        try:
            x_log = x_data[x_data > 0]  # 避免log(0)
            y_log = y_data[x_data > 0]
            
            if len(x_log) > 5:
                log_coeffs = np.polyfit(np.log(x_log), y_log, 1)
                log_fit = np.polyval(log_coeffs, np.log(x_log))
                
                scaling_results['log_law'] = {
                    'slope': log_coeffs[0],
                    'intercept': log_coeffs[1],
                    'r_squared': self._calculate_r_squared(y_log, log_fit),
                    'fitted': True
                }
            else:
                scaling_results['log_law'] = {'fitted': False}
        except:
            scaling_results['log_law'] = {'fitted': False}
        
        return {
            'parameter': parameter,
            'response_variable': response_var,
            'critical_point': critical_point,
            'scaling_laws': scaling_results,
            'data_points': len(x_data)
        }
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算R²决定系数"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def analyze_universality(self, parameters: List[str], response_vars: List[str]) -> Dict:
        """
        分析普适性类
        
        Args:
            parameters: 参数列表
            response_vars: 响应变量列表
        
        Returns:
            普适性分析结果
        """
        universality_classes = {}
        
        for param in parameters:
            for response in response_vars:
                # 对每个参数-响应对进行临界分析
                critical_info = self.identify_critical_points(param, response, 'derivative')
                
                if critical_info['critical_points']:
                    # 取最重要的临界点
                    main_critical = max(critical_info['critical_points'], 
                                       key=lambda x: x.get('significance', 0))
                    
                    # 拟合标度律
                    scaling_info = self.fit_critical_scaling(param, response, 
                                                            main_critical['param_value'])
                    
                    universality_classes[f"{param}_{response}"] = {
                        'critical_point': main_critical,
                        'scaling_behavior': scaling_info,
                        'critical_exponents': self._extract_critical_exponents(scaling_info)
                    }
        
        return universality_classes
    
    def _extract_critical_exponents(self, scaling_info: Dict) -> Dict:
        """提取临界指数"""
        exponents = {}
        
        if scaling_info.get('scaling_laws', {}).get('power_law', {}).get('fitted'):
            power_law = scaling_info['scaling_laws']['power_law']
            exponents['beta'] = power_law['beta']
            exponents['nu'] = 1.0  # 假设值，需要更复杂的分析
            
        return exponents
    
    def plot_critical_analysis(self, save_prefix: str = "critical_analysis"):
        """绘制临界分析图表"""
        
        # 1. 临界点识别图
        fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
        
        # 示例：分析gamma参数对崩盘率的影响
        if 'gamma' in self.results_data.columns and 'collapse_detected' in self.results_data.columns:
            gamma_critical = self.identify_critical_points('gamma', 'collapse_detected', 'derivative')
            
            gamma_values = sorted(self.results_data['gamma'].unique())
            collapse_rates = []
            for value in gamma_values:
                subset = self.results_data[self.results_data['gamma'] == value]
                if not subset.empty:
                    collapse_rates.append(subset['collapse_detected'].mean())
                else:
                    collapse_rates.append(0)
            
            # 绘制响应曲线
            axes1[0, 0].plot(gamma_values, collapse_rates, 'bo-', linewidth=2, markersize=6)
            axes1[0, 0].set_xlabel('Gamma')
            axes1[0, 0].set_ylabel('Collapse Rate')
            axes1[0, 0].set_title('Critical Transition in Gamma Parameter')
            axes1[0, 0].grid(True, alpha=0.3)
            
            # 标记临界点
            for cp in gamma_critical['critical_points']:
                axes1[0, 0].axvline(x=cp['param_value'], color='red', linestyle='--', alpha=0.7)
                axes1[0, 0].text(cp['param_value'], cp['response_value'], 
                               f"Critical\nγ={cp['param_value']:.2f}", 
                               rotation=90, fontsize=8, ha='right')
            
            # 绘制导数
            derivatives = np.gradient(collapse_rates, gamma_values)
            axes1[0, 1].plot(gamma_values, derivatives, 'ro-', linewidth=2, markersize=6)
            axes1[0, 1].set_xlabel('Gamma')
            axes1[0, 1].set_ylabel('Derivative')
            axes1[0, 1].set_title('First Derivative (Sensitivity)')
            axes1[0, 1].grid(True, alpha=0.3)
            axes1[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_critical_points.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 相图
        if 'gamma' in self.results_data.columns and 'beta' in self.results_data.columns:
            phase_info = self.analyze_phase_transitions('gamma', 'beta', 'collapse_detected')
            
            fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
            
            # 响应矩阵
            im1 = axes2[0].imshow(phase_info['response_matrix'], 
                                 extent=[phase_info['param1_values'][0], phase_info['param1_values'][-1],
                                        phase_info['param2_values'][0], phase_info['param2_values'][-1]],
                                 aspect='auto', origin='lower', cmap='RdYlBu_r')
            axes2[0].set_xlabel('Gamma')
            axes2[0].set_ylabel('Beta')
            axes2[0].set_title('Phase Diagram: Collapse Rate')
            plt.colorbar(im1, ax=axes2[0], label='Collapse Rate')
            
            # 方差矩阵（表示不确定性）
            im2 = axes2[1].imshow(phase_info['variance_matrix'], 
                                 extent=[phase_info['param1_values'][0], phase_info['param1_values'][-1],
                                        phase_info['param2_values'][0], phase_info['param2_values'][-1]],
                                 aspect='auto', origin='lower', cmap='viridis')
            axes2[1].set_xlabel('Gamma')
            axes2[1].set_ylabel('Beta')
            axes2[1].set_title('Response Variance (Uncertainty)')
            plt.colorbar(im2, ax=axes2[1], label='Variance')
            
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_phase_diagram.png", dpi=300, bbox_inches='tight')
            plt.show()

def load_and_analyze_results(filename: str):
    """加载并分析结果"""
    
    # 加载数据
    df = pd.read_csv(filename)
    
    print(f"加载数据: {len(df)} 条记录")
    print(f"参数列: {[col for col in df.columns if not col.startswith('final_') and not col.startswith('pc_') and col not in ['collapse_detected', 'params']]}")
    print(f"响应列: {[col for col in df.columns if col.startswith('final_') or col.startswith('pc_') or col == 'collapse_detected']}")
    
    # 创建分析器
    analyzer = CriticalConditionAnalyzer(df)
    
    # 分析关键参数
    key_parameters = ['gamma', 'beta', 'p_up', 'r']
    key_responses = ['final_alive_ratio', 'final_c_mean', 'collapse_detected', 'pc_serial_correlation']
    
    print("\n开始临界条件分析...")
    
    # 对每个参数-响应对进行分析
    for param in key_parameters:
        if param in df.columns:
            for response in key_responses:
                if response in df.columns:
                    print(f"分析 {param} -> {response}")
                    critical_info = analyzer.identify_critical_points(param, response, 'derivative')
                    
                    if critical_info['critical_points']:
                        print(f"  发现 {len(critical_info['critical_points'])} 个临界点")
                        for cp in critical_info['critical_points']:
                            print(f"    临界点: {param} = {cp['param_value']:.3f}, "
                                  f"响应 = {cp['response_value']:.3f}, "
                                  f"显著性 = {cp['significance']:.3f}")
    
    # 普适性分析
    print("\n开始普适性分析...")
    universality = analyzer.analyze_universality(key_parameters, key_responses)
    
    for key, value in universality.items():
        if 'critical_point' in value and value['critical_point']:
            cp = value['critical_point']
            print(f"{key}: 临界点 = {cp['param_value']:.3f}")
            
            if 'scaling_behavior' in value and value['scaling_behavior'].get('scaling_laws'):
                scaling = value['scaling_behavior']['scaling_laws']
                if scaling.get('power_law', {}).get('fitted'):
                    beta = scaling['power_law']['beta']
                    r2 = scaling['power_law']['r_squared']
                    print(f"  幂律标度: β = {beta:.3f}, R² = {r2:.3f}")
    
    # 绘制分析图表
    print("\n生成分析图表...")
    analyzer.plot_critical_analysis()
    
    return analyzer


def main():
    """主函数"""
    
    # 查找最新的参数扫描结果
    import glob
    csv_files = glob.glob("*parameter_scan*.csv") + glob.glob("*ultra_fine_scan*.csv")
    
    if not csv_files:
        print("未找到参数扫描结果文件")
        return
    
    # 使用最新的文件
    latest_file = max(csv_files, key=lambda x: x.split('_')[-1].replace('.csv', ''))
    print(f"使用数据文件: {latest_file}")
    
    # 分析结果
    analyzer = load_and_analyze_results(latest_file)
    
    # 保存分析结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    analysis_summary = {
        'data_file': latest_file,
        'total_experiments': len(analyzer.results_data),
        'parameters_analyzed': ['gamma', 'beta', 'p_up', 'r'],
        'responses_analyzed': ['final_alive_ratio', 'final_c_mean', 'collapse_detected', 'pc_serial_correlation'],
        'analysis_timestamp': timestamp
    }
    
    with open(f"critical_analysis_summary_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n分析完成！结果已保存为 critical_analysis_*.png 和 critical_analysis_summary_{timestamp}.json")


if __name__ == "__main__":
    main()