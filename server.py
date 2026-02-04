from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from singularity_evolution import SingularitySimulator
from siyan_experiment import SiyanSimulator
import uvicorn
import threading

app = FastAPI(title="Wandongyu Simulation API", description="递弱代偿理论仿真后端")

# 允许跨域请求（前端是 localhost:5173）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为前端的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationConfig(BaseModel):
    # 基础参数
    grid_size: int = 50
    steps: int = 1000
    r: float = 0.98  # 可靠性
    base_death: float = 0.01
    
    # 核心参数
    gamma: float = 1.5  # 维护成本指数
    beta: float = 0.5   # 环境敏感性
    n0: float = 1.0     # 基础依赖
    
    # 演化策略
    strategy: str = 'serial' # 'serial' | 'parallel'
    
    # 复杂环境参数 (v2.0 新增)
    resource_clustering: float = 0.0
    crowding_cost: float = 0.0
    mutation_volatility: float = 0.0
    
    # 奇点参数
    enable_singularity: bool = False
    refactor_threshold: int = 5
    refactor_cost: float = 2.0
    
    # 双宇宙模式 (v3.0)
    dual_mode: bool = False
    
    # 随机种子
    seed: Optional[int] = None

@app.get("/")
async def root():
    return {"message": "Wandongyu Simulation API is running"}

@app.post("/run_simulation")
async def run_simulation(config: SimulationConfig):
    """
    运行单次模拟并返回时间序列数据
    """
    print(f"收到模拟请求: {config}")
    
    try:
        if config.seed is not None:
            np.random.seed(config.seed)
            import random
            random.seed(config.seed)
            
        # 准备参数
        params = config.dict()
        steps = params.pop('steps')
        seed = params.pop('seed')
        
        # 移除不在构造函数中的参数
        # SingularitySimulator 的构造函数接受 enable_singularity, refactor_threshold, refactor_cost 和 **kwargs
        # SiyanSimulator 的构造函数接受 grid_size, alpha, base_cost, gamma, r, n0, n_scale, base_death, beta, p_up, p_down, env_sigma
        
        # 为了通用性，我们需要过滤掉 SiyanSimulator 不需要的参数，但保留 SingularitySimulator 需要的
        # 不过 SingularitySimulator 继承自 SiyanSimulator 并接受 **kwargs，所以多传参数应该没问题
        # 但是 SiyanSimulator 并没有 **kwargs，所以 SingularitySimulator 的 super().__init__(**kwargs) 可能会报错
        # 让我们看看 singularity_evolution.py 的代码...
        # 刚才修复了 singularity_evolution.py，现在它的 __init__ 是：
        # def __init__(self, enable_singularity=False, refactor_threshold=5, refactor_cost=3.0, **kwargs):
        #     self.enable_singularity = enable_singularity
        #     ...
        #     super().__init__(**kwargs)
        
        # 而 SiyanSimulator 的 __init__ (在 siyan_experiment.py 中) 并没有 **kwargs。
        # 这意味着我们需要精确地传递参数给 SiyanSimulator。
        
        siyan_keys = [
            'grid_size', 'alpha', 'base_cost', 'gamma', 'r', 'n0', 'n_scale', 
            'base_death', 'beta', 'p_up', 'p_down', 'env_sigma', 'strategy',
            'resource_clustering', 'crowding_cost', 'mutation_volatility'
        ]
        
        # 默认参数补全 (如果前端没传)
        default_siyan_params = {
            'alpha': 0.2,
            'base_cost': 0.3,
            'n_scale': 0.6,
            'p_up': 0.05,
            'p_down': 0.03,
            'env_sigma': 0.05
        }
        
        # 构建传递给模拟器的参数字典
        sim_params = {}
        
        # 1. 填充 SiyanSimulator 的参数
        for key in siyan_keys:
            if key in params:
                sim_params[key] = params[key]
            elif key in default_siyan_params:
                sim_params[key] = default_siyan_params[key]
        
        # 2. 填充 SingularitySimulator 的特有参数
        if config.enable_singularity:
            sim_params['enable_singularity'] = True
            sim_params['refactor_threshold'] = config.refactor_threshold
            sim_params['refactor_cost'] = config.refactor_cost
        else:
            sim_params['enable_singularity'] = False
            # 即使禁用，为了避免 __init__ 报错，也可以传（如果构造函数定义了默认值）
            # 或者我们直接实例化 SingularitySimulator，它会处理 enable_singularity=False 的情况
            
        print(f"模拟器参数: {sim_params}")
        
        # 使用 SingularitySimulator，因为它兼容两者（通过 enable_singularity 开关）
        if config.dual_mode:
            print("启动双宇宙平行模拟...")
            # 宇宙 A (当前参数，默认 strategy)
            # 如果前端传的是 'serial'，那就是标准的递弱代偿宇宙
            # 如果前端传的是 'parallel'，那就是两个都是 parallel，没意义
            # 所以在 dual_mode 下，我们强制：
            # Universe A = 'serial' (Entropy World)
            # Universe B = 'darwin' (Darwinian World) - 我们刚才在 siyan_experiment.py 里加了 'darwin' 策略
            
            # Universe A
            params_a = sim_params.copy()
            params_a['strategy'] = 'serial'
            sim_a = SingularitySimulator(**params_a)
            sim_a.run_simulation(steps)
            
            # Universe B
            params_b = sim_params.copy()
            params_b['strategy'] = 'darwin' # 必须确保 SiyanSimulator 支持这个
            sim_b = SingularitySimulator(**params_b)
            sim_b.run_simulation(steps)
            
            # 合并结果
            history_a = sim_a.history
            history_b = sim_b.history
            
            result_data = []
            for i in range(len(history_a['step'])):
                item = {
                    'step': int(history_a['step'][i]),
                    # Universe A Data
                    'alive_ratio_a': float(history_a['alive_ratio'][i]),
                    'c_mean_a': float(history_a['c_mean'][i]),
                    'p_mean_a': float(history_a['p_mean_serial'][i]),
                    # Universe B Data
                    'alive_ratio_b': float(history_b['alive_ratio'][i]),
                    'c_mean_b': float(history_b['c_mean'][i]),
                    'p_mean_b': float(history_b['p_mean_serial'][i]), # 注意：这里虽然叫 p_mean_serial，但其实是 B 宇宙的 P
                }
                result_data.append(item)
                
            return {
                "status": "success",
                "mode": "dual",
                "data": result_data
            }
            
        else:
            # 单宇宙模式 (兼容旧版)
            simulator = SingularitySimulator(**sim_params)
            simulator.run_simulation(steps)
            
            # 提取结果
            history = simulator.history
            
            # 转换为前端友好的格式 (List of Dicts)
            result_data = []
            for i in range(len(history['step'])):
                item = {
                    'step': int(history['step'][i]),
                    'alive_ratio': float(history['alive_ratio'][i]),
                    'c_mean': float(history['c_mean'][i]),
                    'p_mean_serial': float(history['p_mean_serial'][i]),
                    'pc_serial': float(history['pc_serial'][i]),
                }
                # 如果有奇点事件数据
                if 'singularity_events' in history and i < len(history['singularity_events']):
                     item['singularity_events'] = int(history['singularity_events'][i])
                
                result_data.append(item)
                
            return {
                "status": "success",
                "mode": "single",
                "data": result_data,
                "final_stats": {
                    "alive_ratio": float(history['alive_ratio'][-1]),
                    "c_mean": float(history['c_mean'][-1]),
                    "p_mean_serial": float(history['p_mean_serial'][-1])
                }
            }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
