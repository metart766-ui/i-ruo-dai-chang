# 王东岳"递弱代偿"理论计算模拟与可视化平台
# (Di-Ruo Dai-Chang Theory Computational Simulation Platform)

本项目旨在通过计算机模拟（元胞自动机）、数学建模（系统可靠性工程）和数据可视化，对王东岳先生的"递弱代偿"哲学理论进行定量验证与探索。

## 🧬 项目核心

我们构建了一个基于**元胞自动机 (Cellular Automata)** 的演化系统，模拟个体在资源约束和熵增压力下的演化路径。

核心验证目标：
1.  **存在度 (P)** 与 **代偿度 (C)** 的负相关关系。
2.  **P × C** 的守恒性趋势。
3.  系统在临界点的**相变 (Phase Transition)** 行为。

## 📂 项目结构

```
.
├── siyan_experiment.py           # [核心] 元胞自动机演化模拟器
├── singularity_evolution.py      # [扩展] "奇点计划"演化模拟器 (引入技术重构机制)
├── mars_mode_stress_test.py      # [测试] "火星模式"极端环境压力测试
├── comprehensive_report_generator.py # [工具] 综合实验报告生成器
├── server.py                     # [后端] FastAPI 数据服务，用于前端交互
├── wandongyu-viz/                # [前端] 基于 React + Ant Design 的现代化可视化看板
├── comprehensive_report/         # [产出] 自动生成的实验数据与图表报告
└── ...
```

## 🚀 快速开始

### 1. 环境准备

需要 Python 3.8+ 和 Node.js 16+。

```bash
# 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装 Python 依赖
pip install numpy pandas matplotlib seaborn scipy fastapi uvicorn pydantic
```

### 2. 运行模拟

您可以直接运行脚本进行离线模拟：

```bash
# 运行标准实验
python3 siyan_experiment.py

# 运行火星模式压力测试
python3 mars_mode_stress_test.py
```

### 3. 启动可视化平台

本项目包含一个基于 Web 的交互式实验室。

**后端服务**:
```bash
python3 server.py
# 服务将启动在 http://localhost:8000
```

**前端界面**:
```bash
cd wandongyu-viz
npm install
npm run dev
# 访问 http://localhost:5173
```

## 📊 实验结论摘要

经过多次模拟（包括极端环境测试），我们发现：
*   **递弱代偿铁律**: 在自然演化条件下，随着代偿度（复杂度）的增加，系统的存在度（鲁棒性）呈现不可逆的下降趋势。
*   **奇点无效**: 即使引入类似 Neuralink 的技术重构机制（Singularity Mode），在热力学第二定律的约束下，系统依然无法长期逆转熵增。
*   **反脆弱性**: 在极端波动的环境（火星模式）中，保持低复杂度的系统反而表现出了最强的生存能力。

## 📜 许可证

MIT License
