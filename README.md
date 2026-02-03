# 递弱代偿：文明演化的数学验证 (Di-Ruo Dai-Chang Simulation)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/react-18.0+-61dafb.svg)](https://reactjs.org/)

> "万物存在度的递减，是宇宙演化的根本方向；而代偿度的增加，只是为了对抗这种递减的权宜之计。" —— 王东岳

本项目通过**元胞自动机 (Cellular Automata)** 和 **系统可靠性工程**，将王东岳先生的哲学理论转化为可计算、可验证的数学模型。

---

## 📸 实验看板 (Dashboard)

我们构建了一个现代化的交互式实验室，支持实时参数调整与演化观测。

### 1. 核心指标监控
![Dashboard Overview](wandongyu-viz/public/images/basic_experiment_time_series.png)
*实时追踪存活率、代偿度(C)与存在度(P)的动态关系*

### 2. 相变与临界点分析
![Phase Diagram](wandongyu-viz/public/images/phase_diagram.png)
*系统在不同维护成本($\gamma$)与环境敏感性($\beta$)下的生存相图*

---

## 🧬 核心发现

通过数千次模拟演化，我们的模型揭示了以下规律：

1.  **递弱代偿铁律 (The Iron Law)**
    *   在自然演化条件下，随着代偿度（复杂度 $C$）的上升，系统的存在度（鲁棒性 $P$）呈现不可逆的下降趋势。
    *   验证了 $P \times C \approx k$ 的弱守恒性。

2.  **技术奇点的幻象 (The Singularity Illusion)**
    *   即使引入类似 **Neuralink** 的技术重构机制（允许个体消耗能量降低熵增），在热力学第二定律的约束下，系统依然无法长期逆转崩溃。
    *   *结论：技术本身也是一种代偿，它在解决问题的同时，创造了更大的能量缺口。*

3.  **反脆弱性 (Antifragility in Chaos)**
    *   在 **"火星模式" (Mars Mode)** 的极端环境压力下，保持**低复杂度**的系统反而表现出了最强的生存能力。
    *   *启示：最好的零件就是没有零件 (The best part is no part)。*

---

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/metart766-ui/i-ruo-dai-chang.git
cd i-ruo-dai-chang
```

### 2. 环境准备 (Python)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# 如果没有 requirements.txt，请运行:
# pip install numpy pandas matplotlib seaborn scipy fastapi uvicorn pydantic
```

### 3. 启动后端模拟引擎
```bash
python3 server.py
# 服务启动在 http://localhost:8000
```

### 4. 启动前端可视化 (React)
```bash
cd wandongyu-viz
npm install
npm run dev
# 浏览器访问 http://localhost:5173
```

---

## � 项目结构

*   **核心模拟器**
    *   `siyan_experiment.py`: 基础演化模型（资源约束+随机变异）。
    *   `singularity_evolution.py`: 引入智能体与重构机制的高级模型。
*   **压力测试**
    *   `mars_mode_stress_test.py`: 模拟指数级环境恶化的极端测试。
*   **数据分析**
    *   `comprehensive_report_generator.py`: 生成包含统计学显著性检验的完整报告。
*   **可视化前端**
    *   `wandongyu-viz/`: 基于 React + Ant Design + Recharts 的交互式数据大屏。

---

## 🤝 贡献指南

我们欢迎所有对**复杂系统**、**演化动力学**或**计算哲学**感兴趣的开发者参与贡献！

*   **Issue**: 发现模型漏洞或有新的理论假设？请提交 Issue。
*   **Pull Request**: 欢迎提交代码优化或新的实验场景（例如：引入博弈论机制）。

## 📜 许可证

本项目基于 [MIT License](LICENSE) 开源。
