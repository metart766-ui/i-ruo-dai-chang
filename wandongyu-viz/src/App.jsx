import React, { useState, useMemo } from 'react';
import { Player } from '@remotion/player';
import { EvolutionVideo } from './components/EvolutionVideo';
import { PromoVideo } from './components/PromoVideo';
import { 
  Layout, Menu, Card, Statistic, Row, Col, Form, InputNumber, 
  Switch, Button, Slider, Typography, Tag, Space, Alert, Spin,
  theme, ConfigProvider, Select, Radio, Timeline, Descriptions, Badge, Divider,
  Segmented
} from 'antd';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter
} from 'recharts';
import { 
  DashboardOutlined, 
  LineChartOutlined, 
  ExperimentOutlined, 
  FileTextOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  ThunderboltOutlined,
  BuildOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';

// Import data
import reportData from './data/report.json';
import defaultTimeSeriesData from './data/time_series.json';

const { Header, Content, Sider } = Layout;
const { Title, Text, Paragraph } = Typography;

// 配置 Ant Design 主题
const themeConfig = {
  algorithm: theme.defaultAlgorithm,
  token: {
    colorPrimary: '#1677ff',
    borderRadius: 8,
  },
};

const StatCard = ({ title, value, prefix, suffix, color, loading }) => (
  <Card bordered={false} className="shadow-sm hover:shadow-md transition-shadow">
    <Statistic
      title={<Text type="secondary">{title}</Text>}
      value={value}
      precision={2}
      valueStyle={{ color: color || '#000000E0' }}
      prefix={prefix}
      suffix={suffix}
      loading={loading}
    />
  </Card>
);

const ConfigForm = ({ config, setConfig, onRun, loading }) => {
  const [form] = Form.useForm();

  // 当 config 变化时更新表单
  React.useEffect(() => {
    form.setFieldsValue(config);
  }, [config, form]);

  const handleValuesChange = (changedValues) => {
    setConfig({ ...config, ...changedValues });
  };

  return (
    <Card 
      title={<Space><SettingOutlined /><span>实验参数配置</span></Space>} 
      className="shadow-sm mb-6"
      extra={
        <Space>
          <Button 
            onClick={() => {
              setConfig({
                ...config,
                grid_size: 100,
                steps: 5000,
                r: 0.999, // 分子极其稳定
                strategy: 'serial',
                gamma: 1.01, // 几乎没有维护成本
                beta: 0.1,   // 不怕环境
                resource_clustering: 0.8, // 原始汤里的有机分子团
                crowding_cost: 0.0,
                mutation_volatility: 0.05 // 允许跃迁成为细胞
              });
            }}
          >
            🧪 创世纪 (Genesis)
          </Button>
          <Button 
            type="primary" 
            icon={<PlayCircleOutlined />} 
            loading={loading}
            onClick={onRun}
            size="large"
          >
            运行模拟
          </Button>
        </Space>
      }
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={config}
        onValuesChange={handleValuesChange}
      >
        <Row gutter={24}>
          <Col span={8}>
            <Form.Item 
                label="网格大小 (Grid Size)" 
                name="grid_size"
                tooltip="决定了模拟世界的物理空间大小。越大的世界容纳越多生命，但计算越慢。"
            >
              <Slider min={10} max={100} step={10} marks={{10:'10', 50:'50', 100:'100'}} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item 
                label="模拟步数 (Steps)" 
                name="steps"
                tooltip="模拟演化的时间长度。1000步大约相当于文明演化一万年。"
            >
              <InputNumber min={100} max={5000} step={100} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={8}>
             <Form.Item 
                label="基础可靠性 (r)" 
                name="r"
                tooltip="单个零件/细胞不发生故障的概率。0.99 意味着每100次运行有1次故障。"
             >
              <Slider min={0.90} max={0.999} step={0.001} tooltip={{ formatter: (value) => `${value}` }} />
            </Form.Item>
          </Col>
        </Row>
        
        <Row gutter={24}>
           <Col span={24}>
              <Form.Item 
                label="演化策略 (Evolution Strategy)" 
                name="strategy"
                tooltip="文明选择的发展路径：串联结构追求极致效率但脆弱；并联冗余追求安全但消耗巨大能量。"
              >
                <Radio.Group buttonStyle="solid">
                  <Radio.Button value="serial">
                     <Space><BuildOutlined /> 串联结构 (递弱代偿模型 - 越复杂越脆弱)</Space>
                  </Radio.Button>
                  <Radio.Button value="parallel">
                     <Space><BuildOutlined rotate={90} /> 并联冗余 (反脆弱挑战 - 越复杂越安全?)</Space>
                  </Radio.Button>
                </Radio.Group>
              </Form.Item>
           </Col>
        </Row>

        <Row gutter={24}>
          <Col span={8}>
            <Form.Item 
                label="资源聚集度 (Resource Clustering)" 
                name="resource_clustering" 
                tooltip="资源分布的不均匀程度。0=均匀分布，1=极度聚集（富人区与贫民窟）。迫使个体迁徙或竞争。"
            >
              <Slider min={0.0} max={1.0} step={0.1} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item 
                label="内卷系数 (Crowding Cost)" 
                name="crowding_cost" 
                tooltip="拥挤带来的额外能耗。如果周围人太多，生存成本会指数级上升（内卷）。"
            >
              <InputNumber min={0.0} max={1.0} step={0.01} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={8}>
             <Form.Item 
                label="突变剧烈度 (Mutation Volatility)" 
                name="mutation_volatility" 
                tooltip="发生剧烈进化（跃迁）的概率。模拟寒武纪大爆发或突然的退化。"
             >
               <Slider min={0.0} max={0.1} step={0.001} tooltip={{ formatter: (value) => `${(value*100).toFixed(1)}%` }} />
             </Form.Item>
          </Col>
        </Row>

        <Row gutter={24}>
          <Col span={8}>
            <Form.Item 
                label="维护成本指数 (Gamma)" 
                name="gamma" 
                tooltip="复杂度每增加一点，需要消耗多少能量来维持？Gamma > 1 代表成本爆炸增长（熵增）。"
            >
              <InputNumber min={1.0} max={3.0} step={0.1} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item 
                label="环境敏感性 (Beta)" 
                name="beta" 
                tooltip="外界环境变化对个体的影响程度。数值越大，环境一变，死得越快。"
            >
              <InputNumber min={0.0} max={2.0} step={0.1} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={8}>
             <Form.Item 
                label="奇点模式 (Neuralink Mode)" 
                name="enable_singularity" 
                valuePropName="checked"
                tooltip="是否允许文明消耗巨额能量来重构自身代码（技术奇点），试图逆天改命。"
             >
               <Switch 
                 checkedChildren={<Space><ThunderboltOutlined /> 开启</Space>} 
                 unCheckedChildren="关闭" 
               />
             </Form.Item>
          </Col>
        </Row>
        
        <Row gutter={24}>
           <Col span={24}>
              <Form.Item 
                label="平行宇宙对照 (Multiverse A/B Test)" 
                name="dual_mode" 
                valuePropName="checked"
                tooltip="同时运行两个宇宙：A宇宙遵循'递弱代偿'(越复杂越脆弱)，B宇宙遵循'达尔文进化'(越复杂越强)。直接对比两种法则下的文明命运。"
              >
                <Switch 
                  checkedChildren={<Space><ExperimentOutlined /> 双宇宙对比模式开启</Space>} 
                  unCheckedChildren="单宇宙模式" 
                  size="large"
                />
              </Form.Item>
           </Col>
        </Row>

        {config.enable_singularity && (
          <Row gutter={24} className="bg-blue-50 p-4 rounded-lg mb-4">
             <Col span={12}>
                <Form.Item label="重构阈值 (Refactor Threshold)" name="refactor_threshold">
                  <InputNumber min={2} max={10} />
                </Form.Item>
             </Col>
             <Col span={12}>
                <Form.Item label="重构成本 (Refactor Cost)" name="refactor_cost">
                  <InputNumber min={0.1} max={10.0} step={0.1} />
                </Form.Item>
             </Col>
          </Row>
        )}
      </Form>
    </Card>
  );
};

const ExperimentReport = ({ stats, config, timeSeries }) => {
  // 1. 动态分析逻辑
  const finalAliveRatio = stats.alive_ratio;
  const isCollapse = finalAliveRatio < 0.1;
  const isHealthy = finalAliveRatio > 0.8;
  
  // 计算熵增速率 (C的平均增长率)
  const cGrowth = timeSeries.length > 100 
    ? (timeSeries[timeSeries.length-1].c_mean - timeSeries[0].c_mean) 
    : 0;

  // 辅助函数：获取文明阶段描述
  const getCivilizationStage = (c) => {
      if (c < 1.5) return "混沌期 (Chaos) - 原始汤中的随机分子";
      if (c < 2.5) return "单细胞时代 (Prokaryotic) - 简单的生命形式出现";
      if (c < 5.0) return "多细胞爆发 (Cambrian) - 复杂的生物体开始涌现";
      if (c < 8.0) return "原始部落 (Tribal) - 早期社会结构形成";
      if (c < 12.0) return "农业文明 (Agricultural) - 稳定的资源生产体系";
      if (c < 18.0) return "工业革命 (Industrial) - 机械化与能源消耗激增";
      if (c < 25.0) return "信息时代 (Information) - 全球互联的数字网络";
      return "赛博格奇点 (Singularity) - 硅基生命与意识上传";
  };

  // 历史事件提取
  const generateTimeline = () => {
    const events = [];
    events.push({ 
        color: 'green', 
        dot: <ClockCircleOutlined />,
        children: `Step 0: 宇宙大爆炸 - ${getCivilizationStage(config.initial_complexity || 1)}` 
    });
    
    // 寻找关键节点
    let peakC = 0;
    let peakCStep = 0;
    let halfPopStep = null;
    let collapseStep = null;
    let lastStage = "";

    timeSeries.forEach((step, index) => {
        // 记录文明阶段跃迁
        if (index % 100 === 0) { // 每100步检查一次，避免事件太密
             const currentStage = getCivilizationStage(step.c_mean);
             if (currentStage !== lastStage && step.c_mean > 1.5) {
                 events.push({
                     color: 'blue',
                     children: `Step ${step.step}: 文明晋升 - 进入 ${currentStage}`
                 });
                 lastStage = currentStage;
             }
        }

        // 记录复杂度峰值
        if (step.c_mean > peakC) {
            peakC = step.c_mean;
            peakCStep = step.step;
        }
        // 记录人口减半点
        if (!halfPopStep && step.alive_ratio < 0.5) {
            halfPopStep = step.step;
        }
        // 记录崩溃点
        if (!collapseStep && step.alive_ratio < 0.05) {
            collapseStep = step.step;
        }
    });

    if (peakCStep > 0 && peakC > 1.5) {
        events.push({
            color: 'gold',
            dot: <ThunderboltOutlined />,
            children: `Step ${peakCStep}: 黄金时代 (Golden Age) - 复杂度达到巅峰 C=${peakC.toFixed(2)}`
        });
    }

    if (halfPopStep) {
        events.push({
            color: 'orange',
            dot: <ExclamationCircleOutlined />,
            children: `Step ${halfPopStep}: 大衰退 (Great Recession) - 存活率跌破 50%，资源开始枯竭`
        });
    }

    if (collapseStep) {
        events.push({
            color: 'red',
            children: `Step ${collapseStep}: 文明崩溃 (Collapse) - 系统停止响应，如同罗马帝国的陨落`
        });
    } else {
        events.push({
            color: 'green',
            dot: <CheckCircleOutlined />,
            children: `Step ${timeSeries.length > 0 ? timeSeries[timeSeries.length-1].step : 'End'}: 演化终局 - 系统${isHealthy ? '依然健在' : '苟延残喘'}，处于 ${getCivilizationStage(stats.c_mean)}`
        });
    }
    
    // 按时间排序
    return events.sort((a, b) => {
        const stepA = parseInt(a.children.match(/Step (\d+)/)?.[1] || 0);
        const stepB = parseInt(b.children.match(/Step (\d+)/)?.[1] || 0);
        return stepA - stepB;
    });
  };
  
  // 判定死亡原因
  let deathReason = "未知原因";
  if (config.strategy === 'parallel' && stats.p_mean_serial > 0.9 && isCollapse) {
    deathReason = "资源耗尽 (Resource Exhaustion) - 冗余带来了不可承受的能耗成本";
  } else if (stats.p_mean_serial < 0.5) {
    deathReason = "系统故障 (System Failure) - 复杂度过高导致可靠性崩盘";
  } else {
    deathReason = "自然选择压力 (Natural Selection)";
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title={<Space><FileTextOutlined /><span>全维度实验报告 (Experiment Log)</span></Space>} bordered={false}>
        
        {/* 状态总览 */}
        <Alert
          message={isHealthy ? "系统状态：健康 (Stable)" : (isCollapse ? "系统状态：已崩溃 (Collapsed)" : "系统状态：亚健康 (Sub-optimal)")}
          description={`经过 ${config.steps} 步演化，系统最终存活率为 ${(finalAliveRatio * 100).toFixed(2)}%。`}
          type={isHealthy ? "success" : (isCollapse ? "error" : "warning")}
          showIcon
          className="mb-6"
        />

        {/* 平行宇宙对比 (如果开启) */}
        {config.dual_mode && (
            <>
                <Divider orientation="left"><ExperimentOutlined /> 平行宇宙最终对决</Divider>
                <Row gutter={24}>
                    <Col span={12}>
                        <Card type="inner" title="Universe A: 递弱代偿 (Entropy)" className="bg-purple-50">
                            <Descriptions column={1} size="small">
                                <Descriptions.Item label="最终存活率">{(stats.alive_ratio * 100).toFixed(2)}%</Descriptions.Item>
                                <Descriptions.Item label="最终复杂度 (C)">{stats.c_mean.toFixed(2)}</Descriptions.Item>
                                <Descriptions.Item label="最终可靠性 (P)">{stats.p_mean_serial.toFixed(4)}</Descriptions.Item>
                            </Descriptions>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card type="inner" title="Universe B: 达尔文进化 (Darwin)" className="bg-green-50">
                            <Descriptions column={1} size="small">
                                <Descriptions.Item label="最终存活率">{(stats.alive_ratio_b * 100).toFixed(2)}%</Descriptions.Item>
                                <Descriptions.Item label="最终复杂度 (C)">{stats.c_mean_b.toFixed(2)}</Descriptions.Item>
                                <Descriptions.Item label="最终可靠性 (P)">{stats.p_mean_b.toFixed(4)}</Descriptions.Item>
                            </Descriptions>
                        </Card>
                    </Col>
                </Row>
                <div className="mt-4 p-4 bg-gray-100 rounded">
                    <Text strong>对比结论：</Text> 
                    {stats.alive_ratio_b > stats.alive_ratio 
                        ? " 达尔文宇宙胜出。在这个模拟设定下，'优胜劣汰'战胜了'递弱代偿'。但这可能意味着我们的参数设定过于乐观。"
                        : " 递弱代偿宇宙胜出。即便引入了进化优势，热力学熵增依然是不可逾越的高墙。"}
                </div>
                <Divider />
            </>
        )}

        <Row gutter={24}>
            <Col span={14}>
                 <Title level={4}>1. 演化编年史 (Timeline)</Title>
                 <div className="mt-4">
                    <Timeline items={generateTimeline()} />
                 </div>
            </Col>
            <Col span={10}>
                <Title level={4}>2. 关键指标复盘</Title>
                <Descriptions bordered column={1} size="small" className="mt-4">
                    <Descriptions.Item label="初始设定">
                        {config.resource_clustering > 0.5 ? "贫富差距悬殊 (High Clustering)" : "资源均匀分布"}
                    </Descriptions.Item>
                    <Descriptions.Item label="突变策略">
                        {config.mutation_volatility > 0 ? "激进跃迁 (High Volatility)" : "渐进式改良"}
                    </Descriptions.Item>
                    <Descriptions.Item label="内卷程度">
                        {config.crowding_cost > 0.3 ? "高度内卷 (High Crowding Cost)" : "低竞争环境"}
                    </Descriptions.Item>
                    <Descriptions.Item label="代偿增长">
                        <Space>
                            <Badge status={cGrowth > 0 ? "processing" : "default"} />
                            <span>+{cGrowth.toFixed(2)} 单位</span>
                        </Space>
                    </Descriptions.Item>
                    <Descriptions.Item label="主要死因">
                         <Tag color="red">{deathReason}</Tag>
                    </Descriptions.Item>
                </Descriptions>
            </Col>
        </Row>
          
        <div className="bg-gray-50 p-4 rounded-lg mt-8 border border-gray-200">
            <Text type="secondary" italic>
              "我们所见到的一切文明辉煌，不过是物种为了在递弱的存境中苟延残喘，而被迫堆砌出的华丽墓碑。" —— 模拟器总结
            </Text>
        </div>
      </Card>
    </Space>
  );
};

function App() {
  const [activeMenu, setActiveMenu] = useState('overview');
  const [timeSeriesData, setTimeSeriesData] = useState(defaultTimeSeriesData);
  const [loading, setLoading] = useState(false);
  const [videoMode, setVideoMode] = useState('replay'); // 'replay' | 'promo'
  
  // 默认配置
  const [config, setConfig] = useState({
    grid_size: 50,
    steps: 1000,
    gamma: 1.5,
    beta: 0.5,
    r: 0.98,
    strategy: 'serial',
    // v2.0 New Params
    resource_clustering: 0.0,
    crowding_cost: 0.0,
    mutation_volatility: 0.0,
    enable_singularity: false,
    refactor_threshold: 5,
    refactor_cost: 2.0,
    // v3.0 New Params
    dual_mode: false
  });

  const [currentStats, setCurrentStats] = useState({
    alive_ratio: reportData.experiments.basic.basic_stats.alive_ratio.final,
    c_mean: reportData.experiments.basic.basic_stats.c_mean.final,
    p_mean_serial: reportData.experiments.basic.basic_stats.p_mean_serial.final,
    pc_serial: reportData.experiments.basic.basic_stats.pc_serial.final,
    // Dual mode data defaults
    alive_ratio_b: 0,
    c_mean_b: 0,
    p_mean_b: 0
  });

  const runSimulation = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/run_simulation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });
      
      const result = await response.json();
      
      if (result.status === 'success') {
        setTimeSeriesData(result.data);
        
        const lastStep = result.data[result.data.length - 1];
        
        if (result.mode === 'dual') {
             setCurrentStats({
              alive_ratio: lastStep.alive_ratio_a,
              c_mean: lastStep.c_mean_a,
              p_mean_serial: lastStep.p_mean_a,
              pc_serial: lastStep.c_mean_a * lastStep.p_mean_a, // approx
              // Universe B
              alive_ratio_b: lastStep.alive_ratio_b,
              c_mean_b: lastStep.c_mean_b,
              p_mean_b: lastStep.p_mean_b,
            });
        } else {
            setCurrentStats({
              alive_ratio: lastStep.alive_ratio,
              c_mean: lastStep.c_mean,
              p_mean_serial: lastStep.p_mean_serial,
              pc_serial: lastStep.pc_serial,
              // Reset B
              alive_ratio_b: 0,
              c_mean_b: 0,
              p_mean_b: 0
            });
        }
      } else {
        Alert.error('模拟失败', result.detail);
      }
    } catch (error) {
      console.error('Error running simulation:', error);
      alert('运行模拟时出错，请确保后端服务已启动');
    } finally {
      setLoading(false);
    }
  };

  const scatterData = useMemo(() => {
    return timeSeriesData.map(item => ({
      c: item.c_mean,
      p: item.p_mean_serial,
      step: item.step
    }));
  }, [timeSeriesData]);

  const menuItems = [
    { key: 'overview', icon: <DashboardOutlined />, label: '总览看板' },
    { key: 'charts', icon: <LineChartOutlined />, label: '演化分析' },
    { key: 'phase', icon: <ExperimentOutlined />, label: '相变图谱' },
    { key: 'video', icon: <PlayCircleOutlined />, label: '视频回放' },
    { key: 'report', icon: <FileTextOutlined />, label: '实验报告' },
  ];

  return (
    <ConfigProvider theme={themeConfig}>
      <Layout style={{ minHeight: '100vh' }}>
        <Sider theme="light" width={220} className="shadow-md z-10">
          <div className="p-4 flex items-center gap-2 border-b">
            <div className="bg-blue-600 p-1.5 rounded text-white">
              <ExperimentOutlined style={{ fontSize: '20px' }} />
            </div>
            <div>
              <Title level={5} style={{ margin: 0 }}>递弱代偿</Title>
              <Text type="secondary" style={{ fontSize: '10px' }}>Simulatior v1.0</Text>
            </div>
          </div>
          <Menu
            mode="inline"
            selectedKeys={[activeMenu]}
            onClick={({ key }) => setActiveMenu(key)}
            items={menuItems}
            style={{ borderRight: 0, marginTop: '10px' }}
          />
        </Sider>
        
        <Layout className="bg-gray-50">
          <Header className="bg-white px-6 flex justify-between items-center shadow-sm h-16">
            <Title level={4} style={{ margin: 0 }}>
              {menuItems.find(i => i.key === activeMenu)?.label}
            </Title>
            <Space>
               {loading && <Spin />}
               <Tag color={loading ? "processing" : "success"}>
                 {loading ? "计算中..." : "系统就绪"}
               </Tag>
               <Text type="secondary" style={{ fontSize: '12px' }}>
                 最后更新: {new Date().toLocaleTimeString()}
               </Text>
            </Space>
          </Header>
          
          <Content className="p-6 overflow-y-auto">
            {/* Always show config on top for quick access, or move to drawer if needed */}
            <ConfigForm config={config} setConfig={setConfig} onRun={runSimulation} loading={loading} />

            {activeMenu === 'overview' && (
              <Space direction="vertical" size="large" style={{ width: '100%' }}>
                <Row gutter={[16, 16]}>
                  <Col span={6}>
                    <StatCard 
                      title="系统存活率" 
                      value={currentStats.alive_ratio * 100} 
                      suffix="%" 
                      color={currentStats.alive_ratio < 0.5 ? '#cf1322' : '#3f8600'}
                      loading={loading}
                    />
                  </Col>
                  <Col span={6}>
                    <StatCard 
                      title="平均代偿度 (C)" 
                      value={currentStats.c_mean} 
                      color="#1677ff"
                      loading={loading}
                    />
                  </Col>
                  <Col span={6}>
                    <StatCard 
                      title="平均存在度 (P)" 
                      value={currentStats.p_mean_serial} 
                      color="#722ed1"
                      loading={loading}
                    />
                  </Col>
                  <Col span={6}>
                    <StatCard 
                      title="P×C 守恒积" 
                      value={currentStats.pc_serial} 
                      color="#fa8c16"
                      loading={loading}
                    />
                  </Col>
                </Row>

                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card title="P vs C 演化轨迹" bordered={false} className="shadow-sm">
                      <div style={{ height: 350 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" dataKey="c" name="代偿度(C)" domain={['auto', 'auto']} />
                            <YAxis type="number" dataKey="p" name="存在度(P)" domain={['auto', 'auto']} />
                            <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} />
                            <Scatter name="P-C Relation" data={scatterData} fill="#8884d8" />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </div>
                      <Paragraph type="secondary" className="text-center mt-2">
                        验证核心假设：代偿度(C)增加导致存在度(P)下降
                      </Paragraph>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="关键指标时间演化" bordered={false} className="shadow-sm">
                      <div style={{ height: 350 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={timeSeriesData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="step" />
                            <YAxis yAxisId="left" />
                            <YAxis yAxisId="right" orientation="right" />
                            <RechartsTooltip />
                            <Legend />
                            <Line yAxisId="left" type="monotone" dataKey="p_mean_serial" name={config.dual_mode ? "存在度 P (递弱代偿宇宙)" : "存在度 (P)"} stroke="#722ed1" dot={false} strokeWidth={2} />
                            {config.dual_mode && (
                               <Line yAxisId="left" type="monotone" dataKey="p_mean_b" name="存在度 P (达尔文宇宙)" stroke="#52c41a" dot={false} strokeDasharray="5 5" strokeWidth={2} />
                            )}
                            <Line yAxisId="right" type="monotone" dataKey="c_mean" name={config.dual_mode ? "代偿度 C (递弱代偿宇宙)" : "代偿度 (C)"} stroke="#1677ff" dot={false} />
                            {config.dual_mode && (
                               <Line yAxisId="right" type="monotone" dataKey="c_mean_b" name="代偿度 C (达尔文宇宙)" stroke="#13c2c2" dot={false} strokeDasharray="5 5" />
                            )}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </Card>
                  </Col>
                </Row>
              </Space>
            )}

            {activeMenu === 'charts' && (
              <Space direction="vertical" size="large" style={{ width: '100%' }}>
                <Card title="详细多维演化数据" bordered={false}>
                  <div style={{ height: 500 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={timeSeriesData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="step" />
                        <YAxis domain={[0, 'auto']} />
                        <RechartsTooltip />
                        <Legend />
                        <Line type="monotone" dataKey="alive_ratio" name="存活率" stroke="#52c41a" dot={false} strokeWidth={2} />
                        <Line type="monotone" dataKey="p_mean_env" name="环境适应度" stroke="#fa8c16" dot={false} />
                        <Line type="monotone" dataKey="pc_serial" name="P*C (串联)" stroke="#f5222d" dot={false} />
                        {config.enable_singularity && (
                            <Line type="monotone" dataKey="singularity_events" name="奇点事件" stroke="#722ed1" dot={false} />
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card title="静态分析：时间序列概览" bordered={false}>
                      <img src="/images/basic_experiment_time_series.png" style={{ width: '100%' }} alt="Time Series" />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="静态分析：长期演化" bordered={false}>
                      <img src="/images/long_term_evolution.png" style={{ width: '100%' }} alt="Long Term" />
                    </Card>
                  </Col>
                </Row>
              </Space>
            )}

            {activeMenu === 'phase' && (
              <Row gutter={[16, 16]}>
                 <Col span={12}>
                    <Card title="参数敏感性热力图" bordered={false}>
                      <img src="/images/parameter_sensitivity_heatmap.png" style={{ width: '100%', borderRadius: 8 }} alt="Sensitivity" />
                    </Card>
                 </Col>
                 <Col span={12}>
                    <Card title="系统相图 (Phase Diagram)" bordered={false}>
                      <img src="/images/phase_diagram.png" style={{ width: '100%', borderRadius: 8 }} alt="Phase Diagram" />
                    </Card>
                 </Col>
                 <Col span={24}>
                    <Card title="相关性矩阵" bordered={false}>
                      <div className="flex justify-center">
                        <img src="/images/correlation_heatmap.png" style={{ maxHeight: 600, borderRadius: 8 }} alt="Correlation" />
                      </div>
                    </Card>
                 </Col>
              </Row>
            )}

            {activeMenu === 'video' && (
              <Space direction="vertical" size="large" style={{ width: '100%' }}>
                <Card 
                    title="演化过程视频回放 (Remotion Powered)" 
                    bordered={false}
                    extra={
                        <Segmented
                            value={videoMode}
                            onChange={setVideoMode}
                            options={[
                                { label: '演化回放 (Replay)', value: 'replay', icon: <PlayCircleOutlined /> },
                                { label: '项目宣传片 (Promo)', value: 'promo', icon: <ThunderboltOutlined /> },
                            ]}
                        />
                    }
                >
                   <div className="flex justify-center bg-gray-900 p-8 rounded-lg">
                      <Player
                        key={videoMode} // Force re-render when mode changes
                        component={videoMode === 'replay' ? EvolutionVideo : PromoVideo}
                        inputProps={{ data: timeSeriesData, config: config }}
                        durationInFrames={videoMode === 'replay' ? 30 * 10 : 30 * 24} // Replay: 10s, Promo: 24s
                        fps={30}
                        compositionWidth={1280}
                        compositionHeight={720}
                        style={{
                          width: '100%',
                          maxWidth: 800,
                          aspectRatio: '16/9',
                        }}
                        controls
                        autoPlay
                        loop
                      />
                   </div>
                   <div className="mt-4 text-center text-gray-500">
                     <Text type="secondary">
                        {videoMode === 'replay' 
                            ? "* 实时渲染当前的演化数据。调整参数后，视频内容会自动更新。"
                            : "* 自动生成的项目宣传片，包含片头、理论介绍、模拟演示和片尾。"}
                     </Text>
                   </div>
                </Card>
              </Space>
            )}

            {activeMenu === 'report' && (
              <ExperimentReport stats={currentStats} config={config} timeSeries={timeSeriesData} />
            )}
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
}

export default App;
