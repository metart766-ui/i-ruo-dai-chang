import React, { useState, useMemo } from 'react';
import { 
  Layout, Menu, Card, Statistic, Row, Col, Form, InputNumber, 
  Switch, Button, Slider, Typography, Tag, Space, Alert, Spin,
  theme, ConfigProvider
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
  ThunderboltOutlined
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

  const handleValuesChange = (changedValues, allValues) => {
    setConfig({ ...config, ...changedValues });
  };

  return (
    <Card 
      title={<Space><SettingOutlined /><span>实验参数配置</span></Space>} 
      className="shadow-sm mb-6"
      extra={
        <Button 
          type="primary" 
          icon={<PlayCircleOutlined />} 
          loading={loading}
          onClick={onRun}
          size="large"
        >
          运行模拟
        </Button>
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
            <Form.Item label="网格大小 (Grid Size)" name="grid_size">
              <Slider min={10} max={100} step={10} marks={{10:'10', 50:'50', 100:'100'}} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item label="模拟步数 (Steps)" name="steps">
              <InputNumber min={100} max={5000} step={100} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={8}>
             <Form.Item label="基础可靠性 (r)" name="r">
              <Slider min={0.90} max={0.999} step={0.001} tooltip={{ formatter: (value) => `${value}` }} />
            </Form.Item>
          </Col>
        </Row>
        
        <Row gutter={24}>
          <Col span={8}>
            <Form.Item label="维护成本指数 (Gamma)" name="gamma" tooltip="越高代表系统越脆弱">
              <InputNumber min={1.0} max={3.0} step={0.1} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item label="环境敏感性 (Beta)" name="beta" tooltip="越高代表环境波动影响越大">
              <InputNumber min={0.0} max={2.0} step={0.1} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={8}>
             <Form.Item label="奇点模式 (Neuralink Mode)" name="enable_singularity" valuePropName="checked">
               <Switch 
                 checkedChildren={<Space><ThunderboltOutlined /> 开启</Space>} 
                 unCheckedChildren="关闭" 
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

function App() {
  const [activeMenu, setActiveMenu] = useState('overview');
  const [timeSeriesData, setTimeSeriesData] = useState(defaultTimeSeriesData);
  const [loading, setLoading] = useState(false);
  
  // 默认配置
  const [config, setConfig] = useState({
    grid_size: 50,
    steps: 1000,
    gamma: 1.5,
    beta: 0.5,
    r: 0.98,
    enable_singularity: false,
    refactor_threshold: 5,
    refactor_cost: 2.0
  });

  const [currentStats, setCurrentStats] = useState({
    alive_ratio: reportData.experiments.basic.basic_stats.alive_ratio.final,
    c_mean: reportData.experiments.basic.basic_stats.c_mean.final,
    p_mean_serial: reportData.experiments.basic.basic_stats.p_mean_serial.final,
    pc_serial: reportData.experiments.basic.basic_stats.pc_serial.final
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
        setCurrentStats({
          alive_ratio: lastStep.alive_ratio,
          c_mean: lastStep.c_mean,
          p_mean_serial: lastStep.p_mean_serial,
          pc_serial: lastStep.pc_serial
        });
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
                            <Line yAxisId="left" type="monotone" dataKey="p_mean_serial" name="存在度 (P)" stroke="#722ed1" dot={false} />
                            <Line yAxisId="right" type="monotone" dataKey="c_mean" name="代偿度 (C)" stroke="#1677ff" dot={false} />
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

            {activeMenu === 'report' && (
              <Card title="实验结论" bordered={false}>
                <Typography>
                  <Title level={3}>主要发现</Title>
                  <Paragraph>
                    <ul>
                      <li>
                        <Text strong>系统稳定性：</Text> 
                        最终存活率为 <Text type={currentStats.alive_ratio > 0.8 ? "success" : "danger"}>{(currentStats.alive_ratio * 100).toFixed(2)}%</Text>。
                      </li>
                      <li>
                        <Text strong>P×C 关系：</Text> 
                        实验数据验证了 P 和 C 之间的显著负相关关系。
                      </li>
                      <li>
                        <Text strong>参数敏感性：</Text> 
                        系统对维护成本指数(γ)最为敏感。
                      </li>
                    </ul>
                  </Paragraph>
                </Typography>
              </Card>
            )}
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
}

export default App;
