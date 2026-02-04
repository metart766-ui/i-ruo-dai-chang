import React from 'react';
import { useCurrentFrame, useVideoConfig, AbsoluteFill, interpolate } from 'remotion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Legend } from 'recharts';
import { Typography, Card } from 'antd';

const { Title, Text } = Typography;

export const EvolutionVideo = ({ data, config }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  // Safety check
  if (!data || data.length === 0) {
    return (
      <AbsoluteFill style={{ backgroundColor: 'white', justifyContent: 'center', alignItems: 'center' }}>
        <Title level={3}>No Data Available</Title>
      </AbsoluteFill>
    );
  }

  // 计算当前应该展示多少数据点
  // 假设总数据量对应视频总时长
  const totalSteps = data.length;
  const progress = interpolate(frame, [0, durationInFrames], [0, 1], {
    extrapolateRight: "clamp",
  });
  
  const currentStepIndex = Math.floor(progress * totalSteps);
  const currentData = data.slice(0, Math.max(2, currentStepIndex));
  const currentStepData = data[currentStepIndex] || data[data.length - 1];

  // Safety check for step data
  if (!currentStepData) return null;

  // 动态样式
  const opacity = interpolate(frame, [0, 30], [0, 1]);
  
  // 颜色定义
  const colors = {
    entropy: '#ff4d4f', // Red
    darwin: '#52c41a',  // Green
    complexity: '#1677ff' // Blue
  };

  return (
    <AbsoluteFill style={{ backgroundColor: 'white', padding: 40, opacity }}>
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="mb-8 text-center">
          <Title level={2} style={{ margin: 0 }}>
            Entropy Compensator Simulation
          </Title>
          <Text type="secondary" style={{ fontSize: 20 }}>
            {config.dual_mode ? "Multiverse Comparison: Entropy vs Darwin" : "Evolution Timeline"}
          </Text>
        </div>

        {/* Main Chart Area */}
        <div className="flex-1 relative">
          <div style={{ width: '100%', height: '100%' }}>
            <LineChart width={1000} height={500} data={currentData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" domain={[0, (data[data.length-1]?.step || 100)]} />
              <YAxis yAxisId="left" domain={[0, 1.1]} />
              <YAxis yAxisId="right" orientation="right" />
              <Legend />
              
              {/* Universe A: Entropy */}
              <Line 
                yAxisId="left" 
                type="monotone" 
                dataKey="p_mean_serial" 
                name={config.dual_mode ? "Existence (Entropy)" : "Existence P"} 
                stroke={colors.entropy} 
                strokeWidth={3}
                dot={false}
                isAnimationActive={false}
              />
              
              {/* Universe B: Darwin (if dual mode) */}
              {config.dual_mode && (
                <Line 
                  yAxisId="left" 
                  type="monotone" 
                  dataKey="p_mean_b" 
                  name="Existence (Darwin)" 
                  stroke={colors.darwin} 
                  strokeWidth={3} 
                  strokeDasharray="5 5"
                  dot={false}
                  isAnimationActive={false}
                />
              )}
              
              {/* Complexity */}
              <Line 
                yAxisId="right" 
                type="monotone" 
                dataKey="c_mean" 
                name="Complexity C" 
                stroke={colors.complexity} 
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </div>
          
          {/* Overlay Stats */}
          <div className="absolute top-4 right-4 bg-white/80 p-4 rounded-lg shadow-lg border border-gray-200">
             <div className="text-2xl font-bold mb-2">Step: {currentStepData?.step || 0}</div>
             
             <div style={{ color: colors.entropy }}>
               P (Entropy): {currentStepData?.p_mean_serial?.toFixed(4) || 0}
             </div>
             
             {config.dual_mode && (
               <div style={{ color: colors.darwin }}>
                 P (Darwin): {currentStepData?.p_mean_b?.toFixed(4) || 0}
               </div>
             )}
             
             <div style={{ color: colors.complexity }}>
               C (Complexity): {currentStepData?.c_mean?.toFixed(2) || 0}
             </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-gray-500 italic">
           "The degree of existence of all things declines over time..."
        </div>
      </div>
    </AbsoluteFill>
  );
};
