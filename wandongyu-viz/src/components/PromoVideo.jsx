import React from 'react';
import { AbsoluteFill, Sequence, useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { Typography } from 'antd';
import { EvolutionVideo } from './EvolutionVideo';

const { Title, Text } = Typography;

// --- Components ---

const Intro = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  
  const opacity = spring({
    frame,
    fps,
    config: { damping: 200 },
  });
  
  const scale = interpolate(frame, [0, 100], [1, 1.2]);

  return (
    <AbsoluteFill style={{ backgroundColor: 'black', justifyContent: 'center', alignItems: 'center' }}>
      <div style={{ opacity, transform: `scale(${scale})` }}>
        <h1 style={{ color: 'white', fontSize: 80, fontWeight: 900, letterSpacing: 10, textAlign: 'center' }}>
          ENTROPY<br/>COMPENSATOR
        </h1>
        <div style={{ height: 5, width: 200, backgroundColor: '#ff0055', margin: '20px auto' }}></div>
      </div>
    </AbsoluteFill>
  );
};

const Theory = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 20], [0, 1]);
  const y = interpolate(frame, [0, 20], [50, 0]);

  return (
    <AbsoluteFill style={{ backgroundColor: '#111', justifyContent: 'center', alignItems: 'center' }}>
      <div style={{ opacity, transform: `translateY(${y}px)`, textAlign: 'center', padding: 40 }}>
        <h2 style={{ color: '#aaa', fontSize: 40, fontStyle: 'italic' }}>
          "The degree of existence declines..."
        </h2>
        <h3 style={{ color: '#555', marginTop: 20 }}>— Wang Dongyue</h3>
      </div>
    </AbsoluteFill>
  );
};

const Conclusion = () => {
    const frame = useCurrentFrame();
    const scale = spring({ frame, fps: 30, config: { stiffness: 100 } });
    
    return (
      <AbsoluteFill style={{ backgroundColor: 'black', justifyContent: 'center', alignItems: 'center' }}>
        <h1 style={{ color: '#ff0055', fontSize: 70, fontWeight: 'bold', transform: `scale(${scale})` }}>
          COMPLEXITY = COST
        </h1>
      </AbsoluteFill>
    );
};

const Outro = () => {
    return (
      <AbsoluteFill style={{ backgroundColor: 'white', justifyContent: 'center', alignItems: 'center' }}>
        <h2 style={{ color: 'black', fontSize: 50, marginBottom: 20 }}>
          Open Source on GitHub
        </h2>
        <h3 style={{ color: '#1677ff', fontSize: 30 }}>
          github.com/metart766-ui/i-ruo-dai-chang
        </h3>
      </AbsoluteFill>
    );
};

// --- Main Composition ---

export const PromoVideo = ({ data, config }) => {
  return (
    <AbsoluteFill style={{ backgroundColor: 'black' }}>
      {/* 0-3s: Intro */}
      <Sequence from={0} durationInFrames={90}>
        <Intro />
      </Sequence>

      {/* 3-6s: Theory */}
      <Sequence from={90} durationInFrames={90}>
        <Theory />
      </Sequence>

      {/* 6-16s: Simulation Demo (10s) */}
      <Sequence from={180} durationInFrames={300}>
        <AbsoluteFill style={{ backgroundColor: 'white' }}>
            <EvolutionVideo data={data} config={config} />
        </AbsoluteFill>
      </Sequence>

      {/* 16-19s: Conclusion */}
      <Sequence from={480} durationInFrames={90}>
        <Conclusion />
      </Sequence>

      {/* 19-24s: Outro */}
      <Sequence from={570} durationInFrames={150}>
        <Outro />
      </Sequence>
    </AbsoluteFill>
  );
};
