import React from 'react';
import { Box, Tooltip, Typography } from '@mui/material';

interface SentimentMapProps {
  analyses: {
    emoji: string;
    text: string;
  }[];
}

const mapWidth = 320;
const mapHeight = 480;
const emojiSize = 44;
const minDistance = 60; // Minimum mesafe (px)

function generatePositions(count: number) {
  const positions: { left: number; top: number }[] = [];
  let attempts = 0;
  for (let i = 0; i < count; i++) {
    let pos: { left: number; top: number } = { left: 0, top: 0 };
    let safe = false;
    while (!safe && attempts < 1000) {
      attempts++;
      const x = Math.random() * (mapWidth - emojiSize);
      const y = Math.random() * (mapHeight - emojiSize);
      pos = { left: x, top: y };
      safe = positions.every(
        (p) =>
          Math.hypot(p.left - x, p.top - y) > minDistance
      );
    }
    positions.push(pos);
  }
  return positions;
}

const SentimentMap: React.FC<SentimentMapProps> = ({ analyses }) => {
  const positions = React.useMemo(
    () => generatePositions(analyses.length),
    [analyses.length]
  );

  return (
    <Box
      sx={{
        width: mapWidth,
        height: mapHeight,
        background: 'linear-gradient(135deg, #f8fafc 60%, #e0e7ef 100%)',
        borderRadius: 5,
        boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
        border: '1.5px solid #e3e8ee',
        position: 'relative',
        overflow: 'hidden',
        p: 2,
        mb: 2,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        transition: 'box-shadow 0.3s',
        '&:hover': {
          boxShadow: '0 12px 40px 0 rgba(31, 38, 135, 0.22)',
        },
      }}
    >
      <Typography variant="h6" sx={{ mb: 2, textAlign: 'center', fontWeight: 'bold', letterSpacing: 1 }}>
        Duygu Haritası
      </Typography>
      <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
        {analyses.map((a, i) => {
          const pos = positions[i];
          return (
            <Tooltip key={i} title={a.text} arrow placement="top">
              <Box
                sx={{
                  position: 'absolute',
                  left: pos.left,
                  top: pos.top,
                  fontSize: emojiSize,
                  cursor: 'pointer',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'scale(1.25)',
                    boxShadow: '0 2px 12px 0 rgba(0,0,0,0.12)',
                    zIndex: 3,
                  },
                  zIndex: 2,
                  userSelect: 'none',
                }}
              >
                {a.emoji}
              </Box>
            </Tooltip>
          );
        })}
      </Box>
    </Box>
  );
};

export default SentimentMap;