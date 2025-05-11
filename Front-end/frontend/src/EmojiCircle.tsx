import React from 'react';
import { Box } from '@mui/material';

interface EmojiCircleProps {
  emoji: string;
  sentiment: 'OLUMLU' | 'OLUMSUZ' | 'NÖTR';
  size?: number;
}

const sentimentColors: Record<string, string> = {
  OLUMLU: '#ffe600',   // Sarı
  OLUMSUZ: '#ff4d4f',  // Kırmızı
  NÖTR:   '#40a9ff',   // Mavi
};

export default function EmojiCircle({ emoji, sentiment, size = 56 }: EmojiCircleProps) {
  return (
    <Box
      sx={{
        width: size,
        height: size,
        borderRadius: '50%',
        background: sentimentColors[sentiment] || '#eee',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: '0 4px 16px 0 rgba(0,0,0,0.10)',
        fontSize: size * 0.6,
        border: '4px solid white',
        transition: 'background 0.3s',
      }}
    >
      {emoji}
    </Box>
  );
}