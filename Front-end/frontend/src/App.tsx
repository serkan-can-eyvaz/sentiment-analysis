import React, { useState, useEffect } from 'react';
import {
    Container, CssBaseline, AppBar, Toolbar, Typography,
    TextField, Button, Box, Paper, CircularProgress, List,
    ListItem, ListItemText, Divider, Alert, IconButton
} from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { createTheme } from '@mui/material/styles';
import { PsychologyOutlined, GitHub, Help, Settings } from '@mui/icons-material';
import './App.css';
import EmojiCircle from './EmojiCircle';
import Rating from '@mui/material/Rating';
import SentimentMap from './SentimentMap';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

interface AnalysisResult {
  text: string;
  sentiment: string;
  description: string;
  emoji: string;
  rating?: number;
}

function App() {
  const [comment, setComment] = useState('');
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResult | null>(null);
  const [recentAnalyses, setRecentAnalyses] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userRating, setUserRating] = useState<number | null>(null);
  const [showAnimatedEmoji, setShowAnimatedEmoji] = useState(false);
  const [animatedEmoji, setAnimatedEmoji] = useState<string | null>(null);

  const fetchRecentAnalyses = async () => {
    try {
      const response = await fetch('http://localhost:8000/recent');
      if (!response.ok) throw new Error('Son analizler alınamadı');
      const data = await response.json();
      setRecentAnalyses(data || []);
    } catch (error: any) {
      console.error('Son analizleri alma hatası:', error);
      setError('Son analizler yüklenirken bir sorun oluştu.');
    }
  };

  useEffect(() => {
    fetchRecentAnalyses();
  }, []);

  const analyzeSentiment = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!comment.trim()) return;
    setLoading(true);
    setError(null);
    setCurrentAnalysis(null);
    setUserRating(null);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'text/plain',
        },
        body: comment,
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(errorData || 'Analiz sırasında bir sunucu hatası oluştu');
      }

      const data: AnalysisResult = await response.json();
      if (!data || !data.sentiment) {
        throw new Error('Geçersiz yanıt formatı');
      }

      setCurrentAnalysis(data);
      setAnimatedEmoji(data.emoji);
      setShowAnimatedEmoji(true);
      setTimeout(() => setShowAnimatedEmoji(false), 2200);
      setComment('');
      setRecentAnalyses(prev => [data, ...prev].slice(0, 5));

    } catch (error: any) {
      console.error('Analiz hatası:', error);
      setError(error.message || 'Analiz sırasında bilinmeyen bir hata oluştu.');
    } finally {
      setLoading(false);
    }
  };

  const handleRatingChange = (newValue: number | null) => {
    setUserRating(newValue);
    if (currentAnalysis) {
      setCurrentAnalysis({ ...currentAnalysis, rating: newValue ?? undefined });
      setRecentAnalyses(prev =>
        prev.map((item) =>
          item.text === currentAnalysis.text
            ? { ...item, rating: newValue ?? undefined }
            : item
        )
      );
    }
  };

  const getSentimentColor = (sentiment: string): string => {
      switch (sentiment) {
          case 'OLUMLU': return 'success.main';
          case 'OLUMSUZ': return 'error.main';
          case 'NÖTR': return 'warning.main';
          default: return 'text.secondary';
      }
  };

  const handleClearList = () => {
    setRecentAnalyses([]);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static" sx={{ mb: 4 }}>
        <Toolbar>
          <PsychologyOutlined sx={{ fontSize: 32, mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Duygu Analiz Sistemi
          </Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button color="inherit" startIcon={<Help />}>
              Yardım
            </Button>
            <IconButton color="inherit" size="large">
              <GitHub />
            </IconButton>
            <IconButton color="inherit" size="large">
              <Settings />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>
      
      <Container component="main" maxWidth="md" sx={{ position: 'relative' }}>
        {/* Sol üstte sabit Duygu Haritası */}
        <Box
          sx={{
            position: 'absolute',
            left: -340,
            top: 40,
            zIndex: 10,
          }}
        >
          <SentimentMap analyses={recentAnalyses} />
        </Box>
        {/* Sağ üstte sabit Canlı Yorum Akışı */}
        <Box
          sx={{
            position: 'absolute',
            right: -340,
            top: 40,
            zIndex: 10,
            width: 320,
            height: 480,
            background: 'linear-gradient(135deg, #f8fafc 60%, #e0e7ef 100%)',
            borderRadius: 5,
            boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
            border: '1.5px solid #e3e8ee',
            p: 2,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            overflow: 'hidden',
          }}
        >
          <Typography variant="h6" sx={{ mb: 2, textAlign: 'center', fontWeight: 'bold', letterSpacing: 1 }}>
            Canlı Yorum Akışı
          </Typography>
          <Box sx={{ width: '100%', flex: 1, overflowY: 'auto', pr: 1 }}>
            {recentAnalyses.length === 0 && (
              <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', mt: 4 }}>
                Henüz canlı yorum yok.
              </Typography>
            )}
            {recentAnalyses.map((a, i) => (
              <Box key={i} sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  Birisi az önce
                </Typography>
                <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>
                  '{a.text.length > 30 ? a.text.slice(0, 30) + '...' : a.text}'
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 700, color: getSentimentColor(a.sentiment) }}>
                  – {a.sentiment}
                </Typography>
                <span style={{ fontSize: 22, marginLeft: 4 }}>{a.emoji}</span>
              </Box>
            ))}
          </Box>
        </Box>
        {/* Orta alan: Analiz ve Sonuçlar */}
        <Box>
          <Box component="form" onSubmit={analyzeSentiment} sx={{ mt: 3 }}>
            <TextField
              fullWidth
              multiline
              rows={4}
              variant="outlined"
              label="Analiz edilecek metni girin"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              sx={{ mb: 2 }}
              disabled={loading}
            />
            <Button
              type="submit"
              variant="contained"
              color="primary"
              fullWidth
              size="large"
              disabled={loading}
              sx={{ height: 48 }}
            >
              {loading ? (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CircularProgress size={24} color="inherit" />
                  <Typography>Analiz Ediliyor...</Typography>
                </Box>
              ) : (
                'Analiz Et'
              )}
            </Button>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}

          {currentAnalysis && (
            <Paper elevation={3} sx={{ p: 3, mt: 3, borderRadius: 2 }}>
              <Typography variant="h6" gutterBottom>
                Anlık Analiz Sonucu
              </Typography>
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 2, 
                mb: 2,
                p: 2,
                borderRadius: 1,
                bgcolor: 'background.default'
              }}>
                <EmojiCircle emoji={currentAnalysis.emoji} sentiment={currentAnalysis.sentiment as 'OLUMLU' | 'OLUMSUZ' | 'NÖTR'} />
                <Box>
                  <Typography
                    variant="h5"
                    component="span"
                    sx={{ 
                      color: getSentimentColor(currentAnalysis.sentiment), 
                      fontWeight: 'bold',
                      display: 'block',
                      mb: 0.5
                    }}
                  >
                    {currentAnalysis.sentiment}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {currentAnalysis.description}
                  </Typography>
                </Box>
              </Box>
              <Typography 
                variant="body1" 
                sx={{ 
                  p: 2, 
                  bgcolor: 'background.paper',
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: 'divider'
                }}
              >
                "{currentAnalysis.text}"
              </Typography>
              <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
                <Typography>Sonucu Oyla:</Typography>
                <Rating
                  name="user-rating"
                  value={userRating}
                  onChange={(_, newValue) => handleRatingChange(newValue)}
                  disabled={userRating !== null}
                />
                {userRating && <Typography color="success.main">Teşekkürler!</Typography>}
              </Box>
            </Paper>
          )}

          <Paper elevation={3} sx={{ p: 3, mt: 3, borderRadius: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
              <Typography variant="h6" gutterBottom>
                Son 5 Analiz
              </Typography>
              <Button variant="outlined" color="secondary" size="small" onClick={handleClearList}>
                Listeyi Temizle
              </Button>
            </Box>
            {recentAnalyses.length > 0 ? (
              <List disablePadding>
                {recentAnalyses.map((analysis, index) => (
                  <React.Fragment key={index}>
                    <ListItem 
                      alignItems="flex-start"
                      sx={{
                        p: 2,
                        '&:hover': {
                          bgcolor: 'action.hover',
                          borderRadius: 1
                        }
                      }}
                    >
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            <EmojiCircle emoji={analysis.emoji} sentiment={analysis.sentiment as 'OLUMLU' | 'OLUMSUZ' | 'NÖTR'} />
                            <Typography
                              component="span"
                              variant="subtitle1"
                              sx={{ 
                                color: getSentimentColor(analysis.sentiment), 
                                fontWeight: 'bold' 
                              }}
                            >
                              {analysis.sentiment}
                            </Typography>
                            <Rating
                              name={`list-rating-${index}`}
                              value={analysis.rating ?? null}
                              size="small"
                              readOnly
                              sx={{ ml: 1 }}
                            />
                          </Box>
                        }
                        secondary={
                          <React.Fragment>
                            <Typography
                              sx={{ display: 'inline' }}
                              component="span"
                              variant="body2"
                              color="text.primary"
                            >
                              "{analysis.text}"
                            </Typography>
                            <Typography
                              component="span"
                              variant="body2"
                              color="text.secondary"
                              sx={{ display: 'block', mt: 0.5 }}
                            >
                              {analysis.description}
                            </Typography>
                          </React.Fragment>
                        }
                      />
                    </ListItem>
                    {index < recentAnalyses.length - 1 && <Divider variant="inset" component="li" />}
                  </React.Fragment>
                ))}
              </List>
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                Henüz analiz yapılmadı.
              </Typography>
            )}
          </Paper>
        </Box>
      </Container>

      {showAnimatedEmoji && animatedEmoji && (
        <Box
          sx={{
            position: 'fixed',
            left: 0,
            top: 0,
            width: '100vw',
            height: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            pointerEvents: 'none',
            zIndex: 2000,
          }}
        >
          <Box
            sx={{
              fontSize: { xs: 80, md: 120 },
              animation: 'popAndFade 2s cubic-bezier(0.23, 1, 0.32, 1)',
              filter: 'drop-shadow(0 8px 32px rgba(0,0,0,0.18))',
            }}
          >
            {animatedEmoji}
          </Box>
          <style>
            {`
              @keyframes popAndFade {
                0% { opacity: 0; transform: scale(0.5);}
                20% { opacity: 1; transform: scale(1.2);}
                60% { opacity: 1; transform: scale(1);}
                100% { opacity: 0; transform: scale(0.7);}
              }
            `}
          </style>
        </Box>
      )}
    </ThemeProvider>
  );
}

export default App;
 