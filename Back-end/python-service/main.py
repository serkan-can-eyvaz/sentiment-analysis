from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from collections import deque
from typing import List, Dict
import re

app = FastAPI(title="Duygu Analizi Python Servisi")

# CORS ayarlarını genişlet
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm originlere izin ver
    allow_credentials=False,  # Bu False olmalı çünkü allow_origins="*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model ve tokenizer'ı yükle
MODEL_PATH = "C:/Users/serka/Desktop/YapayZeka/yorum-analizi-projesi/models/best_model.pt"
TOKENIZER_PATH = "dbmdz/bert-base-turkish-128k-cased"

# Tokenizer'ı yükle
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

# Modeli yükle
model = BertForSequenceClassification.from_pretrained(
    TOKENIZER_PATH,
    num_labels=3,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Son 5 analizi saklamak için deque kullan
recent_analyses = deque(maxlen=5)

# Yemek isimleri listesi
FOOD_NAMES = [
    r'hünkar beğendi',
    r'mantı',
    r'kebap',
    r'döner',
    r'lahmacun',
    r'pide',
    r'börek',
    r'çorba',
    r'salata',
    r'pilav'
]

# Duygu analizi özel durumları
SENTIMENT_PATTERNS = {
    'olumlu': [
        r'\blezzetli\b',
        r'çok güzel',
        r'harika',
        r'mükemmel',
        r'süper',
        r'iyi',
        r'başarılı',
        r'kötü değil',
        r'fena değil',
        r'beğendim',
        r'tavsiye ederim',
        r'nefis',
        r'şahane',
        r'muhteşem',
        r'bayıldım',
        r'çok beğendim',
        r'harika olmuş',
        r'çok başarılı',
        r'çok iyi',
        r'çok güzel olmuş',
        r'çok lezzetli',
        r'çok nefis',
        r'çok şahane',
        r'çok muhteşem',
        r'çok başarılı',
        r'çok başarılı olmuş',
        r'çok güzel olmuş',
        r'çok lezzetli olmuş',
        r'çok nefis olmuş',
        r'çok şahane olmuş',
        r'çok muhteşem olmuş'
    ],
    'olumsuz': [
        r'lezzetli değil',
        r'kötü',
        r'berbat',
        r'rezalet',
        r'güzel değil',
        r'iyi değil',
        r'başarısız',
        r'beğenmedim',
        r'tavsiye etmem',
        r'çok kötü',
        r'hiç beğenmedim',
        r'lezzetsiz',
        r'kötü olmuş',
        r'berbat olmuş',
        r'rezalet olmuş',
        r'güzel değil olmuş',
        r'iyi değil olmuş',
        r'başarısız olmuş',
        r'beğenmedim olmuş',
        r'tavsiye etmem olmuş',
        r'çok kötü olmuş',
        r'hiç beğenmedim olmuş',
        r'lezzetsiz olmuş',
        r'kötü olmuş',
        r'berbat olmuş',
        r'rezalet olmuş',
        r'güzel değil olmuş',
        r'iyi değil olmuş',
        r'başarısız olmuş',
        r'beğenmedim olmuş',
        r'tavsiye etmem olmuş'
    ],
    'nötr': [
        r'orta',
        r'fena',
        r'idare eder',
        r'normal',
        r'eh işte',
        r'ortalama',
        r'ne iyi ne kötü',
        r'vasat',
        r'standart',
        r'şöyle böyle',
        r'idare eder',
        r'normal',
        r'eh işte',
        r'ortalama',
        r'ne iyi ne kötü',
        r'vasat',
        r'standart',
        r'şöyle böyle',
        r'idare eder',
        r'normal',
        r'eh işte',
        r'ortalama',
        r'ne iyi ne kötü',
        r'vasat',
        r'standart',
        r'şöyle böyle'
    ]
}

def preprocess_text(text: str) -> str:
    """Metin ön işleme fonksiyonu"""
    text = text.lower()
    text = text.replace('İ', 'i').replace('I', 'ı')
    text = re.sub(r'[^\w\s]', '', text)
    return text

def is_food_name(text: str) -> bool:
    """Metin içinde yemek ismi olup olmadığını kontrol et"""
    text = preprocess_text(text)
    for food in FOOD_NAMES:
        if re.search(food, text):
            return True
    return False

def check_special_patterns(text: str) -> str:
    text_processed = preprocess_text(text)
    yemek_var = is_food_name(text)
    best_match = None
    best_sentiment = None
    max_length = 0

    for sentiment, patterns in SENTIMENT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_processed)
            if match:
                match_length = len(match.group())
                if match_length > max_length:
                    max_length = match_length
                    best_match = pattern
                    best_sentiment = sentiment

    if yemek_var and best_sentiment:
        return best_sentiment
    if yemek_var:
        return None
    return best_sentiment

def get_sentiment_description(sentiment: str) -> dict:
    """Duygu durumunun açıklamasını döndür"""
    descriptions = {
        'olumlu': {
            'label': 'OLUMLU',
            'description': 'Bu yorum olumlu bir görüş bildiriyor',
            'emoji': '😊'
        },
        'olumsuz': {
            'label': 'OLUMSUZ',
            'description': 'Bu yorum olumsuz bir görüş bildiriyor',
            'emoji': '😔'
        },
        'nötr': {
            'label': 'NÖTR',
            'description': 'Bu yorum ne olumlu ne olumsuz',
            'emoji': '😐'
        }
    }
    return descriptions.get(sentiment, {
        'label': 'BELİRSİZ',
        'description': 'Yorumun duygu durumu belirlenemedi',
        'emoji': '❓'
    })

def get_sentiment_label(predictions) -> str:
    """Model tahminini Türkçe sınıf isimlerine dönüştür"""
    predicted_class = torch.argmax(predictions).item()
    
    if predicted_class == 0:
        return 'olumsuz'
    elif predicted_class == 1:
        return 'nötr'
    else:
        return 'olumlu'

@app.post("/analyze")
async def analyze_sentiment(request: Request):
    try:
        # Request body'yi oku
        text = await request.body()
        text = text.decode('utf-8')
        print(f"Alınan metin: {text}")  # Debug log
        
        # Önce özel durumları kontrol et
        special_sentiment = check_special_patterns(text)
        
        if special_sentiment:
            print(f"Özel durum bulundu: {special_sentiment}")  # Debug log
            sentiment_info = get_sentiment_description(special_sentiment)
        else:
            # Özel durum bulunamadıysa modeli kullan
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                sentiment = get_sentiment_label(predictions[0])
                print(f"Model tahmini: {sentiment}")  # Debug log
                sentiment_info = get_sentiment_description(sentiment)
        
        # Sonucu oluştur
        result = {
            "text": text,
            "sentiment": sentiment_info['label'],
            "description": sentiment_info['description'],
            "emoji": sentiment_info['emoji']
        }
        
        # Son analizi kaydet
        recent_analyses.append(result)
        
        return result
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recent")
async def get_recent_analyses():
    return list(recent_analyses)

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 