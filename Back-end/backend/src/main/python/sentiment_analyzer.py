import sys
import json
from transformers import pipeline

def analyze_sentiment(text):
    # Duygu analizi modelini yükle
    sentiment_analyzer = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")
    
    # Metni analiz et
    result = sentiment_analyzer(text)[0]
    
    # Sonucu JSON formatında döndür
    return json.dumps({
        "label": result["label"],
        "score": result["score"]
    })

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = sys.argv[1]
        print(analyze_sentiment(text))
    else:
        print(json.dumps({"error": "Metin parametresi gerekli"})) 