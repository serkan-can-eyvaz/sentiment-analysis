import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import json
import os

def predict_sentiment(text):
    try:
        # Cihazı belirle
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model yolu - doğru dosya adı ile
        MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pt')
        
        # BERTürk tokenizer ve model yapısını yükle
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-cased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "dbmdz/bert-base-turkish-128k-cased",
            num_labels=3
        )
        
        # Eğitilmiş model ağırlıklarını yükle
        if os.path.exists(MODEL_PATH):
            print(f"Model yükleniyor: {MODEL_PATH}")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        else:
            raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")
        
        model.to(device)
        model.eval()
        
        # Yorumu tokenize et
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Input'ları device'a taşı
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Tahmin yap
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence_score = predictions[0][predicted_class].item()
        
        # Sınıf etiketleri
        sentiment_labels = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
        
        sentiment = sentiment_labels[predicted_class]
        
        return {
            "sentiment": sentiment,
            "score": confidence_score,
            "text": text
        }
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return {
            "error": str(e),
            "text": text
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Lütfen bir yorum metni girin!"}))
        sys.exit(1)
    
    yorum = sys.argv[1]
    result = predict_sentiment(yorum)
    print(json.dumps(result, ensure_ascii=False)) 