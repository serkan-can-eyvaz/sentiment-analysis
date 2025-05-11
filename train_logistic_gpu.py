import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import pickle
import os
import json
import re
import nltk
from datetime import datetime
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Türkçe stop words'leri indir
nltk.download('stopwords')
turkish_stop_words = set(stopwords.words('turkish'))

def metin_temizle(text):
    """Metni temizler ve hazırlar"""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[!@#$%^&*(),.?":{}|<>]', ' ', text)
        text = ' '.join(text.split())
        return text
    return str(text)

def puan_temizle(text):
    """'Hız: 10' gibi metinlerden sayıyı çıkarır"""
    if isinstance(text, str):
        sayilar = re.findall(r'\d+', text)
        if sayilar:
            return float(sayilar[0])
    return None

class YorumDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.toarray())
        self.y = torch.LongTensor(pd.get_dummies(y).values)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LogisticRegressionGPU(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionGPU, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def metrikleri_gorselleştir(y_test, y_pred, sonuclar, train_losses=None):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(os.path.dirname(current_dir), 'reports', 'visualizations')
        os.makedirs(reports_dir, exist_ok=True)
        
        tarih = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negatif', 'Nötr', 'Pozitif'],
                   yticklabels=['Negatif', 'Nötr', 'Pozitif'])
        plt.title('Confusion Matrix - Logistic Regression (GPU)')
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'logistic_gpu_confusion_matrix_{tarih}.png'), dpi=300)
        plt.close()
        
        # 2. Classification Report Görselleştirme
        plt.figure(figsize=(12, 8))
        cr = classification_report(y_test, y_pred, output_dict=True)
        
        # Sınıf bazlı metrikler
        metrics_df = pd.DataFrame({
            'Precision': [cr[str(i)]['precision'] for i in range(3)],
            'Recall': [cr[str(i)]['recall'] for i in range(3)],
            'F1-Score': [cr[str(i)]['f1-score'] for i in range(3)]
        }, index=['Negatif', 'Nötr', 'Pozitif'])
        
        ax = metrics_df.plot(kind='bar', width=0.8, figsize=(12, 6))
        plt.title('Sınıf Bazlı Performans Metrikleri')
        plt.xlabel('Sınıf')
        plt.ylabel('Skor')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Değerleri çubukların üzerine yaz
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'logistic_gpu_class_metrics_{tarih}.png'), dpi=300)
        plt.close()
        
        # 3. ROC Curve
        plt.figure(figsize=(10, 8))
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        y_score = label_binarize(y_pred, classes=[0, 1, 2])
        
        for i in range(3):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve (class {i}) (AUC = {roc_auc:0.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Multi-class')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(reports_dir, f'logistic_gpu_roc_curve_{tarih}.png'), dpi=300)
        plt.close()
        
        # 4. Training Loss Curve (eğer varsa)
        if train_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-')
            plt.title('Eğitim Loss Değerleri')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(reports_dir, f'logistic_gpu_training_loss_{tarih}.png'), dpi=300)
            plt.close()
        
        # 5. Model Özeti
        plt.figure(figsize=(12, 8))
        summary_data = {
            'Accuracy': [sonuclar['metrikler']['accuracy']],
            'F1 Score': [sonuclar['metrikler']['f1_score']],
            'Macro Avg': [cr['macro avg']['f1-score']],
            'Weighted Avg': [cr['weighted avg']['f1-score']]
        }
        summary_df = pd.DataFrame(summary_data)
        
        ax = summary_df.plot(kind='bar', width=0.8, figsize=(10, 6))
        plt.title('Genel Model Performansı')
        plt.xlabel('Metrik')
        plt.ylabel('Skor')
        plt.grid(True, alpha=0.3)
        
        # Değerleri çubukların üzerine yaz
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'logistic_gpu_model_summary_{tarih}.png'), dpi=300)
        plt.close()
        
        print(f"\nGörselleştirmeler kaydedildi: {reports_dir}")
        print("Oluşturulan görseller:")
        print("1. Confusion Matrix")
        print("2. Sınıf Bazlı Performans Metrikleri")
        print("3. ROC Curve")
        print("4. Eğitim Loss Değerleri (eğer varsa)")
        print("5. Genel Model Performansı")
        
    except Exception as e:
        print(f"Görselleştirme hatası: {e}")

def model_egit():
    try:
        # CUDA kontrolü
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nKullanılan cihaz: {device}")
        
        # Veri dosyası yolu
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(current_dir), 'data', 'raw', 'yemeklerin_sepeti.csv')
        
        print("Veri okunuyor...")
        df = pd.read_csv(data_path)
        print(f"Toplam {len(df)} yorum okundu.")
        
        # Puanları temizle
        print("\nPuanlar temizleniyor...")
        df['hiz'] = df['hiz'].apply(puan_temizle)
        df['servis'] = df['servis'].apply(puan_temizle)
        df['lezzet'] = df['lezzet'].apply(puan_temizle)
        
        # Ortalama puanı hesapla
        df['ortalama_puan'] = df[['hiz', 'servis', 'lezzet']].mean(axis=1)
        
        # Yorumları temizle
        print("\nYorumlar temizleniyor...")
        df['temiz_yorum'] = df['yorum'].apply(metin_temizle)
        
        # Geçersiz verileri filtrele
        df = df[
            (df['temiz_yorum'].str.len() > 3) & 
            (df['ortalama_puan'].notna()) &
            (df['ortalama_puan'] > 0) &
            (df['ortalama_puan'] <= 10)
        ]
        
        # Sentiment sınıflandırması
        df['sentiment'] = pd.cut(
            df['ortalama_puan'],
            bins=[0, 5, 7, 10],
            labels=['negatif', 'notr', 'pozitif']
        )
        
        print("\nVeri seti istatistikleri:")
        print(df['sentiment'].value_counts())
        
        # TF-IDF dönüşümü
        print("\nTF-IDF dönüşümü yapılıyor...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['temiz_yorum'])
        y = pd.Categorical(df['sentiment']).codes
        
        # SMOTE uygula
        print("\nSMOTE uygulanıyor...")
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Verileri böl
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        # Dataset ve DataLoader oluştur
        train_dataset = YorumDataset(X_train, y_train)
        test_dataset = YorumDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        # Model oluştur
        input_dim = X_train.shape[1]
        num_classes = 3
        model = LogisticRegressionGPU(input_dim, num_classes).to(device)
        
        # Eğitim parametreleri
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10
        
        # Eğitim
        print("\nModel eğitiliyor...")
        train_losses = []  # Loss değerlerini takip etmek için
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.float())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            epoch_loss = total_loss/len(train_loader)
            train_losses.append(epoch_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        # Test
        print("\nModel test ediliyor...")
        model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                predicted = torch.argmax(outputs, dim=1).cpu().numpy()
                y_pred.extend(predicted)
                y_true.extend(torch.argmax(batch_y, dim=1).numpy())
        
        # Sonuçları raporla
        sonuclar = {
            "model_adi": "logistic_regression_gpu",
            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrikler": {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_score": float(f1_score(y_true, y_pred, average='weighted')),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                "classification_report": classification_report(y_true, y_pred, output_dict=True)
            },
            "model_parametreleri": {
                "epochs": num_epochs,
                "batch_size": 64,
                "learning_rate": 0.001,
                "device": str(device)
            }
        }
        
        # Metrikleri görselleştir (train_losses'ı da gönder)
        metrikleri_gorselleştir(y_true, y_pred, sonuclar, train_losses)
        
        # Sonuçları kaydet
        reports_dir = os.path.join(os.path.dirname(current_dir), 'reports')
        rapor_dosyasi = os.path.join(
            reports_dir, 
            f'logistic_regression_gpu_rapor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(rapor_dosyasi, 'w', encoding='utf-8') as f:
            json.dump(sonuclar, f, ensure_ascii=False, indent=4)
        
        print(f"\nRapor kaydedildi: {rapor_dosyasi}")
        
        # Model performansını göster
        print("\nModel Performansı:")
        print(f"Doğruluk (Accuracy): {sonuclar['metrikler']['accuracy']:.4f}")
        print(f"F1 Score: {sonuclar['metrikler']['f1_score']:.4f}")
        
        # Modeli kaydet
        torch.save(model.state_dict(), os.path.join(current_dir, 'logistic_regression_gpu_model.pth'))
        with open(os.path.join(current_dir, 'logistic_tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print("\nModel kaydedildi!")
        
        return model, vectorizer, sonuclar
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None, None, None

if __name__ == "__main__":
    model, vectorizer, sonuclar = model_egit() 