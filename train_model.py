import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
import json
from datetime import datetime
from visualize_model import model_sonuclarini_gorselleştir
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Türkçe stop words'leri indir
nltk.download('stopwords')
turkish_stop_words = set(stopwords.words('turkish'))

def metin_temizle(text):
    """Metni temizler ve hazırlar"""
    if isinstance(text, str):
        # Küçük harfe çevir
        text = text.lower()
        # Sadece temel noktalama işaretlerini kaldır
        text = re.sub(r'[!@#$%^&*(),.?":{}|<>]', ' ', text)
        # Fazla boşlukları temizle
        text = ' '.join(text.split())
        return text
    return str(text)

def puan_temizle(text):
    """'Hız: 10' gibi metinlerden sayıyı çıkarır"""
    if isinstance(text, str):
        # Sayıyı bul
        sayilar = re.findall(r'\d+', text)
        if sayilar:
            return float(sayilar[0])
    return None

def ortalama_puan_hesapla(row):
    """Hız, servis ve lezzet puanlarının ortalamasını alır"""
    puanlar = [row['hiz'], row['servis'], row['lezzet']]
    return np.mean([p for p in puanlar if not np.isnan(p)])

def sentiment_belirle(puan, ortalama, std):
    """Puanı ortalamaya göre sınıflandırır"""
    if puan > ortalama + (0.5 * std):
        return 'pozitif'
    elif puan < ortalama - (0.5 * std):
        return 'negatif'
    else:
        return 'notr'

def sonuclari_raporla(y_test, y_pred, model_adi="naive_bayes"):
    """Eğitim sonuçlarını raporlar ve kaydeder"""
    try:
        # Metrikleri hesapla
        conf_matrix = confusion_matrix(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Rapor için sonuçları hazırla
        sonuclar = {
            "model_adi": model_adi,
            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrikler": {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": clf_report
            },
            "model_parametreleri": {
                "vectorizer": "TfidfVectorizer",
                "max_features": 5000
            }
        }
        
        # Reports klasörünü oluştur
        current_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(os.path.dirname(current_dir), 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Rapor dosyası adını oluştur
        tarih = datetime.now().strftime("%Y%m%d_%H%M%S")
        rapor_dosyasi = os.path.join(reports_dir, f'{model_adi}_rapor_{tarih}.json')
        
        # Raporu JSON formatında kaydet
        with open(rapor_dosyasi, 'w', encoding='utf-8') as f:
            json.dump(sonuclar, f, ensure_ascii=False, indent=4)
        
        print(f"\nRapor kaydedildi: {rapor_dosyasi}")
        
        return sonuclar
        
    except Exception as e:
        print(f"Rapor oluşturma hatası: {e}")
        return None

def metrikleri_gorselleştir(y_test, y_pred, sonuclar):
    """Model metriklerini görselleştirir ve kaydeder"""
    try:
        # Reports klasörünü bul ve visualizations alt klasörünü oluştur
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
        plt.title('Confusion Matrix')
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'confusion_matrix_{tarih}.png'))
        plt.close()
        
        # 2. Sınıf Bazında Metrikler
        plt.figure(figsize=(12, 6))
        cr = classification_report(y_test, y_pred, output_dict=True)
        metrics_df = pd.DataFrame({
            'Precision': [cr[label]['precision'] for label in ['negatif', 'notr', 'pozitif']],
            'Recall': [cr[label]['recall'] for label in ['negatif', 'notr', 'pozitif']],
            'F1-Score': [cr[label]['f1-score'] for label in ['negatif', 'notr', 'pozitif']]
        }, index=['Negatif', 'Nötr', 'Pozitif'])
        
        metrics_df.plot(kind='bar', width=0.8)
        plt.title('Sınıf Bazında Performans Metrikleri')
        plt.xlabel('Sınıf')
        plt.ylabel('Skor')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'class_metrics_{tarih}.png'))
        plt.close()
        
        # 3. Genel Metrikler
        plt.figure(figsize=(8, 6))
        genel_metrikler = {
            'Accuracy': sonuclar['metrikler']['accuracy'],
            'F1 Score': sonuclar['metrikler']['f1_score'],
            'Macro Avg': cr['macro avg']['f1-score'],
            'Weighted Avg': cr['weighted avg']['f1-score']
        }
        
        plt.bar(genel_metrikler.keys(), genel_metrikler.values())
        plt.title('Genel Model Performansı')
        plt.ylabel('Skor')
        plt.ylim(0, 1)
        
        # Değerleri çubukların üzerine yaz
        for i, v in enumerate(genel_metrikler.values()):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'overall_metrics_{tarih}.png'))
        plt.close()
        
        # 4. Sınıf Dağılımı
        plt.figure(figsize=(10, 6))
        class_dist = pd.Series(y_test).value_counts()
        plt.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%',
                colors=sns.color_palette("pastel"))
        plt.title('Test Veri Seti Sınıf Dağılımı')
        plt.savefig(os.path.join(reports_dir, f'class_distribution_{tarih}.png'))
        plt.close()
        
        print(f"\nGörselleştirmeler kaydedildi: {reports_dir}")
        print("Oluşturulan görseller:")
        print("1. Confusion Matrix")
        print("2. Sınıf Bazında Metrikler")
        print("3. Genel Model Performansı")
        print("4. Sınıf Dağılımı")
        
    except Exception as e:
        print(f"Görselleştirme hatası: {e}")

def model_egit():
    try:
        # Veri dosyası yolu
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_path = os.path.join(project_root, 'data', 'raw', 'yemeklerin_sepeti.csv')
        
        # Veriyi oku
        print("Veri okunuyor...")
        print(f"Dosya yolu: {data_path}")
        df = pd.read_csv(data_path)
        
        print(f"Toplam {len(df)} yorum okundu.")
        print("Sütunlar:", df.columns.tolist())
        
        # Puanları temizle
        print("\nPuanlar temizleniyor...")
        df['hiz'] = df['hiz'].apply(puan_temizle)
        df['servis'] = df['servis'].apply(puan_temizle)
        df['lezzet'] = df['lezzet'].apply(puan_temizle)
        
        # Ortalama puanı hesapla
        df['ortalama_puan'] = df[['hiz', 'servis', 'lezzet']].mean(axis=1)
        
        print("\nPuan dağılımı:")
        print(df[['hiz', 'servis', 'lezzet', 'ortalama_puan']].describe())
        
        # Veriyi temizle
        print("\nYorumlar temizleniyor...")
        df['temiz_yorum'] = df['yorum'].apply(metin_temizle)
        
        # Geçersiz verileri filtrele
        df = df[
            (df['temiz_yorum'].str.len() > 3) & 
            (df['ortalama_puan'].notna()) &
            (df['ortalama_puan'] > 0) &
            (df['ortalama_puan'] <= 10)  # Max puan 10
        ]
        
        # Puanları kategorilere ayır
        puan_ortalama = df['ortalama_puan'].mean()
        puan_std = df['ortalama_puan'].std()
        
        print("\nPuan İstatistikleri:")
        print(f"Ortalama Puan: {puan_ortalama:.2f}")
        print(f"Standart Sapma: {puan_std:.2f}")
        
        # Sentiment sınıflandırması (1-10 skalası için)
        df['sentiment'] = pd.cut(
            df['ortalama_puan'],
            bins=[0, 5, 7, 10],  # 5 altı negatif, 5-7 arası nötr, 7 üstü pozitif
            labels=['negatif', 'notr', 'pozitif']
        )
        
        print("\nVeri seti istatistikleri:")
        print(f"Toplam yorum sayısı: {len(df)}")
        print("\nSentiment dağılımı:")
        print(df['sentiment'].value_counts())
        
        print("\nÖrnek yorumlar:")
        for sentiment in ['pozitif', 'negatif', 'notr']:
            print(f"\n{sentiment.upper()} örnek yorum:")
            sample = df[df['sentiment'] == sentiment].sample(1)
            if not sample.empty:
                print(f"Puan: {sample['ortalama_puan'].iloc[0]:.2f}")
                print(f"Yorum: {sample['yorum'].iloc[0]}")
        
        # Verileri hazırla
        X = df['temiz_yorum']
        y = df['sentiment']
        
        # TF-IDF dönüşümü
        print("\nTF-IDF dönüşümü yapılıyor...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(X)
        
        # SMOTE uygula
        print("\nSMOTE uygulanıyor...")
        print("Orijinal sınıf dağılımı:")
        print(Counter(y))
        
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_tfidf, y)
        
        print("\nSMOTE sonrası sınıf dağılımı:")
        print(Counter(y_balanced))
        
        # Train-test split
        print("\nVeriler eğitim ve test olarak ayrılıyor...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, 
            test_size=0.2, 
            random_state=42,
            stratify=y_balanced
        )
        
        # Model eğitimi
        print("\nModel eğitiliyor...")
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # Tahminler
        print("\nModel test ediliyor...")
        y_pred = model.predict(X_test)
        
        # Metrikleri hesapla
        sonuclar = {
            "model_adi": "naive_bayes_smote",
            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrikler": {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred, average='weighted')),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            },
            "model_parametreleri": {
                "vectorizer": "TfidfVectorizer",
                "max_features": 5000,
                "balancing": "SMOTE"
            }
        }
        
        # Metrikleri görselleştir
        metrikleri_gorselleştir(y_test, y_pred, sonuclar)
        
        # Raporu kaydet
        reports_dir = os.path.join(os.path.dirname(current_dir), 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        rapor_dosyasi = os.path.join(
            reports_dir, 
            f'naive_bayes_smote_rapor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(rapor_dosyasi, 'w', encoding='utf-8') as f:
            json.dump(sonuclar, f, ensure_ascii=False, indent=4)
        
        print(f"\nRapor kaydedildi: {rapor_dosyasi}")
        
        # Model performansını göster
        print("\nModel Performansı:")
        print(f"Doğruluk (Accuracy): {sonuclar['metrikler']['accuracy']:.4f}")
        print(f"F1 Score: {sonuclar['metrikler']['f1_score']:.4f}")
        
        print("\nKarmaşıklık Matrisi:")
        print(np.array(sonuclar['metrikler']['confusion_matrix']))
        
        # Örnek tahminler
        print("\nÖrnek tahminler:")
        test_yorumlar = [
            "Yemekler çok lezzetliydi, kesinlikle tavsiye ederim",
            "Soğuk geldi, servis çok yavaştı",
            "Fiyatı biraz yüksek ama idare eder"
        ]
        
        for yorum in test_yorumlar:
            temiz_yorum = metin_temizle(yorum)
            yorum_vector = vectorizer.transform([temiz_yorum])
            tahmin = model.predict(yorum_vector)[0]
            print(f"\nYorum: {yorum}")
            print(f"Tahmin: {tahmin}")
        
        # Model ve vektörizeri kaydet
        print("\nModel kaydediliyor...")
        with open('naive_bayes_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print("Model başarıyla kaydedildi!")
        
        return model, vectorizer, sonuclar
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None, None, None

def yorum_tahmin_et(model, vectorizer, yorum):
    """Yeni bir yorumu tahmin eder"""
    try:
        temiz_yorum = metin_temizle(yorum)
        yorum_vector = vectorizer.transform([temiz_yorum])
        tahmin = model.predict(yorum_vector)[0]
        return tahmin
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return None

if __name__ == "__main__":
    # Modeli eğit
    model, vectorizer, sonuclar = model_egit()
    
    if model and vectorizer:
        # Test için örnek yorumlar
        test_yorumlar = [
            "Yemekler çok lezzetliydi, kesinlikle tavsiye ederim",
            "Soğuk geldi, servis çok yavaştı",
            "Fiyatı biraz yüksek ama idare eder"
        ]
        
        print("\nÖrnek tahminler:")
        for yorum in test_yorumlar:
            tahmin = yorum_tahmin_et(model, vectorizer, yorum)
            print(f"\nYorum: {yorum}")
            print(f"Tahmin: {tahmin}") 