import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import pickle
import os
import json
import re
import nltk
from datetime import datetime
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import SMOTE

# Türkçe stop words'leri indir
nltk.download('stopwords')
turkish_stop_words = set(stopwords.words('turkish'))

class YorumDataset(Dataset):
    def __init__(self, texts, labels, max_len=100):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
            
        return self.fc(self.dropout(hidden))

def metin_temizle(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join([word for word in text.split() if word not in turkish_stop_words])
        return text
    return str(text)

def puan_temizle(text):
    if isinstance(text, str):
        sayilar = re.findall(r'\d+', text)
        if sayilar:
            return float(sayilar[0])
    return None

def metrikleri_gorselleştir(y_test, y_pred, sonuclar, train_losses, val_losses):
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
        plt.title('LSTM - Confusion Matrix')
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'lstm_confusion_matrix_{tarih}.png'), dpi=300)
        plt.close()
        
        # 2. Training ve Validation Loss Curves
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Eğitim Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('LSTM - Eğitim ve Validation Loss Değerleri')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(reports_dir, f'lstm_loss_curves_{tarih}.png'), dpi=300)
        plt.close()
        
        # 3. Classification Report Visualization
        cr = classification_report(y_test, y_pred, output_dict=True)
        
        plt.figure(figsize=(12, 6))
        metrics_df = pd.DataFrame({
            'Precision': [cr[str(i)]['precision'] for i in range(3)],
            'Recall': [cr[str(i)]['recall'] for i in range(3)],
            'F1-Score': [cr[str(i)]['f1-score'] for i in range(3)]
        }, index=['Negatif', 'Nötr', 'Pozitif'])
        
        ax = metrics_df.plot(kind='bar', width=0.8)
        plt.title('LSTM - Sınıf Bazında Performans Metrikleri')
        plt.xlabel('Sınıf')
        plt.ylabel('Skor')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Değerleri çubukların üzerine yaz
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'lstm_class_metrics_{tarih}.png'), dpi=300)
        plt.close()
        
        # 4. Model Özeti
        plt.figure(figsize=(10, 6))
        summary_metrics = {
            'Accuracy': sonuclar['metrikler']['accuracy'],
            'F1 Score': sonuclar['metrikler']['f1_score'],
            'Macro Avg': cr['macro avg']['f1-score'],
            'Weighted Avg': cr['weighted avg']['f1-score']
        }
        
        plt.bar(summary_metrics.keys(), summary_metrics.values())
        plt.title('LSTM - Genel Model Performansı')
        plt.ylabel('Skor')
        plt.ylim(0, 1)
        
        # Değerleri çubukların üzerine yaz
        for i, v in enumerate(summary_metrics.values()):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'lstm_model_summary_{tarih}.png'), dpi=300)
        plt.close()
        
        # 5. Epoch Bazında Accuracy
        plt.figure(figsize=(10, 6))
        train_acc = [1 - loss for loss in train_losses]  # Yaklaşık accuracy
        val_acc = [1 - loss for loss in val_losses]      # Yaklaşık accuracy
        
        plt.plot(epochs, train_acc, 'b-', label='Eğitim Accuracy')
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        plt.title('LSTM - Eğitim ve Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(reports_dir, f'lstm_accuracy_curves_{tarih}.png'), dpi=300)
        plt.close()
        
        # Metrikleri text dosyasına kaydet
        with open(os.path.join(reports_dir, f'lstm_metrics_{tarih}.txt'), 'w', encoding='utf-8') as f:
            f.write("LSTM Model Metrikleri\n")
            f.write("===================\n\n")
            f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\n")
            
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\n")
            
            f.write("Model Özeti:\n")
            for metric, value in summary_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        print(f"\nGörselleştirmeler kaydedildi: {reports_dir}")
        print("\nOluşturulan görseller:")
        print("1. Confusion Matrix")
        print("2. Eğitim ve Validation Loss Eğrileri")
        print("3. Sınıf Bazında Performans Metrikleri")
        print("4. Genel Model Performansı")
        print("5. Accuracy Eğrileri")
        print("6. Detaylı Metrik Raporu (TXT)")
        
    except Exception as e:
        print(f"Görselleştirme hatası: {e}")

def model_egit():
    try:
        # CUDA kontrolü
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nKullanılan cihaz: {device}")
        
        # Veri yükleme ve temizleme
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(current_dir), 'data', 'raw', 'yemeklerin_sepeti.csv')
        
        print("Veri okunuyor...")
        df = pd.read_csv(data_path)
        print(f"Toplam {len(df)} yorum okundu.")
        
        # Veri temizleme
        print("\nVeriler temizleniyor...")
        df['temiz_yorum'] = df['yorum'].apply(metin_temizle)
        df['hiz'] = df['hiz'].apply(puan_temizle)
        df['servis'] = df['servis'].apply(puan_temizle)
        df['lezzet'] = df['lezzet'].apply(puan_temizle)
        
        df['ortalama_puan'] = df[['hiz', 'servis', 'lezzet']].mean(axis=1)
        
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
        
        # Veri hazırlama
        print("\nTokenization yapılıyor...")
        MAX_WORDS = 10000
        MAX_LEN = 100
        
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
        tokenizer.fit_on_texts(df['temiz_yorum'])
        
        sequences = tokenizer.texts_to_sequences(df['temiz_yorum'])
        padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Label Encoding
        le = LabelEncoder()
        labels = le.fit_transform(df['sentiment'])
        
        # SMOTE uygula
        print("\nSMOTE uygulanıyor...")
        print("SMOTE öncesi sınıf dağılımı:")
        print(pd.Series(labels).value_counts())
        
        smote = SMOTE(random_state=42)
        padded_sequences_reshaped = padded_sequences.reshape(padded_sequences.shape[0], -1)
        X_balanced, y_balanced = smote.fit_resample(padded_sequences_reshaped, labels)
        
        # SMOTE sonrası şekli düzelt
        X_balanced = X_balanced.reshape(-1, MAX_LEN)
        
        print("\nSMOTE sonrası sınıf dağılımı:")
        print(pd.Series(y_balanced).value_counts())
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, 
            y_balanced,
            test_size=0.2,
            random_state=42,
            stratify=y_balanced
        )
        
        print(f"\nEğitim seti boyutu: {X_train.shape}")
        print(f"Test seti boyutu: {X_test.shape}")
        
        # Dataset oluştur
        train_dataset = YorumDataset(X_train, y_train)
        test_dataset = YorumDataset(X_test, y_test)
        
        # DataLoader
        BATCH_SIZE = 32
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # Model parametreleri
        VOCAB_SIZE = min(len(tokenizer.word_index) + 1, MAX_WORDS)
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 128
        OUTPUT_DIM = 3
        N_LAYERS = 2
        BIDIRECTIONAL = True
        DROPOUT = 0.5
        
        print(f"\nModel parametreleri:")
        print(f"Vocabulary Size: {VOCAB_SIZE}")
        print(f"Embedding Dimension: {EMBEDDING_DIM}")
        print(f"Hidden Dimension: {HIDDEN_DIM}")
        print(f"Batch Size: {BATCH_SIZE}")
        
        # Model oluştur
        model = LSTMClassifier(
            VOCAB_SIZE, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT
        ).to(device)
        
        # Early Stopping için parametreler
        patience = 3
        early_stopping_counter = 0
        best_valid_loss = float('inf')
        
        # Loss ve optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Eğitim
        print("\nModel eğitiliyor...")
        n_epochs = 15  # Epoch sayısını artırdık
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_text, batch_labels in train_loader:
                batch_text = batch_text.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_text)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Accuracy hesapla
                _, predicted = torch.max(predictions.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            train_loss = epoch_loss / len(train_loader)
            train_accuracy = correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation
            model.eval()
            epoch_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_text, batch_labels in test_loader:
                    batch_text = batch_text.to(device)
                    batch_labels = batch_labels.to(device)
                    predictions = model(batch_text)
                    loss = criterion(predictions, batch_labels)
                    epoch_loss += loss.item()
                    
                    # Accuracy hesapla
                    _, predicted = torch.max(predictions.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            val_loss = epoch_loss / len(test_loader)
            val_accuracy = correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            print(f'Epoch: {epoch+1:02} | '
                  f'Train Loss: {train_loss:.3f} | '
                  f'Train Acc: {train_accuracy:.3f} | '
                  f'Val. Loss: {val_loss:.3f} | '
                  f'Val. Acc: {val_accuracy:.3f}')
            
            # Early Stopping kontrolü
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(model.state_dict(), os.path.join(current_dir, 'lstm_model.pth'))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f'\nEarly stopping triggered after epoch {epoch+1}')
                    break
        
        # Test sonuçları
        model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for batch_text, batch_labels in test_loader:
                batch_text = batch_text.to(device)
                predictions = model(batch_text)
                _, predicted = torch.max(predictions.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(batch_labels.numpy())
        
        # Sonuçları raporla
        sonuclar = {
            "model_adi": "lstm_smote",
            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrikler": {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_score": float(f1_score(y_true, y_pred, average='weighted')),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                "classification_report": classification_report(y_true, y_pred, output_dict=True)
            },
            "model_parametreleri": {
                "vocab_size": VOCAB_SIZE,
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": HIDDEN_DIM,
                "n_layers": N_LAYERS,
                "bidirectional": BIDIRECTIONAL,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "epochs": epoch + 1,
                "early_stopping_patience": patience
            },
            "egitim_metrikleri": {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accuracies": train_accuracies,
                "val_accuracies": val_accuracies
            }
        }
        
        # Metrikleri görselleştir
        metrikleri_gorselleştir(y_true, y_pred, sonuclar, train_losses, val_losses)
        
        # Sonuçları kaydet
        reports_dir = os.path.join(os.path.dirname(current_dir), 'reports')
        rapor_dosyasi = os.path.join(
            reports_dir, 
            f'lstm_rapor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(rapor_dosyasi, 'w', encoding='utf-8') as f:
            json.dump(sonuclar, f, ensure_ascii=False, indent=4)
        
        print(f"\nRapor kaydedildi: {rapor_dosyasi}")
        
        # Model performansını göster
        print("\nModel Performansı:")
        print(f"Doğruluk (Accuracy): {sonuclar['metrikler']['accuracy']:.4f}")
        print(f"F1 Score: {sonuclar['metrikler']['f1_score']:.4f}")
        
        # Tokenizer'ı kaydet
        with open(os.path.join(current_dir, 'lstm_tokenizer.pkl'), 'wb') as f:
            pickle.dump(tokenizer, f)
        
        print("\nModel ve tokenizer kaydedildi!")
        
        return model, tokenizer, sonuclar
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None, None, None

if __name__ == "__main__":
    model, tokenizer, sonuclar = model_egit() 