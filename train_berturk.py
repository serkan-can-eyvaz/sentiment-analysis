import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
from imblearn.over_sampling import SMOTE
import re
import traceback

class YorumDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None):
        # texts artık bir dictionary olduğu için direkt kullanabiliriz
        self.input_ids = torch.tensor(texts['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(texts['attention_mask'], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

def metrikleri_gorselleştir(y_true, y_pred, train_losses, val_losses, train_accs, val_accs, n_epochs):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(os.path.dirname(current_dir), 'reports', 'visualizations', 'berturk')
        os.makedirs(reports_dir, exist_ok=True)
        
        tarih = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negatif', 'Nötr', 'Pozitif'],
                   yticklabels=['Negatif', 'Nötr', 'Pozitif'])
        plt.title('BERTurk - Confusion Matrix')
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'berturk_confusion_matrix_{tarih}.png'), dpi=300)
        plt.close()
        
        # 2. Classification Report
        cr = classification_report(y_true, y_pred, output_dict=True)
        plt.figure(figsize=(12, 6))
        
        # Sınıf bazlı metrikler
        metrics_df = pd.DataFrame({
            'Precision': [cr[str(i)]['precision'] for i in range(3)],
            'Recall': [cr[str(i)]['recall'] for i in range(3)],
            'F1-Score': [cr[str(i)]['f1-score'] for i in range(3)]
        }, index=['Negatif', 'Nötr', 'Pozitif'])
        
        ax = metrics_df.plot(kind='bar', width=0.8)
        plt.title('BERTurk - Sınıf Bazlı Performans Metrikleri')
        plt.xlabel('Sınıf')
        plt.ylabel('Skor')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Değerleri çubukların üzerine yaz
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'berturk_class_metrics_{tarih}.png'), dpi=300)
        plt.close()
        
        # 3. Training Curves
        plt.figure(figsize=(15, 5))
        
        # Loss eğrileri
        plt.subplot(1, 2, 1)
        plt.plot(range(1, n_epochs + 1), train_losses, 'b-', label='Train Loss')
        plt.plot(range(1, n_epochs + 1), val_losses, 'r-', label='Validation Loss')
        plt.title('BERTurk - Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy eğrileri
        plt.subplot(1, 2, 2)
        plt.plot(range(1, n_epochs + 1), train_accs, 'b-', label='Train Accuracy')
        plt.plot(range(1, n_epochs + 1), val_accs, 'r-', label='Validation Accuracy')
        plt.title('BERTurk - Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'berturk_training_curves_{tarih}.png'), dpi=300)
        plt.close()
        
        # 4. Model Summary
        plt.figure(figsize=(10, 6))
        summary_metrics = {
            'Accuracy': cr['accuracy'],
            'Macro Avg F1': cr['macro avg']['f1-score'],
            'Weighted Avg F1': cr['weighted avg']['f1-score']
        }
        
        plt.bar(summary_metrics.keys(), summary_metrics.values())
        plt.title('BERTurk - Genel Model Performansı')
        plt.ylabel('Skor')
        plt.ylim(0, 1)
        
        # Değerleri çubukların üzerine yaz
        for i, v in enumerate(summary_metrics.values()):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'berturk_model_summary_{tarih}.png'), dpi=300)
        plt.close()
        
        # 5. Detaylı metrikleri TXT dosyasına kaydet
        with open(os.path.join(reports_dir, f'berturk_metrics_{tarih}.txt'), 'w', encoding='utf-8') as f:
            f.write("BERTurk Model Metrikleri\n")
            f.write("=====================\n\n")
            f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\n")
            
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred))
            f.write("\n\n")
            
            f.write("Training History:\n")
            f.write("Epoch\tTrain Loss\tVal Loss\tTrain Acc\tVal Acc\n")
            for i in range(n_epochs):
                f.write(f"{i+1}\t{train_losses[i]:.4f}\t{val_losses[i]:.4f}\t{train_accs[i]:.4f}\t{val_accs[i]:.4f}\n")
        
        print(f"\nGörselleştirmeler kaydedildi: {reports_dir}")
        print("\nOluşturulan görseller:")
        print("1. Confusion Matrix")
        print("2. Sınıf Bazlı Performans Metrikleri")
        print("3. Training Curves")
        print("4. Model Summary")
        print("5. Detaylı Metrikler (TXT)")
        
    except Exception as e:
        print(f"Görselleştirme hatası: {e}")

def train_one_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(test_loader), correct / total, all_preds, all_labels

def model_egit():
    try:
        # CUDA kontrolü
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nKullanılan cihaz: {device}")
        
        # BERTürk model ve tokenizer
        print("\nBERTürk model ve tokenizer yükleniyor...")
        model_name = "dbmdz/bert-base-turkish-128k-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            problem_type="single_label_classification"
        ).to(device)
        
        # Veri yükleme
        print("\nVeri yükleniyor...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(current_dir), 'data', 'raw', 'yemeklerin_sepeti.csv')
        df = pd.read_csv(data_path)
        
        # Puanları temizle ve sentiment hesapla
        print("\nVeriler hazırlanıyor...")
        def puan_temizle(x):
            if isinstance(x, str):
                sayilar = re.findall(r'\d+', x)
                return float(sayilar[0]) if sayilar else None
            return None
        
        df['hiz'] = df['hiz'].apply(puan_temizle)
        df['servis'] = df['servis'].apply(puan_temizle)
        df['lezzet'] = df['lezzet'].apply(puan_temizle)
        
        # NaN değerleri temizle
        df = df.dropna(subset=['yorum', 'hiz', 'servis', 'lezzet'])
        
        # Ortalama puan ve sentiment hesapla
        df['ortalama_puan'] = df[['hiz', 'servis', 'lezzet']].mean(axis=1)
        df['sentiment'] = pd.cut(
            df['ortalama_puan'],
            bins=[0, 5, 7, 10],
            labels=[0, 1, 2]
        ).astype(int)
        
        print("\nSMOTE öncesi sınıf dağılımı:")
        print(df['sentiment'].value_counts())
        
        # Önce metinleri tokenize et
        print("\nMetinler tokenize ediliyor...")
        MAX_LEN = 128
        
        encoded_data = tokenizer.batch_encode_plus(
            df['yorum'].tolist(),
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        
        # SMOTE uygula
        print("\nSMOTE uygulanıyor...")
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        labels = df['sentiment'].values
        
        # Veriyi düzleştir
        input_ids_flat = input_ids.reshape(input_ids.shape[0], -1)
        
        # SMOTE uygula
        smote = SMOTE(random_state=42)
        input_ids_balanced, labels_balanced = smote.fit_resample(input_ids_flat, labels)
        
        # Veriyi orijinal şekline geri döndür
        input_ids_balanced = input_ids_balanced.reshape(-1, MAX_LEN)
        attention_masks_balanced = np.ones_like(input_ids_balanced)
        
        print("\nSMOTE sonrası sınıf dağılımı:")
        print(pd.Series(labels_balanced).value_counts())
        
        # Train-test split
        X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
            input_ids_balanced,
            attention_masks_balanced,
            labels_balanced,
            test_size=0.2,
            random_state=42,
            stratify=labels_balanced
        )
        
        # Dataset oluştur
        print("\nDataLoader'lar oluşturuluyor...")
        train_dataset = YorumDataset(
            texts={
                'input_ids': X_train_ids,
                'attention_mask': X_train_masks
            },
            labels=y_train
        )
        
        test_dataset = YorumDataset(
            texts={
                'input_ids': X_test_ids,
                'attention_mask': X_test_masks
            },
            labels=y_test
        )
        
        # DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0
        )
        
        # Eğitim parametreleri
        n_epochs = 10
        optimizer = AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * n_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Eğitim döngüsü
        print("\nModel eğitiliyor...")
        best_accuracy = 0
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            
            # Eğitim
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, device
            )
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Değerlendirme
            val_loss, val_acc, predictions, true_labels = evaluate(
                model, test_loader, device
            )
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            # En iyi modeli kaydet
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(), 'best_model.pt')
                print("En iyi model kaydedildi!")
        
        # Metrikleri görselleştir
        metrikleri_gorselleştir(
            true_labels, predictions,
            train_losses, val_losses,
            train_accuracies, val_accuracies,
            n_epochs
        )
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\nHata oluştu: {str(e)}")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    model, tokenizer = model_egit() 