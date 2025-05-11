import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

def model_sonuclarini_gorselleştir(rapor_data):
    """Model sonuçlarını görselleştirir ve reports klasörüne kaydeder"""
    try:
        # Reports klasörünü bul
        current_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(os.path.dirname(current_dir), 'reports')
        vizualizasyon_dir = os.path.join(reports_dir, 'visualizations')
        os.makedirs(vizualizasyon_dir, exist_ok=True)
        
        tarih = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(10, 8))
        conf_matrix = np.array(rapor_data['metrikler']['confusion_matrix'])
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negatif', 'Nötr', 'Pozitif'],
                   yticklabels=['Negatif', 'Nötr', 'Pozitif'])
        plt.title('Confusion Matrix')
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek Değer')
        plt.tight_layout()
        plt.savefig(os.path.join(vizualizasyon_dir, f'confusion_matrix_{tarih}.png'))
        plt.close()
        
        # 2. Sınıf Dağılımı Pasta Grafiği
        plt.figure(figsize=(10, 8))
        class_dist = rapor_data['metrikler']['classification_report']
        class_counts = {k: v['support'] for k, v in class_dist.items() 
                       if k in ['negatif', 'notr', 'pozitif']}
        
        plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%',
                colors=sns.color_palette("pastel"))
        plt.title('Sınıf Dağılımı')
        plt.savefig(os.path.join(vizualizasyon_dir, f'class_distribution_{tarih}.png'))
        plt.close()
        
        # 3. Performans Metrikleri Bar Grafiği
        plt.figure(figsize=(12, 6))
        metrics = {
            'Accuracy': rapor_data['metrikler']['accuracy'],
            'F1 Score': rapor_data['metrikler']['f1_score'],
            'Precision (Avg)': np.mean([v['precision'] for k, v in class_dist.items() 
                                      if k in ['negatif', 'notr', 'pozitif']]),
            'Recall (Avg)': np.mean([v['recall'] for k, v in class_dist.items() 
                                   if k in ['negatif', 'notr', 'pozitif']])
        }
        
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
        plt.title('Model Performans Metrikleri')
        plt.ylim(0, 1)
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(vizualizasyon_dir, f'performance_metrics_{tarih}.png'))
        plt.close()
        
        # 4. Sınıf Bazında Performans
        plt.figure(figsize=(12, 6))
        class_metrics = {k: [v['precision'], v['recall'], v['f1-score']] 
                        for k, v in class_dist.items() 
                        if k in ['negatif', 'notr', 'pozitif']}
        
        df_metrics = pd.DataFrame(class_metrics, index=['Precision', 'Recall', 'F1-Score']).T
        df_metrics.plot(kind='bar', width=0.8)
        plt.title('Sınıf Bazında Performans Metrikleri')
        plt.xlabel('Sınıf')
        plt.ylabel('Skor')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(vizualizasyon_dir, f'class_metrics_{tarih}.png'))
        plt.close()
        
        print(f"\nGörselleştirmeler kaydedildi: {vizualizasyon_dir}")
        
        return vizualizasyon_dir
        
    except Exception as e:
        print(f"Görselleştirme hatası: {e}")
        return None 