from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import os
from datetime import datetime

class YorumScraper:
    def __init__(self):
        self.driver = None
    
    def tarayici_baslat(self):
        chrome_options = Options()
        chrome_options.add_argument('--start-maximized')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # User-Agent ekle
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        chrome_options.add_argument(f'user-agent={user_agent}')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    def yorum_cek(self, restoran_url, platform="getir"):
        if not self.driver:
            self.tarayici_baslat()
        
        yorumlar = []
        
        try:
            print(f"URL'ye gidiliyor: {restoran_url}")
            self.driver.get(restoran_url)
            time.sleep(10)
            
            print("Yorumlar aranıyor...")
            
            # Sayfayı aşağı kaydır
            self.driver.execute_script("window.scrollTo(0, 400);")
            time.sleep(2)
            
            # Tüm metinleri al
            tum_metinler = self.driver.find_elements(By.XPATH, "//*[string-length(text()) > 20]")
            
            for element in tum_metinler:
                try:
                    text = element.text.strip()
                    
                    # Yorum olabilecek metinleri filtrele
                    if (len(text) > 20 and 
                        not "internet sitemizde" in text.lower() and
                        not "çerez" in text.lower() and
                        not "getir" in text.lower() and
                        not "menü" in text.lower() and
                        not "sipariş vermek için" in text.lower() and
                        not "teknoloji" in text.lower() and
                        not "sosyal" in text.lower() and
                        not "yardım" in text.lower() and
                        not "sıkça" in text.lower() and
                        not "kişisel" in text.lower() and
                        not "bilgi toplumu" in text.lower() and
                        (
                            "lezzet" in text.lower() or
                            "burger" in text.lower() or
                            "sipariş" in text.lower() or
                            "yemek" in text.lower() or
                            "soğuk" in text.lower() or
                            "sıcak" in text.lower() or
                            "porsiyon" in text.lower()
                        )):
                        
                        yorumlar.append({
                            'platform': 'Getir',
                            'yorum': text,
                            'tarih': datetime.now().strftime("%Y-%m-%d")
                        })
                        print(f"Yorum eklendi: {text[:100]}...")
                except:
                    continue
            
            print(f"\nToplam {len(yorumlar)} yorum bulundu.")
            
            return yorumlar
            
        except Exception as e:
            print(f"Genel hata: {e}")
            return []
    
    def veri_setini_kaydet(self, yorumlar):
        """Yorumları CSV dosyasına ekler (var olan yorumları korur)"""
        try:
            if yorumlar:
                # Dosya yolunu oluştur
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                data_dir = os.path.join(project_root, 'data', 'raw')
                file_path = os.path.join(data_dir, 'yorumlar.csv')
                
                # Klasörü oluştur
                os.makedirs(data_dir, exist_ok=True)
                
                # Yeni yorumları DataFrame'e çevir
                yeni_df = pd.DataFrame(yorumlar)
                
                try:
                    # Eğer dosya varsa, mevcut yorumları oku
                    if os.path.exists(file_path):
                        mevcut_df = pd.read_csv(file_path, encoding='utf-8')
                        
                        # Yeni yorumları mevcut yorumlara ekle
                        birlesik_df = pd.concat([mevcut_df, yeni_df], ignore_index=True)
                        
                        # Tekrarlayan yorumları temizle
                        birlesik_df = birlesik_df.drop_duplicates(subset=['yorum'])
                        
                        # Güncellenmiş veriyi kaydet
                        birlesik_df.to_csv(file_path, index=False, encoding='utf-8')
                        print(f"\nMevcut yorumlara {len(yeni_df)} yeni yorum eklendi!")
                        print(f"Toplam yorum sayısı: {len(birlesik_df)}")
                    else:
                        # Dosya yoksa yeni oluştur
                        yeni_df.to_csv(file_path, index=False, encoding='utf-8')
                        print(f"\nYeni dosya oluşturuldu: {file_path}")
                        print(f"Toplam {len(yeni_df)} yorum kaydedildi")
                    
                    print("\nSon eklenen yorumlardan örnekler:")
                    print(yeni_df.head())
                    
                except Exception as e:
                    print(f"CSV işlemleri sırasında hata: {e}")
                    
            else:
                print("Kaydedilecek yorum bulunamadı!")
                
        except Exception as e:
            print(f"Kaydetme hatası: {e}")
    
    def tarayici_kapat(self):
        if self.driver:
            self.driver.quit()

def main():
    scraper = YorumScraper()
    
    # Test URL
    url = "https://getir.com/yemek/restoran/ata-pilav-uskudar-mimar-sinan-mah-uskudar-istanbul/?adTrackingId=34c8128c-374c-4e59-8c24-f5cd208c4a85"
    
    try:
        # Yorumları çek
        print("Yorumlar çekiliyor...")
        yorumlar = scraper.yorum_cek(url)
        
        # Yorumları kaydet
        print("\nYorumlar kaydediliyor...")
        scraper.veri_setini_kaydet(yorumlar)
        
    except Exception as e:
        print(f"Hata: {e}")
    finally:
        scraper.tarayici_kapat()

if __name__ == "__main__":
    main()