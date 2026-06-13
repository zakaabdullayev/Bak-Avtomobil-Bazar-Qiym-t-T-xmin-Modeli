import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Modeli yaddaşdan yükləyirik
@st.cache_resource # Sayt hər dəfə yenilənəndə modeli təkrar-təkrar yükləyib saytı gecikdirməsin deyə
def load_model():
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Əgər Windows arxa planda uzantını fərqli saxlayıbsa, hər iki ehtimalı sığortalayırıq:
    ehtimal_1 = os.path.join(current_dir, 'baku_car_price_pipeline.joblib')
    ehtimal_2 = os.path.join(current_dir, 'baku_car_price_pipeline.joblib.py') # Python ikonu olduğu üçün
    
    if os.path.exists(ehtimal_1):
        return joblib.load(ehtimal_1)
    elif os.path.exists(ehtimal_2):
        return joblib.load(ehtimal_2)
    else:
        # Əgər heç biri tapılmazsa, qovluqdakı mövcud faylların əsl adını ekrana basıb baxaq:
        fayllar = os.listdir(current_dir)
        raise FileNotFoundError(f"Fayl tapılmadı. Qovluqdakı real fayl adları: {fayllar}")
pipeline = load_model()
# Saytın başlığı və dizaynı
st.set_page_config(page_title="Bakı Avtomobil Bazarı - Qiymət Təxmini", layout="centered")
st.title("🚗 Bakı Avtomobil Bazarı Qiymət Təxmin Modeli")
st.write("Maşının göstəricilərini daxil edin və AI modelinin təyin etdiyi bazar qiymətini görün.")

st.divider()

# 2. İSTİFADƏÇİDƏN MƏLUMATLARIN ALINMASI (İnterfeys elementləri)
col1, col2 = st.columns(2)

with col1:
    marka = st.selectbox("Marka", ["Hyundai", "Mercedes", "BMW", "Toyota", "Kia"]) # Datasetinizdəki populyar markaları bura yaza bilərsiniz
    model = st.text_input("Model (məs: Elantra, C 200)", "Elantra")
    il = st.number_input("Buraxılış ili", min_value=1990, max_value=2026, value=2018)
    yürüş = st.number_input("Yürüş (km)", min_value=0, max_value=1000000, value=95000)
    engine = st.number_input("Mühərrik Həcmi (L)", min_value=0.5, max_value=7.0, value=2.0, step=0.1)
    hp = st.number_input("At Gücü (Horsepower)", min_value=10, max_value=1000, value=150)

with col2:
    city = st.selectbox("Şəhər", ["Bakı", "Sumqayıt", "Gəncə"])
    ban = st.selectbox("Ban növü", ["Sedan", "Universal", "Ofroad / SUV", "Hetçbek"])
    gear = st.selectbox("Sürətlər qutusu", ["Avtomat", "Mexaniki", "Variator", "Robot"])
    fuel = st.selectbox("Yanacaq növü", ["Benzin", "Dizel", "Hibrid", "Elektro"])
    owners = st.selectbox("Sahiblər", ["Birinci", "İkinci", "Üçüncü", "Dördüncü və ya daha çox"])
    veziyyet = st.selectbox("Vəziyyəti", ["Vuruğu yoxdur", "Vuruğu var"])

st.divider()

extra_info = st.text_input("Opsiyalar (extra_info)", "Kondisioner * Lyuk * Yüngül lehimli disklər")
description = st.text_area("Açıqlama (description)", "Maşın ideal vəziyyətdədir, bezkraskadır.")

# 3. AVTOMATİK PREPROCESSING VƏ PROQNOZ FUNKSİYASI
if st.button("💰 Qiyməti Təxmin Et", type="primary"):
    # Gələn datanı lüğət formatına salırıq
    yeni_elan = {
        'city': city, 'vip': 'Xeyr', 'featured': 'Xeyr', 'barter': 'Xeyr', 'loan': 'Xeyr', 'salon': 'Şəxsi',
        'Ban növü': ban, 'Buraxılış ili': il, 'Hansı bazar üçün yığılıb': 'Amerika', 'Marka': marka, 'Model': model,
        'Qəzalı': 'Xeyr', 'spare_parts': 'Xeyr', 'Rəng': 'Gümüşü', 'Sahiblər': owners, 'Sürətlər qutusu': gear,
        'Vəziyyəti': veziyyet, 'Yeni': 'Xeyr', 'Yerlərin sayı': '5', 'Yürüş': yürüş, 'Ötürücü': 'Ön',
        'Engine_Volume': engine, 'Horsepower': hp, 'Fuel_Type': fuel,
        'extra_info': extra_info, 'description': description
    }
    
    input_df = pd.DataFrame([yeni_elan])
    
    # Feature Engineering addımları
    input_df['Car_Age'] = 2026 - input_df['Buraxılış ili']
    input_df['Mileage_Per_Year'] = input_df['Yürüş'] / input_df['Car_Age'].replace(0, 1)
    
    ext = str(input_df['extra_info'].values[0])
    input_df['Total_Options'] = len(ext.split('*')) if ext != '' and ext != 'nan' else 0
    input_df['Has_Lyuk'] = 1 if 'Lyuk' in ext else 0
    input_df['Has_Leather_Salon'] = 1 if 'Dəri salon' in ext else 0
    input_df['Has_Kondisioner'] = 1 if 'Kondisioner' in ext else 0
    
    desc = str(input_df['description'].values[0]).lower()
    input_df['Is_Damaged'] = 1 if any(word in desc for word in ['vuruq', 'udar', 'əzik']) else 0
    input_df['Is_Bezkraska'] = 1 if any(word in desc for word in ['bezkraska', 'rənglənməyib']) else 0
    
    # Sütunları modelin sırasına salırıq
    gözlenilen_sütunlar = [
        'city', 'vip', 'featured', 'barter', 'loan', 'salon', 'Ban növü', 'Buraxılış ili',
        'Hansı bazar üçün yığılıb', 'Marka', 'Model', 'Qəzalı', 'spare_parts', 'Rəng',
        'Sahiblər', 'Sürətlər qutusu', 'Vəziyyəti', 'Yeni', 'Yerlərin sayı', 'Yürüş',
        'Ötürücü', 'Engine_Volume', 'Horsepower', 'Fuel_Type', 'Car_Age', 'Mileage_Per_Year',
        'Total_Options', 'Has_Lyuk', 'Has_Leather_Salon', 'Has_Kondisioner', 'Is_Damaged', 'Is_Bezkraska'
    ]
    input_df = input_df[gözlenilen_sütunlar]
    
    # Təxmin etmə
    log_pred = pipeline.predict(input_df)
    real_qiymet = np.expm1(log_pred)[0]
    
    # Ekrana böyük və yaşıl rənglə çıxarırıq
    st.success(f"### 🎯 Modelin təyin etdiyi bazar qiyməti: {real_qiymet:,.2f} AZN")

