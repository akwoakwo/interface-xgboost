import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import joblib 
import os
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier


st.set_page_config(layout="wide", page_title="Klasifikasi Data Diabetes")

st.sidebar.title("🔍 Fitur")
page = st.sidebar.radio("Pilih Menu :", [
    "Ringkasan Penelitian",
    "Proses Klasifikasi Data",
    "Prediksi Baru"
])

st.title("🩺 IMPLEMENTASI METODE OVERSAMPLING ADAPTIVE SYNTHETIC SAMPLING (ADASYN) PADA KLASIFIKASI DIABETES MELITUS MENGGUNAKAN MODEL EXTREME GRADIENT BOOSTING (XGBOOST)")

if page == "Ringkasan Penelitian":
    st.subheader("1.1 Latar Belakang")
    st.write("Perkembangan signifikan di dunia medis dan pencegahan berbagai macam penyakit sangat diperlukan sebagai deteksi dini dan prediksi resiko infeksi juga menjadi prioritas yang harus diperhatikan dalam upaya pengendalian penyebaran penyakit. Penerapan klasifikasi dapat digunakan karena dengan teknik ini, data akan dibagi menjadi data training untuk mengetahui data yang dikaitkan dengan label atau kelas dan data testing sebagai sampel uji penentuan label atau kelas. Dalam proses klasifikasi, model akan memperlihatkan kumpulan fitur yang saling berhubungan pada diagnosa dan melihat pola atau hubungan guna membangun model yang dapat mengenali pola atau hubungan yang dicari sehingga dapat diklasifikasikan untuk menghasilkan nilai dengan akurasi yang tinggi. Ketika metode klasifikasi diimplementasikan pada data yang mengalami ketidakseimbangan kelas, model klasifikasi akan lebih mengabaikan kelas minoritas akibat kecenderungan nilai klasifikasi akan dengan mudah terpolarisasi kepada kelas mayoritas. Data yang tidak seimbang (Imbalance Data) dapat terjadi ketika salah satu atau beberapa kelas lebih dominan terhadap keseluruhan data.")
    
    st.subheader("1.2 Pertanyaan Penelitian")
    st.write("Apakah penerapan teknik oversampling dengan metode Adaptive Synthetic Sampling (ADASYN) membuat performa model klasifikasi menggunakan Extreme Gradient Boosting (XGBOOST) lebih optimal.")
        
    st.subheader("1.3 Tujuan Penelitian")
    st.write("Berdasarkan penjabaran latar belakang dan permasalahan yang telah dicantumkan, tujuan dari penelitian untuk melakukan klasifikasi penyakit Diabetes Melitus dengan algoritma Extreme Gradient Boosting (XGBOOST) menggunakan teknik oversampling Adaptive Synthetic Sampling (ADASYN) agar mendapatkan kinerja performa terbaik.")
    
    st.subheader("1.4 Alur Penelitian")
    st.write("Alur penelitian akan menunjukan tahapan - tahapan yang akan diimplementasikan pada penelitian.")
    st.image("asset/Alur Penelitian Skripsi.png", caption="Gambar 1. Alur Penelitian")
    
    st.subheader("1.4.1 Pengumpulan Data")
    st.write("Terdapat 2 jenis data yang memiliki fungsi masing – masing. Data pertama adalah data yang akan berfungsi sebagai data yang membangun model klasifikasi. Data ini berisi kumpulan pasien penyakit diabetes yang diperoleh dari platform Kaggle dengan tautan https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset.")
    data = pd.read_csv("diabetes_prediction_dataset.csv")
    st.write(data)
    st.write("Data yang berfungsi sebagai data penguji dari model yang telah dilatih adalah data pasien penyakit Diabetes Melitus pada Puskesmas, Bandaran, Kabupaten Pamekasan.")
    data_puskesmas = pd.read_csv("data_puskesmas.csv")
    st.write(data_puskesmas)
    
    st.subheader("1.4.2 Cleaning Data & Penyesuaian Fitur")
    st.write("Hasil perhitungan missing value dan data duplikat yang ditunjukan pada Gambar 4.3 yang diperoleh pada data public tidak ditemukan adanya missing value. Namun, program mendeteksi adanya data duplikat sebanyak 3.854 record.")
    cleaning_data = pd.DataFrame(
        [
            {"Deteksi": "Missing Value", "Jumlah": 0},
            {"Deteksi": "Duplikat Data", "Jumlah": 3854}
        ]
    )
    st.dataframe(cleaning_data)
    st.write("Terdapat penyesuaian fitur antara data Kaggle dan data Puskesmas untuk menghindari kesalahan prediksi dari model akibat ketidaksesuaian atribut di dalamnya. Fitur yang digunakan telah disesuaikan dengan data sebenarnya yaitu gender, age, HbA1c_level, dan blood_glucose_level")    
    
    st.subheader("1.4.3 Transformasi Data")
    st.write("Atribut dengan kondisi tipe data numerik akan mempermudah model untuk melakukan perhitungan klasifikasi.")
    gender_transfor = pd.DataFrame(
        [
            {"Gender": "Female", "Transformasi": "0"},
            {"Gender": "Male", "Transformasi": "1"}
        ]
    )
    st.dataframe(gender_transfor)
    
    smoking_transfor = pd.DataFrame(
        [
            {"Smoking_history": "No Info", "Transformasi": "0"},
            {"Smoking_history": "current", "Transformasi": "1"},
            {"Smoking_history": "ever", "Transformasi": "2"},
            {"Smoking_history": "former", "Transformasi": "3"},
            {"Smoking_history": "never", "Transformasi": "4"},
            {"Smoking_history": "not current", "Transformasi": "5"},
        ]
    )
    st.dataframe(smoking_transfor)
    
    st.subheader("1.4.4 Rekayasa Atribut")
    st.write("Dalam upaya meningkatkan performa model klasifikasi, penelitian ini menerapkan teknik rekayasa fitur menggunakan metode polynomial features. Rekayasa fitur bertujuan menciptakan atribut baru yang bersifat turunan atau interaksi dari fitur-fitur numerik awal, sehingga mampu mengungkapkan pola non-linear yang mungkin tidak tertangkap oleh fitur asli.")    
    st.write("Berikut merupakan contoh hasil penambahan rekayasa fitur pada Data Puskesmas")
    fitur_polynomial = pd.DataFrame(
        [
            {"gender": 0, "age": 29, "hypertension": 0, "heart_disease": 0, "smoking_history": 4, "bmi": 3915.0, "HbA1c_level": 9.7, "blood_glucose_level": 135},
            {"gender": 0, "age": 27, "hypertension": 1, "heart_disease": 0, "smoking_history": 0, "bmi": 3942.0, "HbA1c_level": 6.4, "blood_glucose_level": 146},
            {"gender": 0, "age": 28, "hypertension": 0, "heart_disease": 0, "smoking_history": 4, "bmi": 5236.0, "HbA1c_level": 9.1, "blood_glucose_level": 187}
        ]
    )
    st.dataframe(fitur_polynomial)
    
    st.subheader("1.4.5 Balancing Data")
    st.write("Kendala yang ditemukan pada data Kaggle sebagai data yang akan membangun model klasifikasi terlatih adalah adanya ketidakseimbangan target atau kelas antara kelas mayoritas dan minoritas sehingga diterapkan teknik oversampling dengan metode Adaptive Synthetic Sampling (ADASYN) untuk mengatasi ketidakseimbangan kelas pada data.")
    st.write("Berikut merupakan grafik yang menampilkan perbandingan jumlah kelas sebelum dan setelah penerapan ADASYN")
    st.image("asset/balancing_diagram.png", caption="Gambar 2. Diagram Perbandingan Kelas Pada Data Sebelum & Setelah ADASYN")
    
    st.subheader("1.4.6 Normalisasi Data")
    st.write("Penerapan normalisasi untuk meningkatkan kontribusi dari skala fitur dengan tipe data numerik yang distandarisasi dalam proses kinerja model. Hal ini menghindari adanya skala fitur yang berbanding jauh karena dapat menyebabkan bias terhadap fitur dengan nilai yang cukup tinggi.")
    before_normalize = pd.DataFrame(
        [
            {"gender": 0, "age": 29, "hypertension": 0, "heart_disease": 0, "smoking_history": 4, "bmi": 3915.0, "HbA1c_level": 9.7, "blood_glucose_level": 135},
            {"gender": 0, "age": 27, "hypertension": 1, "heart_disease": 0, "smoking_history": 0, "bmi": 3942.0, "HbA1c_level": 6.4, "blood_glucose_level": 146},
            {"gender": 0, "age": 28, "hypertension": 0, "heart_disease": 0, "smoking_history": 4, "bmi": 5236.0, "HbA1c_level": 9.1, "blood_glucose_level": 187}
        ]
    )
    st.dataframe(before_normalize)
    st.write("Berikut merupakan contoh data setelah dilakukan normalisasi")
    after_normalize = pd.DataFrame(
        [
            {"gender": -0.72, "age": 1.43, "hypertension": -0.28, "heart_disease": 5.16, "smoking_history": 0.94, "bmi": -0.543667, "HbA1c_level": 0.496, "blood_glucose_level": -0.123},
            {"gender": -0.72, "age": 0.21, "hypertension": -0.28, "heart_disease": -0.19, "smoking_history": -1.33, "bmi": -0.250808, "HbA1c_level": 0.496, "blood_glucose_level": -1.873},
            {"gender": 1.3, "age": -1.00, "hypertension": -0.28, "heart_disease": -0.19, "smoking_history": 0.94, "bmi": -0.250808, "HbA1c_level": -3.110, "blood_glucose_level": 0.401}
        ]
    )
    st.dataframe(after_normalize)
    
    st.subheader("1.4.7 Pembagian Data")
    st.write("Data penelitian akan dipisah ke dalam dua bagian, yaitu data training dan data testing. Presentase pembagian dari data training dan data testing akan menerapkan 3 jenis rasio pembagian yang berbeda mulai dari 90:10, 80:20, dan 70:30 dengan membagi jumlah kelas yang telah seimbang.")
    st.image("asset/split_90%.png", caption="Gambar 3. Pembagian Data Dengan Rasio 90:10")
    st.image("asset/split_80%.png", caption="Gambar 4. Pembagian Data Dengan Rasio 80:20")
    st.image("asset/split_70%.png", caption="Gambar 5. Pembagian Data Dengan Rasio 70:30")
    
    st.subheader("1.4.8 Hasil Akhir Skenario Pengujian")
    st.write("Pada proses klasifikasi menggunakan Extreme Gradient Boosting (XGBOOST) terdapat 2 skenario yang dilakukan uji coba lalu dibandingkan satu sama lain. Skenario pertama dengan menerapkan metode XGBOOST untuk klasifikasi diabetes tanpa disertai teknik oversampling menggunakan Adaptive Synthetic Sampling (ADASYN) sebagai penyeimbang data. Sementara skenario kedua akan menerapkan ADASYN pada klasifikasi diabetes menggunakan model XGBOOST. Selain itu, di setiap skenario terdapat 3 jenis pembagian data training dan data testing yang digunakan yaitu 90%, 80%, dan 70%. Pembagian rasio ini bertujuan untuk melihat kinerja model pada rasio pembagian data yang berbeda.")
    classification_table = pd.DataFrame(
        [
            {"Skenario": "XGBOOST", "Data": "Data Kaggle 8 Fitur + Data Puskesmas 8 Fitur", "Rasio Split": "90:10", "Akurasi": "0.7516", "Presisi": "1.0000", "Recall": "0.1720", "F1-score": "0.2936"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle 8 Fitur + Data Puskesmas 8 Fitur", "Rasio Split": "80:20", "Akurasi": "0.7032", "Presisi": "1.0000", "Recall": "0.0108", "F1-score": "0.0213"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle 8 Fitur + Data Puskesmas 8 Fitur", "Rasio Split": "70:30", "Akurasi": "0.7032", "Presisi": "1.0000", "Recall": "0.0108", "F1-score": "0.0213"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle 8 Fitur + Data Puskesmas 4 Fitur", "Rasio Split": "90:10", "Akurasi": "0.7871", "Presisi": "0.6134", "Recall": "0.7849", "F1-score": "0.6887"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle 8 Fitur + Data Puskesmas 4 Fitur", "Rasio Split": "80:20", "Akurasi": "0.7903", "Presisi": "0.6228", "Recall": "0.7634", "F1-score": "0.6860"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle 8 Fitur + Data Puskesmas 4 Fitur", "Rasio Split": "70:30", "Akurasi": "0.7000", "Presisi": "0.5000", "Recall": "0.7527", "F1-score": "0.6009"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle 4 Fitur + Data Puskesmas 4 Fitur", "Rasio Split": "90:10", "Akurasi": "0.7581", "Presisi": "0.5738", "Recall": "0.7527", "F1-score": "0.6512"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle 4 Fitur + Data Puskesmas 4 Fitur", "Rasio Split": "80:20", "Akurasi": "0.8032", "Presisi": "0.6481", "Recall": "0.7527", "F1-score": "0.6925"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle 4 Fitur + Data Puskesmas 4 Fitur", "Rasio Split": "70:30", "Akurasi": "0.7806", "Presisi": "0.6087", "Recall": "0.7527", "F1-score": "0.6731"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle 8 Fitur + Data Puskesmas 8 Fitur", "Rasio Split": "90:10", "Akurasi": "0.7129", "Presisi": "1.0000", "Recall": "0.0430", "F1-score": "0.0825"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle 8 Fitur + Data Puskesmas 8 Fitur", "Rasio Split": "80:20", "Akurasi": "0.7000", "Presisi": "0.0000", "Recall": "0.0000", "F1-score": "0.0000"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle 8 Fitur + Data Puskesmas 8 Fitur", "Rasio Split": "70:30", "Akurasi": "0.7000", "Presisi": "0.0000", "Recall": "0.0000", "F1-score": "0.0000"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle 8 Fitur + Data Puskesmas 4 Fitur", "Rasio Split": "90:10", "Akurasi": "0.6871", "Presisi": "0.4867", "Recall": "0.7849", "F1-score": "0.6008"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle 8 Fitur + Data Puskesmas 4 Fitur", "Rasio Split": "80:20", "Akurasi": "0.6871", "Presisi": "0.4867", "Recall": "0.7849", "F1-score": "0.6008"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle 8 Fitur + Data Puskesmas 4 Fitur", "Rasio Split": "70:30", "Akurasi": "0.6806", "Presisi": "0.4803", "Recall": "0.7849", "F1-score": "0.5959"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle 4 Fitur + Data Puskesmas 4 Fitur", "Rasio Split": "90:10", "Akurasi": "0.9065", "Presisi": "0.8810", "Recall": "0.7957", "F1-score": "0.8362"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle 4 Fitur + Data Puskesmas 4 Fitur", "Rasio Split": "80:20", "Akurasi": "0.8129", "Presisi": "0.6549", "Recall": "0.7957", "F1-score": "0.7184"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle 4 Fitur + Data Puskesmas 4 Fitur", "Rasio Split": "70:30", "Akurasi": "0.6871", "Presisi": "0.4867", "Recall": "0.7849", "F1-score": "0.6008"},
        ]
    )
    st.dataframe(classification_table)
    
    st.subheader("1.4.8 Kesimpulan")
    st.write("Berdasarkan serangkaian skenario pengujian yang telah dilakukan penerapan metode Adaptive Synthetic Sampling (ADASYN) pada klasifikasi diabetes menggunakan Extreme Gradient Boosting (XGBOOST) terbukti memberikan dampak yang signifikan dalam meningkatkan performa model khususnya dalam menghadapi ketidakseimbangan kelas pada data. Seluruh skenario dalam penelitian ini telah menguji berbagai kombinasi rasio data latih dan uji 90:10, 80:20, dan 70:30 serta berbagai kombinasi fitur 8 fitur dengan penambahan 4 fitur polynomial pada masing – masing data, 8 fitur utama pada data Kaggle dan 4 fitur sepadan pada data Puskesmas yang ditambahkan 4 fitur polynomial, dan 4 fitur sepadan dari masing – masing data. Dari hasil yang diperoleh performa terbaik dicapai pada skenario dengan rasio data 90:10 dan kombinasi fitur 4x4 yaitu 4 fitur dari data Kaggle dan 4 dari datas Puskesmas. Pada skenario tersebut, model berhasil mencapai akurasi sebesar 0.9065, presisi sebesar 0.8810, recall 0.7957, dan F1-score sebesar 0.8362. Nilai-nilai tersebut menunjukkan bahwa model mampu mengidentifikasi kelas positif dengan sangat baik sekaligus mempertahankan keseimbangan antara kemampuan deteksi dan minimnya kesalahan klasifikasi yang dibuktikan adanya peningkatan nilai recall yang semula sangat rendah menjadi lebih seimbang terhadap presisi serta menghasilkan F1-score yang lebih tinggi. Selain itu, pengaruh rasio data latih dan uji juga terbukti berperan penting dalam kinerja model. Rasio 90:10 memungkinkan model memperoleh lebih banyak data untuk pembelajaran, yang berdampak langsung terhadap peningkatan performa, tanpa menyebabkan overfitting.")
    
elif page == "Proses Klasifikasi Data":
    # Load Data
    st.subheader("📊 Input Data Kaggle")
    st.write("Program dibawah merupakan kode untuk melakukan load data dari python.")
    code = ''' data = pd.read_csv('/content/diabetes_prediction_dataset.csv') '''
    st.code(code, language="python")
    st.write("Output:")
    
    data = pd.read_csv("diabetes_prediction_dataset.csv")
    st.write(data)
    
    # Missing Value
    st.subheader("🔍 Cek Missing Value")
    st.write("Program dibawah merupakan kode untuk mendeteksi missing value pada setiap atribut.")
    code = ''' data.isnull().sum() '''
    st.code(code, language="python")
    st.write("Output:")
    
    st.write(data.isnull().sum())
    
    # Cleaning Data
    st.subheader("📛 Menghapus Data Duplikat")
    st.write("Program dibawah merupakan kode untuk mendeteksi dan menghapus adanya data duplikat.")
    
    initial_rows = data.shape[0]
    data = data.drop_duplicates()
    duplicates_removed = initial_rows - data.shape[0]
    
    code = ''' 
    print("Jumlah Data Duplikat :" ,data.duplicated().sum())
    data = data.drop_duplicates()
    '''
    st.code(code, language="python")
    st.write("Output:")
    
    st.write(f"Jumlah Data Duplikat : {duplicates_removed}")
    
    # Transformasi Data
    st.subheader("🎭 Transformasi Data")
    st.write("Program dibawah merupakan kode untuk mengubah atribut dengan tipe data kategorikal menjadi numerik.")
    
    label_encoding_gender = LabelEncoder()
    data['gender'] = label_encoding_gender.fit_transform(data['gender'])
    
    label_encoding_smoking = LabelEncoder()
    data['smoking_history'] = label_encoding_smoking.fit_transform(data['smoking_history'])
    
    code = ''' 
    label_encoding = LabelEncoder()
    data['gender'] = label_encoding.fit_transform(data['gender'])
    
    label_encoding_smoking = LabelEncoder()
    data['smoking_history'] = label_encoding_smoking.fit_transform(data['smoking_history'])
    '''
    st.code(code, language="python")
    st.write("Output:")
    
    st.write(data)
    
    # Balancing Data
    st.subheader("⚖️ Balancing Data dengan ADASYN")
    st.write("Program dibawah merupakan kode untuk menyeimbangkan data dengan menerapkan teknik Oversampling ADASYN.")
    
    X = data.drop(columns=['diabetes'])
    y = data['diabetes']
    
    data_balancing = pd.read_csv("data_balancing.csv")
    X_balance = data_balancing.drop(columns=['diabetes'])
    y_balance = data_balancing['diabetes']
    
    code = ''' 
    X = data.drop(columns=['diabetes'])
    y = data['diabetes']
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    '''
    st.code(code, language="python")
    st.write("Output:")
    
    st.write("Sebelum Dilakukan Balancing")
    st.dataframe(pd.Series(y).value_counts().rename_axis('Kelas').reset_index(name='Jumlah'))
    st.write("Setelah Dilakukan Balancing")
    st.dataframe(pd.Series(y_balance).value_counts().rename_axis('Kelas').reset_index(name='Jumlah'))
    
    # Normalisasi Data
    st.subheader("📐 Normalisasi Data dengan Z-Score")
    st.write("Program dibawah merupakan kode untuk melakukan normalisasi data menggunakan Z-Score.")
    
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_balance)
    X_normalized_df = pd.DataFrame(X_normalized, columns=X_balance.columns)
    
    code = ''' 
    scaler = StandardScaler()
    X_train_normalized = pd.DataFrame(scaler.fit_transform(X_resampled), index=X_resampled.index, columns=X_resampled.columns)
    '''
    st.code(code, language="python")
    st.write("Output:")
    
    st.write(X_normalized_df)
    
    # Split Data
    st.subheader("✂️ Pembagian Data")
    st.write("Program dibawah merupakan kode untuk membagi data menjadi data training sebesar 80% dan data training 20%.")
    
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_balance, test_size=0.2, random_state=42)
    
    code = ''' 
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_balance, test_size=0.2, random_state=42)
    '''
    st.code(code, language="python")
    st.write("Output:")
    
    st.write("Jumlah Data Training :", X_train.shape)
    st.write("Jumlah Data Testing :", X_test.shape)
    
    # Tuning Hyperparameter
    st.subheader("🧪 Hyperparameter Tuning dengan Bayesian Optimization")
    st.write("Program dibawah merupakan kode untuk melakukan pencarian nilai parameter terbaik menggunakan pendekatan Bayesian Optimization.")
    
    code = ''' 
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 250),
            'max_depth': trial.suggest_int('max_depth', 3, 11),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
            'gamma': trial.suggest_float('gamma', 1e-9, 0.5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
        }
        model = xgb.XGBClassifier(**params)
        f1 = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(f1_score), n_jobs=-1)
        return f1.mean()
    
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    '''
    st.code(code, language="python")
    st.write("Output:")
    
    st.write("Kombinasi Nilai Parameter Terbaik :")
    st.write(" 'n_estimators': 185")
    st.write(" 'max_depth': 11")
    st.write(" 'learning_rate': 0.19720799914920925")
    st.write(" 'colsample_bytree': 0.6785791475774542")
    st.write(" 'gamma': 0.08733666463315318")
    st.write(" 'min_child_weight': 1")
    
    # Modeling
    st.subheader("📈 Modeling dengan Metode XGBOOST")
    st.write("Program dibawah merupakan kode untuk melakukan klasifikasi dengan metode XGBOOST menggunakan nilai parameter terbaik serta menampilkan evaluasi hasil.")
    
    code = ''' 
    final_model = xgb.XGBClassifier(**study.best_params, random_state=42, eval_metric='logloss')
    final_model.fit(X_train, y_train)
    y_pred_test = final_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    rec = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    '''
    
    table_results = pd.DataFrame(
        [
            {"Akurasi": 0.9720, "Presisi": 0.9815, "Recall": 0.9623, "F1-score": 0.9718}
        ]
    )
    st.code(code, language="python")
    st.write("Output:")
    
    st.write(table_results, use_container_width=True)
    
    # Penerapan Model ke Data Puskesmas
    st.subheader("📊 Input Data Puskesmas")
    st.write("Program dibawah merupakan kode untuk melakukan load data dari python.")
    code = ''' data = pd.read_csv('/content/data_puskesmas.csv') '''
    st.code(code, language="python")
    st.write("Output:")
    
    data_puskesmas = pd.read_csv("data_puskesmas.csv")
    df_puskesmas = data_puskesmas.drop(columns=['no','nama','HCT','RBC','WBC','PLT'])
    st.write(df_puskesmas)
    
    st.subheader("🧬 Rekayasa Fitur dengan Polynomial Feature")
    st.write("Program dibawah merupakan kode untuk melakukan penambahan fitur pada data menggunakan turunan dari fitur yang telah tersedia.")
    
    code = ''' 
    X_original_puskes = df_puskes[['gender', 'age', 'HbA1c_level', 'blood_glucose_level']]
    y_puskes = df_puskes['diabetes'] 
    
    poly_puskes = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_all_puskes = poly_puskes.fit_transform(X_original_puskes[['age', 'HbA1c_level', 'blood_glucose_level']])
    poly_feature_names_puskes = poly_puskes.get_feature_names_out(['age', 'HbA1c_level', 'blood_glucose_level'])
    
    df_poly_puskes = pd.DataFrame(X_poly_all_puskes, columns=poly_feature_names_puskes)
    
    original_feature_names_puskes = ['age', 'HbA1c_level', 'blood_glucose_level']
    interaction_feature_names_puskes = [f for f in poly_feature_names_puskes if f not in original_feature_names_puskes]
    X_interaction_puskes = df_poly_puskes[interaction_feature_names_puskes]
    selector = SelectKBest(score_func=f_classif, k=1)
    X_selected_interactions_puskes = selector.fit_transform(X_interaction_puskes, y_puskes)
    selected_interaction_names_puskes = X_interaction_puskes.columns[selector.get_support()]
    '''
    st.code(code, language="python")
    st.write("Output:")
    
    data_rekayasa = pd.read_csv("data_rekayasa.csv")
    st.write(data_rekayasa)
    
    st.subheader("🔮 Klasifikasi Pada Data Puskesmas menggunakan Model Terlatih")
    st.write("Program dibawah merupakan kode untuk melakukan klasifikasi pada data puskesmas menggunakan model XGBOOST data kaggle serta menampilkan evaluasi hasil.")
    
    code = ''' 
    model_puskes = joblib.load('xgboost_model.pkl')
    y_pred_puskes = model_puskes.predict(X_puskesmas)

    acc = accuracy_score(y_puskesmas, y_pred_puskes)
    prec = precision_score(y_puskesmas, y_pred_puskes)
    rec = recall_score(y_puskesmas, y_pred_puskes)
    f1 = f1_score(y_puskesmas, y_pred_puskes)
    '''
    
    table_results = pd.DataFrame(
        [
            {"Akurasi": 0.6871, "Presisi": 0.4867, "Recall": 0.7849, "F1-score": 0.6008}
        ]
    )
    st.code(code, language="python")
    st.write("Output:")
    
    st.write(table_results, use_container_width=True)

elif page == "Prediksi Baru":
    st.header("🧪 Prediksi Baru Diabetes Menggunakan Model Terlatih")
    st.subheader("📝 Masukkan Data Pasien Baru")
    
    # ==== Load Model dan Scaler ====
    model_xgb = joblib.load('model_artifacts/xgboost_model.pkl')
    le_gender = joblib.load('model_artifacts/gender_encoder.pkl')
    le_smoking = joblib.load('model_artifacts/smoking_history_encoder.pkl')
    scaler = joblib.load('model_artifacts/scaler.pkl')
    
    # ==== Form Input ====
    with st.form("input_form"):
        gender = st.selectbox("Jenis Kelamin", options=["Female", "Male"])
        age = st.number_input("Usia", min_value=0.0, max_value=120.0)
        hypertension = st.selectbox("Hipertensi", options=[0, 1])
        heart_disease = st.selectbox("Penyakit Jantung", options=[0, 1])
        smoking_history = st.selectbox("Riwayat Merokok", options=["No Info", "current", "ever", "former", "never", "not current"])
        bmi = st.number_input("BMI", min_value=0.0, step=0.1)
        hba1c = st.number_input("HbA1c Level", min_value=0.0, step=0.1)
        glucose = st.number_input("Blood Glucose Level", min_value=0, step=1)

        gender_map = {"Female": 0, "Male": 1}
        smoking_map = {
            "No Info": 0,
            "current": 1,
            "ever": 2,
            "former": 3,
            "never": 4,
            "not current": 5
        }
    
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        input_data = pd.DataFrame({
            'gender': gender_map[gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': smoking_map[smoking_history],
            'bmi': [bmi],
            'HbA1c_level': [hba1c],
            'blood_glucose_level': [glucose]
        })
        
        st.write(input_data)

        # Normalisasi
        input_scaled = scaler.transform(input_data)
        st.write(input_scaled)

        # Prediksi
        prediction = model_xgb.predict(input_scaled)[0]
        label = "Positif Diabetes" if prediction == 1 else "Negatif Diabetes"

        st.success(f"Hasil Prediksi: **{label}**")
    
    
    # with st.form("form_prediksi"):
    #     gender = st.selectbox("Gender", ["Female", "Male"])
    #     age = st.number_input("Age", min_value=0, max_value=120)
    #     hypertension = st.selectbox("Hypertension", ["0", "1"])
    #     heart_disease = st.selectbox("Heart Disease", ["0", "1"])
    #     smoking_history = st.selectbox("Smoking History", ["No Info", "current", "ever", "former", "never", "not current"])
    #     bmi = st.number_input("BMI", min_value=0.0, max_value=100.0)
    #     hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0)
    #     glucose = st.number_input("Blood Glucose Level", min_value=0.0, max_value=300.0)

    #     # Mapping ke bentuk numerik sesuai LabelEncoder
    #     gender_map = {"Female": 0, "Male": 1}
    #     smoking_map = {
    #         "No Info": 0,
    #         "current": 1,
    #         "ever": 2,
    #         "former": 3,
    #         "never": 4,
    #         "not current": 5
    #     }

    #     submitted = st.form_submit_button("Prediksi")
        
    #     if submitted:
    #         input_data = {
    #             "gender": gender_map[gender],
    #             "age": age,
    #             "hypertension": hypertension,
    #             "heart_disease": heart_disease,
    #             "smoking_history": smoking_map[smoking_history],
    #             "bmi": bmi,
    #             "HbA1c_level": hba1c,
    #             "blood_glucose_level": glucose
    #         }
            
    #         columns_order = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    #         input_df = pd.DataFrame([[input_data[col] for col in columns_order]], columns=columns_order)

    #         # Normalisasi fitur
    #         scaler = joblib.load("model_artifacts/scaler.pkl")
    #         input_scaled = scaler.transform(input_df)

    #         # Prediksi
    #         model = joblib.load('model_artifacts/xgboost_model.pkl')
    #         prediction = model.predict(input_scaled)[0]
    #         probability = model.predict_proba(input_scaled)[0][1]

    #         # Tampilkan hasil
    #         if prediction == 1:
    #             st.error(f"Hasil Prediksi: Positif Diabetes (Probabilitas: {probability:.2f})")
    #         else:
    #             st.success(f"Hasil Prediksi: Negatif Diabetes (Probabilitas: {probability:.2f})")