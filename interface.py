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
    transformation_data = pd.DataFrame(
        [
            {"Data": "Female", "Transformasi": "0"},
            {"Data": "Male", "Transformasi": "1"}
        ]
    )
    st.dataframe(transformation_data)
    
    st.subheader("1.4.4 Balancing Data")
    st.write("Kendala yang ditemukan pada data Kaggle sebagai data yang akan membangun model klasifikasi terlatih adalah adanya ketidakseimbangan target atau kelas antara kelas mayoritas dan minoritas sehingga diterapkan teknik oversampling dengan metode Adaptive Synthetic Sampling (ADASYN) untuk mengatasi ketidakseimbangan kelas pada data.")
    st.image("asset/balancing_diagram.png", caption="Gambar 2. Diagram Perbandingan Kelas Pada Data Sebelum & Setelah ADASYN")
    
    st.subheader("1.4.5 Normalisasi Data")
    st.write("Penerapan normalisasi untuk meningkatkan kontribusi dari skala fitur dengan tipe data numerik yang distandarisasi dalam proses kinerja model. Hal ini menghindari adanya skala fitur yang berbanding jauh karena dapat menyebabkan bias terhadap fitur dengan nilai yang cukup tinggi.")
    before_normalize = pd.DataFrame(
        [
            {"gender": 0, "age": 80, "HbA1c_level": 6.6, "blood_glucose_level": 140},
            {"gender": 0, "age": 54, "HbA1c_level": 6.6, "blood_glucose_level": 80},
            {"gender": 1, "age": 28, "HbA1c_level": 5.7, "blood_glucose_level": 158}
        ]
    )
    st.dataframe(before_normalize)
    st.write("Berikut merupakan contoh data setelah dilakukan normalisasi")
    after_normalize = pd.DataFrame(
        [
            {"gender": -0.83, "age": 1.42, "HbA1c_level": 0.704, "blood_glucose_level": -0.123},
            {"gender": -0.83, "age": 0.21, "HbA1c_level": 0.704, "blood_glucose_level": -1.877},
            {"gender": 1.20, "age": -1.00, "HbA1c_level": -0.224, "blood_glucose_level": 0.402}
        ]
    )
    st.dataframe(after_normalize)
    
    st.subheader("1.4.6 Pembagian Data")
    st.write("Data penelitian akan dipisah ke dalam dua bagian, yaitu data training dan data testing. Presentase pembagian dari data training dan data testing akan menerapkan 3 jenis rasio pembagian yang berbeda mulai dari 90:10, 80:20, dan 70:30 dengan membagi jumlah kelas yang telah seimbang.")
    st.image("asset/split_90%.png", caption="Gambar 3. Pembagian Data Dengan Rasio 90:10")
    st.image("asset/split_80%.png", caption="Gambar 4. Pembagian Data Dengan Rasio 80:20")
    st.image("asset/split_70%.png", caption="Gambar 5. Pembagian Data Dengan Rasio 70:30")
    
    st.subheader("1.4.7 Hasil Akhir Skenario Pengujian")
    st.write("Pada proses klasifikasi menggunakan Extreme Gradient Boosting (XGBOOST) terdapat 2 skenario yang dilakukan uji coba lalu dibandingkan satu sama lain. Skenario pertama dengan menerapkan metode XGBOOST untuk klasifikasi diabetes tanpa disertai teknik oversampling menggunakan Adaptive Synthetic Sampling (ADASYN) sebagai penyeimbang data. Sementara skenario kedua akan menerapkan ADASYN pada klasifikasi diabetes menggunakan model XGBOOST. Selain itu, di setiap skenario terdapat 3 jenis pembagian data training dan data testing yang digunakan yaitu 90%, 80%, dan 70%. Pembagian rasio ini bertujuan untuk melihat kinerja model pada rasio pembagian data yang berbeda.")
    classification_table = pd.DataFrame(
        [
            {"Skenario": "XGBOOST", "Data": "Data Kaggle", "Rasio Split": "90:10", "Akurasi": "0.9706", "Presisi": "1.0000", "Recall": "0.6736", "F1-score": "0.8050"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle", "Rasio Split": "80:20", "Akurasi": "0.9706", "Presisi": "1.0000", "Recall": "0.6711", "F1-score": "0.8032"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle", "Rasio Split": "70:30", "Akurasi": "0.9711", "Presisi": "1.0000", "Recall": "0.6760", "F1-score": "0.8067"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle + Data Puskesmas", "Rasio Split": "90:10", "Akurasi": "0.7581", "Presisi": "0.5738", "Recall": "0.7527", "F1-score": "0.6512"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle + Data Puskesmas", "Rasio Split": "80:20", "Akurasi": "0.8032", "Presisi": "0.6481", "Recall": "0.7527", "F1-score": "0.6965"},
            {"Skenario": "XGBOOST", "Data": "Data Kaggle + Data Puskesmas", "Rasio Split": "70:30", "Akurasi": "0.7806", "Presisi": "0.6087", "Recall": "0.7527", "F1-score": "0.6731"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle", "Rasio Split": "90:10", "Akurasi": "0.9411", "Presisi": "0.9334", "Recall": "0.9505", "F1-score": "0.9418"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle", "Rasio Split": "80:20", "Akurasi": "0.9334", "Presisi": "0.9271", "Recall": "0.9414", "F1-score": "0.9342"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle", "Rasio Split": "70:30", "Akurasi": "0.9412", "Presisi": "0.9368", "Recall": "0.9471", "F1-score": "0.9419"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle + Data Puskesmas", "Rasio Split": "90:10", "Akurasi": "0.9263", "Presisi": "0.9512", "Recall": "0.8986", "F1-score": "0.9242"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle + Data Puskesmas", "Rasio Split": "80:20", "Akurasi": "0.8594", "Presisi": "0.8333", "Recall": "0.8986", "F1-score": "0.8647"},
            {"Skenario": "ADASYN + XGBOOST", "Data": "Data Kaggle + Data Puskesmas", "Rasio Split": "70:30", "Akurasi": "0.7719", "Presisi": "0.7169", "Recall": "0.8986", "F1-score": "0.7975"},
        ]
    )
    st.dataframe(classification_table)
    
    st.subheader("1.4.8 Kesimpulan")
    st.write("Melalui penelitian ini, telah dilakukan upaya untuk meningkatkan performa model Extreme Gradient Boosting (XGBOOST) yang dilatih menggunakan data public yang bersumber dari platform kaggle dengan menerapkan teknik oversampling Adaptive Synthetic Sampling (ADASYN) dan diterapkan ke data pasien diabetes Puskemas Bandaran Pamekasan. Model data kaggle menggunakan ADASYN dengan rasio 80:20 menunjukan performa terbaik dengan nilai f1-score tertinggi serta presisi dan recall yang seimbang sehingga model lebih optimal dalam rasio ini. Sedangkan pada data puskesmas menggunakan ADASYN dengan rasio 90:10 menjadi model dengan performa terbaik karena capaian akurasi 0.9263, presisi 0.9512, recall 0.8916, dan f1-score 0.9242 yang seimbang. Dengan demikian, penerapan teknik oversampling Adaptive Synthetic Sampling (ADASYN) untuk penyeimbang data yang akan dilakukan klasifikasi dengan model Extreme Gradient Boosting (XGBOOST) memiliki pengaruh sebagai pendekatan yang efektif dalam meningkatkan performa model, khususnya pada situasi data yang tidak seimbang. Kombinasi keduanya membuat model dapat memanfaatkan data training lebih efisien.")
    
elif page == "Proses Klasifikasi Data":
    # Load Data
    st.subheader("📊 Input Data Public")
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
    
    # Menghapus Kolom
    st.subheader("🗂️ Menyesuaikan Atribut Dengan Data Puskesmas")
    st.write("Program dibawah merupakan kode untuk menghapus kolom atribut untuk menyesuaikan atribut yang tersedia pada Data Puskesmas.")
    
    data = data.drop(columns=['hypertension', 'heart_disease', 'smoking_history', 'bmi'])
    
    code = ''' 
    data = data.drop(columns=['hypertension', 'heart_disease', 'smoking_history', 'bmi'])
    '''
    st.code(code, language="python")
    st.write("Output:")
    
    st.write(data)
    
    # Transformasi Data
    st.subheader("🎭 Transformasi Data")
    st.write("Program dibawah merupakan kode untuk mengubah atribut dengan tipe data kategorikal menjadi numerik.")
    
    label_encoding = LabelEncoder()
    data['gender'] = label_encoding.fit_transform(data['gender'])
    
    code = ''' 
    label_encoding = LabelEncoder()
    data['gender'] = label_encoding.fit_transform(data['gender'])
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
    st.write(" 'n_estimators': 108")
    st.write(" 'max_depth': 11")
    st.write(" 'learning_rate': 0.3282004146218894")
    st.write(" 'colsample_bytree': 0.7541613152436578")
    st.write(" 'gamma': 0.09307489905925012")
    st.write(" 'min_child_weight': 3")
    
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
            {"Accuracy": 0.9334, "Precision": 0.9271, "Recall": 0.9414, "F1-score": 0.9342}
        ]
    )
    st.code(code, language="python")
    st.write("Output:")
    
    st.write(table_results, use_container_width=True)
    
    # Penerapan Model ke Data Puskesmas
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
            {"Accuracy": 0.8594, "Precision": 0.8333, "Recall": 0.8986, "F1-score": 0.8647}
        ]
    )
    st.code(code, language="python")
    st.write("Output:")
    
    st.write(table_results, use_container_width=True)

elif page == "Prediksi Baru":
    st.header("🧪 Prediksi Baru Diabetes Menggunakan Model Terlatih")

    # ==== Load Model dan Alat Preprocessing ====
    try:
        model = joblib.load("model_artifacts/xgboost_model.pkl")
        scaler = joblib.load("model_artifacts/scaler.pkl")
        label_encoders = joblib.load("model_artifacts/label_encoders_diabetes.pkl")
    except Exception as e:
        st.error(f"Gagal memuat model atau preprocessing tools: {e}")
    
    # ==== Form Input ====
    st.subheader("📝 Masukkan Data Pasien Baru")

    with st.form("form_prediksi"):
        gender = st.selectbox("Jenis Kelamin", options=["Male", "Female"])
        age = st.number_input("Umur", min_value=0, max_value=120)
        HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, step=0.1)
        blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0, max_value=500)
        
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        try:
            # Membuat DataFrame dari inputan
            input_data = pd.DataFrame([{
                "gender": gender,
                "age": age,
                "HbA1c_level": HbA1c_level,
                "blood_glucose_level": blood_glucose_level
            }])

            st.write("📋 Data yang Dimasukkan:")
            st.dataframe(input_data)

            # ==== Transformasi Kategorikal ====
            if "gender" in label_encoders:
                le = label_encoders["gender"]
                input_data["gender"] = le.transform(input_data["gender"])
            else:
                st.warning("Encoder untuk kolom 'gender' tidak ditemukan.")

            # ==== Normalisasi ====
            input_scaled = scaler.transform(input_data)
            input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)

            # ==== Prediksi ====
            pred_result = model.predict(input_scaled_df)[0]
            pred_prob = model.predict_proba(input_scaled_df)[0]

            label_mapping = {0: "Tidak Diabetes", 1: "Diabetes"}
            st.subheader("🔍 Hasil Prediksi")
            st.write(f"**Prediksi**: {label_mapping[pred_result]}")
            st.write(f"**Probabilitas Positif**: {pred_prob[1]:.2%}")
            st.write(f"**Probabilitas Negatif**: {pred_prob[0]:.2%}")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")