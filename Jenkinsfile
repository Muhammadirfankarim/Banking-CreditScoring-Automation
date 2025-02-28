pipeline {
    agent any
    
    stages {
        stage('Setup Environment') {
            steps {
                echo 'Menyiapkan lingkungan Docker...'
                
                // Membuat direktori untuk hasil
                bat 'mkdir models reports visualizations data predictions templates 2>nul'
            }
        }
        
        stage('Run ML Pipeline in Docker') {
            steps {
                echo 'Menjalankan pipeline dalam container Docker...'
                
                // Jalankan seluruh pipeline dalam container Docker
                bat '''
                    docker run --rm -v %CD%:/app -w /app python:3.9-slim bash -c "
                    pip install --upgrade pip &&
                    pip install pandas numpy matplotlib seaborn scikit-learn flask pytest requests pytest-cov &&
                    
                    # Data Validation
                    python -c \\"
import pandas as pd
import sys

try:
    # Memuat data
    data = pd.read_csv('bank.csv', sep=';')
    
    # Validasi data sederhana
    assert 'y' in data.columns, 'Kolom target tidak ditemukan'
    print('Validasi data sukses!')
except Exception as e:
    print(f'Error saat validasi data: {str(e)}')
    sys.exit(1)
                    \\" &&
                    
                    # EDA
                    python -c \\"
from bank_ml_pipeline import BankMarketingPipeline

# Inisialisasi pipeline
pipeline = BankMarketingPipeline(data_path='bank.csv')

# Melakukan EDA
pipeline.perform_eda()
print('EDA selesai')
                    \\" &&
                    
                    # Preprocessing
                    python -c \\"
from bank_ml_pipeline import BankMarketingPipeline

# Inisialisasi pipeline
pipeline = BankMarketingPipeline(data_path='bank.csv')

# Memuat data
pipeline.load_data()

# Melakukan preprocessing
pipeline.preprocess_data()
print('Preprocessing data selesai')
                    \\" &&
                    
                    # Training
                    python -c \\"
from bank_ml_pipeline import BankMarketingPipeline

# Inisialisasi pipeline
pipeline = BankMarketingPipeline(data_path='bank.csv')

# Memuat data dan preprocess
pipeline.load_data()
pipeline.preprocess_data()

# Melatih model
best_model, metrics = pipeline.train_models()
print(f'Model terbaik: {metrics[\\\"best_model\\\"]}')
                    \\"
                    "
                '''
            }
        }
        
        stage('Deploy Web Server') {
            steps {
                echo 'Men-deploy webserver dalam container Docker...'
                
                bat '''
                    docker run -d --name flask-bank-app -p 5000:5000 -v %CD%:/app -w /app python:3.9-slim bash -c "
                    pip install --upgrade pip &&
                    pip install pandas numpy matplotlib seaborn scikit-learn flask requests &&
                    python app.py
                    "
                '''
                
                echo 'Menunggu server dimulai...'
                bat 'ping -n 10 127.0.0.1 > nul'
            }
        }
        
        stage('Functional Testing') {
            steps {
                echo 'Menjalankan pengujian fungsional...'
                
                bat '''
                    docker run --rm --network host -v %CD%:/app -w /app python:3.9-slim bash -c "
                    pip install --upgrade pip &&
                    pip install requests &&
                    python -c \\"
import requests
import sys
import json
import time

# Base URL API 
base_url = 'http://localhost:5000'

# Tunggu server siap
max_retries = 5
retry_delay = 2
success = False

for i in range(max_retries):
    try:
        response = requests.get(f'{base_url}/')
        if response.status_code == 200:
            success = True
            break
    except Exception as e:
        print(f'Error: {str(e)}')
        print(f'Server belum siap, mencoba lagi dalam {retry_delay} detik...')
        time.sleep(retry_delay)

if not success:
    print('Server tidak dapat dijangkau setelah beberapa percobaan')
    sys.exit(1)

try:
    # Test 1: Mendapatkan informasi model
    response = requests.get(f'{base_url}/model-info')
    assert response.status_code == 200, 'Endpoint model-info tidak berfungsi'
    
    # Test 2: Melakukan prediksi
    test_data = {
        'age': 35,
        'job': 'management',
        'marital': 'married',
        'education': 'tertiary',
        'default': 'no',
        'balance': 1500,
        'housing': 'yes',
        'loan': 'no',
        'contact': 'cellular',
        'day': 15,
        'month': 'may',
        'duration': 180,
        'campaign': 1,
        'pdays': -1,
        'previous': 0,
        'poutcome': 'unknown'
    }
    
    response = requests.post(f'{base_url}/predict', json=test_data)
    assert response.status_code == 200, 'Endpoint predict tidak berfungsi'
    
    # Menulis hasil tes ke file
    with open('functional_test_results.json', 'w') as f:
        json.dump({
            'status': 'success',
            'message': 'Pengujian fungsional berhasil'
        }, f, indent=4)
    
    print('Pengujian fungsional berhasil!')
    sys.exit(0)
except Exception as e:
    with open('functional_test_results.json', 'w') as f:
        json.dump({
            'status': 'error',
            'message': f'Pengujian fungsional gagal: {str(e)}'
        }, f, indent=4)
    
    print(f'Pengujian fungsional gagal: {str(e)}')
    sys.exit(1)
                    \\"
                    "
                '''
                
                // Mengarsipkan hasil pengujian
                archiveArtifacts artifacts: 'functional_test_results.json, models/*.pkl, models/*.json, reports/*.json, reports/*.txt, visualizations/*.png', fingerprint: true
            }
        }
    }
    
    post {
        always {
            echo 'Membersihkan container...'
            bat 'docker stop flask-bank-app || echo "Container tidak berjalan"'
            bat 'docker rm flask-bank-app || echo "Container tidak ada"'
        }
        success {
            echo 'Pipeline machine learning berhasil dijalankan!'
        }
        failure {
            echo 'Pipeline machine learning gagal dijalankan!'
        }
    }
}
