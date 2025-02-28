pipeline {
    agent any
    
    stages {
        stage('Verifikasi Docker') {
            steps {
                echo 'Memeriksa instalasi Docker...'
                
                // Memeriksa versi Docker untuk memastikan tersedia
                bat 'docker --version || exit 1'
                
                // Menampilkan informasi sistem Docker
                bat 'docker info || exit 1'
                
                // Memeriksa apakah image Python bisa di-pull
                bat 'docker pull python:3.9-slim || exit 1'
            }
        }
        
        stage('Setup Environment') {
            steps {
                echo 'Menyiapkan lingkungan untuk pipeline...'
                
                // Membuat direktori untuk hasil
                bat 'if not exist models mkdir models'
                bat 'if not exist reports mkdir reports'
                bat 'if not exist visualizations mkdir visualizations'
                bat 'if not exist data mkdir data'
                bat 'if not exist predictions mkdir predictions'
                bat 'if not exist templates mkdir templates'
                
                // Verifikasi direktori
                bat 'dir'
            }
        }
        
        stage('Data Validation') {
            steps {
                echo 'Memvalidasi data dengan Docker...'
                
                // Jalankan validasi data dalam container Docker dengan path absolut
                bat '''
                    docker run --rm -v "%CD%:/app" -w /app python:3.9-slim python -c "
import pandas as pd
import sys

try:
    # Memuat data
    data = pd.read_csv('bank.csv', sep=';')
    
    # Validasi data sederhana
    assert 'y' in data.columns, 'Kolom target tidak ditemukan'
    print('Validasi data sukses!')
    sys.exit(0)
except Exception as e:
    print(f'Error saat validasi data: {str(e)}')
    sys.exit(1)
"
                '''
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo 'Menginstal dependensi di dalam container...'
                
                bat '''
                    docker run --rm -v "%CD%:/app" -w /app python:3.9-slim pip install pandas numpy matplotlib seaborn scikit-learn flask pytest requests pytest-cov
                '''
            }
        }
        
        stage('EDA & Preprocessing') {
            steps {
                echo 'Menjalankan EDA dan preprocessing...'
                
                bat '''
                    docker run --rm -v "%CD%:/app" -w /app python:3.9-slim python -c "
from bank_ml_pipeline import BankMarketingPipeline

# Inisialisasi pipeline
pipeline = BankMarketingPipeline(data_path='bank.csv')

# Melakukan EDA
print('Memulai EDA...')
pipeline.perform_eda()
print('EDA selesai')

# Preprocessing data
print('Memulai preprocessing data...')
pipeline.preprocess_data()
print('Preprocessing data selesai')
"
                '''
            }
        }
        
        stage('Model Training') {
            steps {
                echo 'Melatih model machine learning...'
                
                bat '''
                    docker run --rm -v "%CD%:/app" -w /app python:3.9-slim python -c "
from bank_ml_pipeline import BankMarketingPipeline

# Inisialisasi pipeline
pipeline = BankMarketingPipeline(data_path='bank.csv')

# Memuat data dan preprocess
pipeline.load_data()
pipeline.preprocess_data()

# Melatih model
print('Melatih model...')
best_model, metrics = pipeline.train_models()
print(f'Model terbaik: {metrics[\"best_model\"]}')
print('Training model selesai.')

# Menyimpan model terbaik
pipeline.save_model()
print('Model disimpan.')
"
                '''
                
                // Mengarsipkan model dan hasil
                archiveArtifacts artifacts: 'models/*.pkl, models/*.json, reports/*.json, reports/*.txt, visualizations/*.png', fingerprint: true
            }
        }
        
        stage('Deploy Web Server') {
            steps {
                echo 'Men-deploy webserver dalam container Docker...'
                
                // Hentikan container yang berjalan jika ada
                bat 'docker stop flask-bank-app 2>nul || echo "Tidak ada container berjalan"'
                bat 'docker rm flask-bank-app 2>nul || echo "Tidak ada container untuk dihapus"'
                
                // Jalankan webserver di container baru
                bat '''
                    docker run -d --name flask-bank-app -p 5000:5000 -v "%CD%:/app" -w /app python:3.9-slim cmd /c "
                    pip install pandas numpy matplotlib seaborn scikit-learn flask requests &&
                    python app.py
                    "
                '''
                
                // Verifikasi container berjalan
                bat 'docker ps | find "flask-bank-app" || echo "Warning: Container tidak berjalan"'
                
                echo 'Menunggu server dimulai...'
                bat 'ping -n 10 127.0.0.1 > nul'
            }
        }
        
        stage('Functional Testing') {
            steps {
                echo 'Menjalankan pengujian fungsional...'
                
                bat '''
                    docker run --rm -v "%CD%:/app" -w /app python:3.9-slim python -c "
import requests
import sys
import json
import time

# Base URL API 
base_url = 'http://host.docker.internal:5000'

# Tunggu server siap
max_retries = 5
retry_delay = 2
success = False

print('Memulai pengujian fungsional...')

for i in range(max_retries):
    try:
        print(f'Mencoba koneksi ke {base_url}...')
        response = requests.get(f'{base_url}/')
        if response.status_code == 200:
            print('Koneksi berhasil!')
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
    print('Mengetes endpoint model-info...')
    response = requests.get(f'{base_url}/model-info')
    assert response.status_code == 200, 'Endpoint model-info tidak berfungsi'
    print('Test model-info berhasil')
    
    # Test 2: Melakukan prediksi
    print('Mengetes endpoint predict...')
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
    print('Test predict berhasil')
    
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
"
                '''
                
                // Mengarsipkan hasil pengujian
                archiveArtifacts artifacts: 'functional_test_results.json', fingerprint: true
            }
        }
    }
    
    post {
        always {
            echo 'Membersihkan container...'
            bat 'docker stop flask-bank-app 2>nul || echo "Container tidak berjalan"'
            bat 'docker rm flask-bank-app 2>nul || echo "Container tidak ada"'
        }
        success {
            echo 'Pipeline machine learning berhasil dijalankan!'
        }
        failure {
            echo 'Pipeline machine learning gagal dijalankan!'
        }
    }
}
