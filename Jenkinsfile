pipeline {
    agent any
    
    environment {
        // Path ke Python virtual environment
        VENV_PATH = "${WORKSPACE}/venv"
    }
    
    stages {
        stage('Setup Environment') {
            steps {
                echo 'Menyiapkan lingkungan pengembangan...'

                // Menetapkan path yang benar berdasarkan platform
                script {
                    if (isUnix()) {
                        env.PATH_TO_PYTHON = "${VENV_PATH}/bin:${PATH}"
                    } else {
                        env.PATH_TO_PYTHON = "${VENV_PATH}\\Scripts;${PATH}"
                    }
                }
                
                // Membuat direktori untuk hasil
                script {
                    if (isUnix()) {
                        sh 'mkdir -p models reports visualizations data predictions'
                    } else {
                        bat 'mkdir models reports visualizations data predictions 2>nul'
                    }
                }
                
                // Membuat virtual environment dan menginstal dependensi
                script {
                    if (isUnix()) {
                        sh '''
                            python -m venv venv
                            . venv/bin/activate
                            pip install --upgrade pip
                            pip install pandas numpy matplotlib seaborn scikit-learn flask pytest pytest-cov
                        '''
                    } else {
                        bat '''
                            python -m venv venv
                            call venv\\Scripts\\activate.bat
                            pip install --upgrade pip
                            pip install pandas numpy matplotlib seaborn scikit-learn flask pytest pytest-cov
                        '''
                    }
                }
            }
        }
        
        stage('Data Validation') {
            steps {
                echo 'Memvalidasi data...'
                
                script {
                    if (isUnix()) {
                        sh '''
                            . venv/bin/activate
                            python -c "
import pandas as pd
import sys

try:
    # Memuat data
    data = pd.read_csv('bank.csv', sep=';')
    
    # Validasi jumlah baris dan kolom
    assert data.shape[0] > 100, 'Data memiliki terlalu sedikit baris'
    assert data.shape[1] >= 16, 'Data tidak memiliki kolom yang cukup'
    
    # Validasi kolom target
    assert 'y' in data.columns, 'Kolom target tidak ditemukan'
    
    # Validasi tipe data
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    for col in numeric_cols:
        assert col in data.columns, f'Kolom {col} tidak ditemukan'
        assert pd.api.types.is_numeric_dtype(data[col]), f'Kolom {col} bukan tipe numerik'
    
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for col in categorical_cols:
        assert col in data.columns, f'Kolom {col} tidak ditemukan'
    
    # Validasi nilai target
    assert set(data['y'].unique()) == {'yes', 'no'}, 'Nilai kolom target tidak valid'
    
    print('Validasi data sukses!')
    sys.exit(0)
except Exception as e:
    print(f'Validasi data gagal: {str(e)}')
    sys.exit(1)
            "
                        '''
                    } else {
                        bat '''
                            venv\\Scripts\\activate.bat
                            python -c "
import pandas as pd
import sys

try:
    # Memuat data
    data = pd.read_csv('bank.csv', sep=';')
    
    # Validasi jumlah baris dan kolom
    assert data.shape[0] > 100, 'Data memiliki terlalu sedikit baris'
    assert data.shape[1] >= 16, 'Data tidak memiliki kolom yang cukup'
    
    # Validasi kolom target
    assert 'y' in data.columns, 'Kolom target tidak ditemukan'
    
    # Validasi tipe data
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    for col in numeric_cols:
        assert col in data.columns, f'Kolom {col} tidak ditemukan'
        assert pd.api.types.is_numeric_dtype(data[col]), f'Kolom {col} bukan tipe numerik'
    
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for col in categorical_cols:
        assert col in data.columns, f'Kolom {col} tidak ditemukan'
    
    # Validasi nilai target
    assert set(data['y'].unique()) == {'yes', 'no'}, 'Nilai kolom target tidak valid'
    
    print('Validasi data sukses!')
    sys.exit(0)
except Exception as e:
    print(f'Validasi data gagal: {str(e)}')
    sys.exit(1)
            "
                        '''
                    }
                }
            }
        }
        
        stage('Exploratory Data Analysis') {
            steps {
                echo 'Melakukan Exploratory Data Analysis...'
                
                script {
                    if (isUnix()) {
                        sh '''
                            . venv/bin/activate
                            python -c "
from bank_ml_pipeline import BankMarketingPipeline

# Inisialisasi pipeline
pipeline = BankMarketingPipeline(data_path='bank.csv')

# Melakukan EDA
insights = pipeline.perform_eda()
print('EDA selesai dan insight dihasilkan')
            "
                        '''
                    } else {
                        bat '''
                            venv\\Scripts\\activate.bat
                            python -c "
from bank_ml_pipeline import BankMarketingPipeline

# Inisialisasi pipeline
pipeline = BankMarketingPipeline(data_path='bank.csv')

# Melakukan EDA
insights = pipeline.perform_eda()
print('EDA selesai dan insight dihasilkan')
            "
                        '''
                    }
                }
                
                // Mengarsipkan hasil EDA
                archiveArtifacts artifacts: 'visualizations/*.png, reports/eda_insights.txt, reports/data_info.json', fingerprint: true
            }
        }
        
        stage('Data Preprocessing') {
            steps {
                echo 'Melakukan preprocessing data...'
                
                script {
                    if (isUnix()) {
                        sh '''
                            . venv/bin/activate
                            python -c "
from bank_ml_pipeline import BankMarketingPipeline

# Inisialisasi pipeline
pipeline = BankMarketingPipeline(data_path='bank.csv')

# Memuat data
pipeline.load_data()

# Melakukan preprocessing
X_train, X_test, y_train, y_test = pipeline.preprocess_data()
print('Preprocessing data selesai')
print(f'Data training: {X_train.shape[0]} sampel, {X_train.shape[1]} fitur')
print(f'Data testing: {X_test.shape[0]} sampel, {X_test.shape[1]} fitur')
            "
                        '''
                    } else {
                        bat '''
                            venv\\Scripts\\activate.bat
                            python -c "
from bank_ml_pipeline import BankMarketingPipeline

# Inisialisasi pipeline
pipeline = BankMarketingPipeline(data_path='bank.csv')

# Memuat data
pipeline.load_data()

# Melakukan preprocessing
X_train, X_test, y_train, y_test = pipeline.preprocess_data()
print('Preprocessing data selesai')
print(f'Data training: {X_train.shape[0]} sampel, {X_train.shape[1]} fitur')
print(f'Data testing: {X_test.shape[0]} sampel, {X_test.shape[1]} fitur')
            "
                        '''
                    }
                }
                
                // Mengarsipkan hasil preprocessing
                archiveArtifacts artifacts: 'data/train_data.csv, data/test_data.csv', fingerprint: true
            }
        }
        
        stage('Model Training') {
            steps {
                echo 'Melatih model...'
                
                script {
                    if (isUnix()) {
                        sh '''
                            . venv/bin/activate
                            python -c "
from bank_ml_pipeline import BankMarketingPipeline

# Inisialisasi pipeline
pipeline = BankMarketingPipeline(data_path='bank.csv')

# Memuat data dan preprocess
pipeline.load_data()
pipeline.preprocess_data()

# Melatih model
best_model, metrics = pipeline.train_models()
print(f'Model terbaik: {metrics[\"best_model\"]}')
print(f'ROC-AUC Test: {metrics[\"models\"][metrics[\"best_model\"]][\"test_roc_auc\"]:.4f}')
            "
                        '''
                    } else {
                        bat '''
                            venv\\Scripts\\activate.bat
                            python -c "
from bank_ml_pipeline import BankMarketingPipeline

# Inisialisasi pipeline
pipeline = BankMarketingPipeline(data_path='bank.csv')

# Memuat data dan preprocess
pipeline.load_data()
pipeline.preprocess_data()

# Melatih model
best_model, metrics = pipeline.train_models()
print(f'Model terbaik: {metrics[\"best_model\"]}')
print(f'ROC-AUC Test: {metrics[\"models\"][metrics[\"best_model\"]][\"test_roc_auc\"]:.4f}')
            "
                        '''
                    }
                }
                
                // Mengarsipkan model dan metrik
                archiveArtifacts artifacts: 'models/*.pkl, models/*.json, reports/model_metrics.json, reports/model_evaluation.json, reports/evaluation_summary.txt, visualizations/roc_curve.png, visualizations/precision_recall_curve.png, visualizations/confusion_matrix.png', fingerprint: true
            }
        }
        
        stage('Deploy Web Server') {
            steps {
                echo 'Men-deploy webserver untuk monitoring model...'
                
                script {
                    if (isUnix()) {
                        sh '''
                            . venv/bin/activate
                            mkdir -p templates
                            
                            # Menjalankan Flask server sebagai background process
                            nohup python app.py > flask.log 2>&1 &
                            echo $! > .pidfile
                            sleep 5  # Memberi waktu server untuk startup
                            
                            # Memeriksa apakah server berjalan
                            if ps -p `cat .pidfile` > /dev/null; then
                                echo "Flask server berjalan pada PID: `cat .pidfile`"
                            else
                                echo "ERROR: Flask server gagal dimulai"
                                exit 1
                            fi
                        '''
                    } else {
                        bat '''
                            venv\\Scripts\\activate.bat
                            mkdir templates 2>nul
                            
                            # Menjalankan Flask server menggunakan start command (Windows)
                            start /b python app.py > flask.log 2>&1
                            
                            # Memberikan waktu server untuk startup
                            ping 127.0.0.1 -n 6 > nul
                            
                            # Memeriksa apakah server berjalan (Windows tidak dapat menyimpan PID dengan mudah)
                            netstat -ano | findstr :5000
                            if %errorlevel% equ 0 (
                                echo "Flask server berjalan pada port 5000"
                            ) else (
                                echo "ERROR: Flask server gagal dimulai"
                                exit /b 1
                            )
                        '''
                    }
                }
                
                echo 'Webserver untuk monitoring model berhasil di-deploy'
            }
        }
        
        stage('Functional Testing') {
            steps {
                echo 'Menjalankan pengujian fungsional...'
                
                script {
                    if (isUnix()) {
                        sh '''
                            . venv/bin/activate
                            
                            # Menjalankan pengujian fungsional terhadap API
                            python -c "
import requests
import sys
import json

# Base URL API 
base_url = 'http://localhost:5000'

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
    
    result = response.json()
    assert 'prediction' in result, 'Response API tidak memiliki hasil prediksi'
    assert 'probability' in result, 'Response API tidak memiliki probabilitas'
    
    # Test 3: Mendapatkan data monitoring
    response = requests.get(f'{base_url}/monitoring-data')
    assert response.status_code == 200, 'Endpoint monitoring-data tidak berfungsi'
    
    # Menulis hasil tes ke file
    with open('functional_test_results.json', 'w') as f:
        json.dump({
            'status': 'success',
            'message': 'Semua pengujian fungsional berhasil',
            'timestamp': requests.get(f'{base_url}/model-info').json().get('timestamp')
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
                    } else {
                        bat '''
                            venv\\Scripts\\activate.bat
                            
                            # Menjalankan pengujian fungsional terhadap API
                            python -c "
import requests
import sys
import json

# Base URL API 
base_url = 'http://localhost:5000'

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
    
    result = response.json()
    assert 'prediction' in result, 'Response API tidak memiliki hasil prediksi'
    assert 'probability' in result, 'Response API tidak memiliki probabilitas'
    
    # Test 3: Mendapatkan data monitoring
    response = requests.get(f'{base_url}/monitoring-data')
    assert response.status_code == 200, 'Endpoint monitoring-data tidak berfungsi'
    
    # Menulis hasil tes ke file
    with open('functional_test_results.json', 'w') as f:
        json.dump({
            'status': 'success',
            'message': 'Semua pengujian fungsional berhasil',
            'timestamp': requests.get(f'{base_url}/model-info').json().get('timestamp')
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
                    }
                }
                
                // Mengarsipkan hasil pengujian
                archiveArtifacts artifacts: 'functional_test_results.json', fingerprint: true
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline machine learning berhasil dijalankan!'
        }
        failure {
            echo 'Pipeline machine learning gagal dijalankan!'
        }
        always {
            // Membersihkan proses background (server Flask)
            script {
                if (isUnix()) {
                    sh '''
                        if [ -f .pidfile ]; then
                            kill `cat .pidfile` || true
                            rm .pidfile
                        fi
                    '''
                } else {
                    bat '''
                        for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING 2^>nul') do (
                            taskkill /F /PID %%a 2>nul || echo Process tidak ditemukan
                        )
                    '''
                }
            }
        }
    }
}
