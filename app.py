from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import json
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webserver.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('bank_webserver')

# Direktori untuk menyimpan data prediksi baru
PREDICTIONS_DIR = 'predictions'
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Memuat model terbaik
def load_model():
    try:
        model_path = 'models/best_model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model berhasil dimuat dari {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error saat memuat model: {str(e)}")
        return None

# Memuat metadata model
def load_model_metadata():
    try:
        metadata_path = 'models/best_model_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Error saat memuat metadata model: {str(e)}")
        return {}

# Memuat ringkasan evaluasi model
def load_evaluation_summary():
    try:
        with open('reports/evaluation_summary.txt', 'r') as f:
            summary = f.read()
        return summary
    except Exception as e:
        logger.error(f"Error saat memuat ringkasan evaluasi: {str(e)}")
        return "Ringkasan evaluasi tidak tersedia"

# Memuat metrik model
def load_model_metrics():
    try:
        with open('reports/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logger.error(f"Error saat memuat metrik model: {str(e)}")
        return {}

# Endpoint untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html', 
                           evaluation_summary=load_evaluation_summary(),
                           model_metrics=load_model_metrics())

# Endpoint untuk prediksi via API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Memuat model
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model tidak dapat dimuat'}), 500
        
        # Mendapatkan data dari request
        data = request.json
        
        # Membuat DataFrame dari data input
        input_data = pd.DataFrame([data])
        
        # Melakukan prediksi
        prediction_proba = model.predict_proba(input_data)[0, 1]
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Mencatat prediksi
        prediction_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input': data,
            'prediction': prediction,
            'probability': prediction_proba
        }
        
        # Menyimpan prediksi ke file
        predictions_file = os.path.join(PREDICTIONS_DIR, 'predictions.jsonl')
        with open(predictions_file, 'a') as f:
            f.write(json.dumps(prediction_record) + '\n')
        
        # Mencatat aktivitas prediksi
        logger.info(f"Prediksi dilakukan: {prediction_record['timestamp']}, hasil: {prediction}")
        
        return jsonify({
            'prediction': prediction,
            'probability': prediction_proba,
            'result_text': 'Berlangganan' if prediction == 1 else 'Tidak Berlangganan'
        })
    
    except Exception as e:
        logger.error(f"Error saat melakukan prediksi: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mengambil data monitoring
@app.route('/monitoring-data')
def monitoring_data():
    try:
        # Memuat riwayat prediksi
        predictions = []
        predictions_file = os.path.join(PREDICTIONS_DIR, 'predictions.jsonl')
        
        if os.path.exists(predictions_file):
            with open(predictions_file, 'r') as f:
                for line in f:
                    if line.strip():
                        predictions.append(json.loads(line))
        
        # Menghitung metrik monitoring
        total_predictions = len(predictions)
        
        # Data distribusi prediksi per waktu
        timestamps = [p['timestamp'] for p in predictions]
        prediction_results = [p['prediction'] for p in predictions]
        probabilities = [p['probability'] for p in predictions]
        
        # Statistik sederhana
        if total_predictions > 0:
            positive_rate = sum(prediction_results) / total_predictions
            avg_probability = sum(probabilities) / total_predictions
        else:
            positive_rate = 0
            avg_probability = 0
        
        return jsonify({
            'total_predictions': total_predictions,
            'positive_rate': positive_rate,
            'average_probability': avg_probability,
            'predictions': predictions[-100:]  # Mengembalikan 100 prediksi terakhir
        })
    
    except Exception as e:
        logger.error(f"Error saat mengambil data monitoring: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk melihat distribusi prediksi
@app.route('/prediction-distribution')
def prediction_distribution():
    try:
        # Memuat riwayat prediksi
        predictions = []
        predictions_file = os.path.join(PREDICTIONS_DIR, 'predictions.jsonl')
        
        if os.path.exists(predictions_file):
            with open(predictions_file, 'r') as f:
                for line in f:
                    if line.strip():
                        predictions.append(json.loads(line))
        
        # Jika tidak ada prediksi
        if not predictions:
            return jsonify({'error': 'Tidak ada data prediksi'}), 404
        
        # Membuat DataFrame dari prediksi
        df_pred = pd.DataFrame(predictions)
        
        # Plot distribusi probabilitas
        plt.figure(figsize=(10, 6))
        sns.histplot(df_pred['probability'], bins=20, kde=True)
        plt.title('Distribusi Probabilitas Prediksi')
        plt.xlabel('Probabilitas')
        plt.ylabel('Frekuensi')
        
        # Mengkonversi plot ke base64 untuk ditampilkan di browser
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        graph = base64.b64encode(image_png).decode('utf-8')
        
        return jsonify({
            'distribution_graph': f'data:image/png;base64,{graph}'
        })
    
    except Exception as e:
        logger.error(f"Error saat menghasilkan distribusi prediksi: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk retraining model
@app.route('/retrain', methods=['POST'])
def retrain_model():
    try:
        from bank_ml_pipeline import BankMarketingPipeline
        
        # Menjalankan pipeline retraining
        pipeline = BankMarketingPipeline()
        result = pipeline.run_pipeline()
        
        logger.info(f"Retraining model selesai dengan hasil: {result}")
        
        return jsonify({
            'status': 'success',
            'message': 'Model berhasil dilatih ulang',
            'result': result
        })
    
    except Exception as e:
        logger.error(f"Error saat melatih ulang model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error saat melatih ulang model: {str(e)}'
        }), 500

# Endpoint untuk mendapatkan informasi model
@app.route('/model-info')
def model_info():
    try:
        # Memuat metadata dan metrik model
        metadata = load_model_metadata()
        metrics = load_model_metrics()
        
        # Menggabungkan informasi
        model_information = {
            'metadata': metadata,
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(model_information)
    
    except Exception as e:
        logger.error(f"Error saat mengambil informasi model: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan data drift
@app.route('/data-drift')
def data_drift():
    try:
        # Memuat data asli
        original_data = pd.read_csv('data/original_data.csv')
        
        # Memuat riwayat prediksi
        predictions = []
        predictions_file = os.path.join(PREDICTIONS_DIR, 'predictions.jsonl')
        
        if os.path.exists(predictions_file):
            with open(predictions_file, 'r') as f:
                for line in f:
                    if line.strip():
                        predictions.append(json.loads(line))
        
        # Jika tidak ada prediksi
        if not predictions:
            return jsonify({'error': 'Tidak ada data prediksi untuk analisis drift'}), 404
        
        # Mengekstrak input data dari prediksi
        input_records = [p['input'] for p in predictions]
        new_data = pd.DataFrame(input_records)
        
        # Membandingkan statistik dasar untuk fitur numerik
        original_num_stats = original_data.select_dtypes(include=['int64', 'float64']).describe().to_dict()
        new_num_stats = new_data.select_dtypes(include=['int64', 'float64']).describe().to_dict()
        
        # Menghitung perbedaan persentase untuk mean
        drift_metrics = {}
        
        for feature in original_num_stats:
            if feature in new_num_stats:
                original_mean = original_num_stats[feature]['mean']
                new_mean = new_num_stats[feature]['mean']
                
                if original_mean != 0:
                    drift_percentage = abs((new_mean - original_mean) / original_mean) * 100
                else:
                    drift_percentage = abs(new_mean - original_mean) * 100
                
                drift_metrics[feature] = {
                    'original_mean': original_mean,
                    'new_mean': new_mean,
                    'drift_percentage': drift_percentage,
                    'significant_drift': drift_percentage > 10  # Threshold 10%
                }
        
        return jsonify({
            'drift_metrics': drift_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        logger.error(f"Error saat menganalisis data drift: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Memastikan model dapat dimuat sebelum menjalankan aplikasi
    model = load_model()
    if model is None:
        logger.error("Aplikasi tidak dapat dimulai karena model tidak dapat dimuat")
    else:
        app.run(debug=True, host='0.0.0.0', port=5000) 