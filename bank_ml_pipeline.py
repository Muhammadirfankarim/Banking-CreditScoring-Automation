import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import os
import json
import logging
from datetime import datetime

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bank_ml_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('bank_ml_pipeline')

class BankMarketingPipeline:
    def __init__(self, data_path='bank.csv', random_state=42):
        """
        Inisialisasi pipeline machine learning untuk dataset Bank Marketing
        
        Args:
            data_path: Path ke file CSV
            random_state: Random seed untuk reprodusibilitas
        """
        self.data_path = data_path
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.model_metrics = {}
        self.feature_importance = None
        
        # Membuat direktori untuk menyimpan hasil
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        logger.info("Pipeline Bank Marketing ML diinisialisasi")
    
    def load_data(self):
        """
        Membaca data dari file CSV
        """
        logger.info(f"Memuat data dari {self.data_path}")
        try:
            self.data = pd.read_csv(self.data_path, sep=';')
            logger.info(f"Data dimuat sukses dengan {self.data.shape[0]} baris dan {self.data.shape[1]} kolom")
            
            # Menyimpan data asli
            self.data.to_csv('data/original_data.csv', index=False)
            
            return self.data
        except Exception as e:
            logger.error(f"Error saat memuat data: {str(e)}")
            raise
    
    def perform_eda(self):
        """
        Melakukan Exploratory Data Analysis (EDA) dan menghasilkan insight
        """
        logger.info("Melakukan Exploratory Data Analysis (EDA)")
        
        if self.data is None:
            self.load_data()
        
        # Informasi umum dan statistik dasar
        data_info = {
            "jumlah_baris": self.data.shape[0],
            "jumlah_kolom": self.data.shape[1],
            "kolom": list(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict(),
            "tipe_data": {col: str(dtype) for col, dtype in self.data.dtypes.items()}
        }
        
        # Menyimpan informasi dasar
        with open('reports/data_info.json', 'w') as f:
            json.dump(data_info, f, indent=4)
        
        # Membuat visualisasi distribusi target
        plt.figure(figsize=(8, 6))
        sns.countplot(x='y', data=self.data)
        plt.title('Distribusi Target Variable (Subscription)')
        plt.savefig('visualizations/target_distribution.png')
        plt.close()
        
        # Membuat visualisasi untuk fitur numerik
        numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        for feature in numeric_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[feature], kde=True)
            plt.title(f'Distribusi dari {feature}')
            plt.savefig(f'visualizations/{feature}_distribution.png')
            plt.close()
            
            # Box plot berdasarkan target
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='y', y=feature, data=self.data)
            plt.title(f'{feature} berdasarkan Target')
            plt.savefig(f'visualizations/{feature}_by_target.png')
            plt.close()
        
        # Visualisasi untuk fitur kategorikal
        categorical_features = self.data.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            if feature != 'y':  # Mengecualikan target variable
                plt.figure(figsize=(12, 6))
                sns.countplot(x=feature, hue='y', data=self.data)
                plt.title(f'{feature} berdasarkan Target')
                plt.xticks(rotation=45)
                plt.savefig(f'visualizations/{feature}_by_target.png')
                plt.close()
        
        # Correlation matrix untuk fitur numerik
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.data.select_dtypes(include=['int64', 'float64']).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Matriks Korelasi Fitur Numerik')
        plt.savefig('visualizations/correlation_matrix.png')
        plt.close()
        
        # Menghasilkan insight
        insights = self._generate_insights()
        
        # Menyimpan insight ke file
        with open('reports/eda_insights.txt', 'w') as f:
            f.write(insights)
        
        logger.info("EDA selesai, insight dan visualisasi disimpan")
        return insights
    
    def _generate_insights(self):
        """
        Menghasilkan insight berdasarkan EDA
        """
        insights = "INSIGHT DARI EXPLORATORY DATA ANALYSIS (EDA)\n"
        insights += "="*50 + "\n\n"
        
        # Insight tentang distribusi target
        target_counts = self.data['y'].value_counts()
        target_percentage = target_counts / len(self.data) * 100
        
        insights += f"1. DISTRIBUSI TARGET:\n"
        insights += f"   - Jumlah 'yes' (berlangganan): {target_counts['yes']} ({target_percentage['yes']:.2f}%)\n"
        insights += f"   - Jumlah 'no' (tidak berlangganan): {target_counts['no']} ({target_percentage['no']:.2f}%)\n"
        
        if target_percentage['yes'] < 30:
            insights += f"   - Dataset tidak seimbang, perlu strategi penanganan khusus\n\n"
        else:
            insights += f"   - Distribusi target cukup seimbang\n\n"
        
        # Insight tentang fitur numerik
        numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        insights += f"2. FITUR NUMERIK:\n"
        
        for feature in numeric_features:
            mean_by_target = self.data.groupby('y')[feature].mean()
            insights += f"   - {feature}:\n"
            insights += f"     * Rata-rata untuk 'yes': {mean_by_target['yes']:.2f}\n"
            insights += f"     * Rata-rata untuk 'no': {mean_by_target['no']:.2f}\n"
            
            if abs(mean_by_target['yes'] - mean_by_target['no']) > 0.1 * self.data[feature].std():
                insights += f"     * Terdapat perbedaan signifikan antara kedua grup\n"
        
        insights += "\n"
        
        # Insight tentang fitur kategorikal
        categorical_features = self.data.select_dtypes(include=['object']).columns
        insights += f"3. FITUR KATEGORIKAL:\n"
        
        for feature in categorical_features:
            if feature != 'y':  # Mengecualikan target variable
                insights += f"   - {feature}:\n"
                
                # Menghitung proporsi konversi untuk setiap kategori
                category_conversion = self.data.groupby(feature)['y'].apply(
                    lambda x: (x == 'yes').mean() * 100
                ).sort_values(ascending=False)
                
                # Menampilkan kategori dengan konversi tertinggi dan terendah
                insights += f"     * Kategori dengan tingkat konversi tertinggi: {category_conversion.index[0]} ({category_conversion.iloc[0]:.2f}%)\n"
                insights += f"     * Kategori dengan tingkat konversi terendah: {category_conversion.index[-1]} ({category_conversion.iloc[-1]:.2f}%)\n"
        
        insights += "\n"
        
        # Insight tentang korelasi
        insights += f"4. KORELASI DAN HUBUNGAN:\n"
        
        # Korelasi antara fitur numerik
        numeric_corr = self.data.select_dtypes(include=['int64', 'float64']).corr()
        high_corr_pairs = []
        
        for i in range(len(numeric_corr.columns)):
            for j in range(i+1, len(numeric_corr.columns)):
                if abs(numeric_corr.iloc[i, j]) > 0.5:
                    high_corr_pairs.append((
                        numeric_corr.columns[i],
                        numeric_corr.columns[j],
                        numeric_corr.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            insights += "   - Fitur dengan korelasi tinggi:\n"
            for pair in high_corr_pairs:
                insights += f"     * {pair[0]} dan {pair[1]}: {pair[2]:.2f}\n"
        else:
            insights += "   - Tidak ada fitur numerik dengan korelasi tinggi\n"
        
        insights += "\n"
        
        # Kesimpulan dan rekomendasi
        insights += f"5. KESIMPULAN DAN REKOMENDASI:\n"
        insights += f"   - Target tidak seimbang, pertimbangkan teknik resampling atau class_weight\n"
        insights += f"   - Fitur duration kemungkinan akan menjadi sangat prediktif namun bersifat leak information\n"
        insights += f"   - Perlu encoding untuk variabel kategorikal\n"
        insights += f"   - Standardisasi fitur numerik dapat membantu meningkatkan performa\n"
        
        return insights
    
    def preprocess_data(self, test_size=0.2):
        """
        Melakukan preprocessing data dan membagi menjadi train dan test set
        """
        logger.info("Memulai preprocessing data")
        
        if self.data is None:
            self.load_data()
        
        # Memisahkan fitur dan target
        X = self.data.drop('y', axis=1)
        y = (self.data['y'] == 'yes').astype(int)  # Mengkonversi ke 0 dan 1
        
        # Mengidentifikasi tipe kolom
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Fitur numerik: {numeric_features}")
        logger.info(f"Fitur kategorikal: {categorical_features}")
        
        # Membagi data menjadi training dan testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Data dibagi: {self.X_train.shape[0]} sampel training, {self.X_test.shape[0]} sampel testing")
        
        # Menyimpan data training dan testing
        train_data = pd.concat([self.X_train, self.y_train.rename('y')], axis=1)
        test_data = pd.concat([self.X_test, self.y_test.rename('y')], axis=1)
        
        train_data.to_csv('data/train_data.csv', index=False)
        test_data.to_csv('data/test_data.csv', index=False)
        
        logger.info("Preprocessing dan pembagian data selesai")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        Melatih dan mengevaluasi beberapa model machine learning
        dan memilih model terbaik
        """
        logger.info("Memulai proses training model")
        
        if self.X_train is None or self.y_train is None:
            self.preprocess_data()
        
        # Mengidentifikasi tipe kolom
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.X_train.select_dtypes(include=['object']).columns.tolist()
        
        # Membuat preprocessor
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
        # Mendefinisikan model-model yang akan dilatih
        models = {
            'logistic_regression': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(max_iter=1000, random_state=self.random_state))
            ]),
            'random_forest': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=self.random_state))
            ]),
            'gradient_boosting': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(random_state=self.random_state))
            ])
        }
        
        # Parameter grid untuk Grid Search
        param_grids = {
            'logistic_regression': {
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__class_weight': [None, 'balanced']
            },
            'random_forest': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5],
                'classifier__class_weight': [None, 'balanced']
            },
            'gradient_boosting': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5]
            }
        }
        
        best_models = {}
        model_scores = {}
        
        # Training dan evaluasi untuk setiap model
        for model_name, model in models.items():
            logger.info(f"Training model: {model_name}")
            
            # Melakukan Grid Search
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=5, scoring='roc_auc', n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Menyimpan model terbaik
            best_models[model_name] = grid_search.best_estimator_
            model_scores[model_name] = {
                'best_params': grid_search.best_params_,
                'train_roc_auc': grid_search.best_score_,
                'test_roc_auc': roc_auc_score(self.y_test, best_models[model_name].predict_proba(self.X_test)[:, 1])
            }
            
            logger.info(f"Model {model_name} - Train ROC-AUC: {model_scores[model_name]['train_roc_auc']:.4f}, "
                       f"Test ROC-AUC: {model_scores[model_name]['test_roc_auc']:.4f}")
            
            # Menyimpan model
            with open(f"models/{model_name}.pkl", 'wb') as f:
                pickle.dump(best_models[model_name], f)
        
        # Memilih model terbaik berdasarkan ROC-AUC pada test set
        best_model_name = max(model_scores, key=lambda x: model_scores[x]['test_roc_auc'])
        self.best_model = best_models[best_model_name]
        
        logger.info(f"Model terbaik: {best_model_name} dengan Test ROC-AUC: {model_scores[best_model_name]['test_roc_auc']:.4f}")
        
        # Menyimpan informasi model
        self.model_metrics = {
            'best_model': best_model_name,
            'models': model_scores,
            'feature_importance': self._get_feature_importance(self.best_model, model_name=best_model_name)
        }
        
        # Menyimpan metrik model ke file JSON
        with open('reports/model_metrics.json', 'w') as f:
            json.dump(self.model_metrics, f, indent=4)
        
        # Mengevaluasi model terbaik
        self.evaluate_best_model()
        
        logger.info("Training model selesai")
        return self.best_model, self.model_metrics
    
    def _get_feature_importance(self, model, model_name):
        """
        Mengekstrak feature importance dari model (jika tersedia)
        """
        try:
            if model_name == 'logistic_regression':
                # Untuk model regresi logistik
                feature_names = (
                    self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist() +
                    model.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(
                        self.X_train.select_dtypes(include=['object']).columns
                    ).tolist()
                )
                coefficients = model.named_steps['classifier'].coef_[0]
                return dict(zip(feature_names, coefficients))
            
            elif model_name in ['random_forest', 'gradient_boosting']:
                # Untuk model tree-based
                feature_names = (
                    self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist() +
                    model.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(
                        self.X_train.select_dtypes(include=['object']).columns
                    ).tolist()
                )
                importances = model.named_steps['classifier'].feature_importances_
                return dict(zip(feature_names, importances))
            
            else:
                return {}
        
        except Exception as e:
            logger.warning(f"Tidak dapat mengekstrak feature importance: {str(e)}")
            return {}
    
    def evaluate_best_model(self):
        """
        Mengevaluasi model terbaik dan menghasilkan laporan performa
        """
        if self.best_model is None:
            logger.warning("Model belum dilatih, tidak dapat melakukan evaluasi")
            return
        
        logger.info("Mengevaluasi model terbaik")
        
        # Memprediksi pada data testing
        y_pred = self.best_model.predict(self.X_test)
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        class_report = classification_report(self.y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        
        # Precision-Recall
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        pr_auc = auc(recall, precision)
        
        # Menyimpan metrics ke file
        evaluation = {
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
        with open('reports/model_evaluation.json', 'w') as f:
            json.dump(evaluation, f, indent=4)
        
        # Membuat visualisasi evaluasi
        self._create_evaluation_plots(y_proba)
        
        logger.info(f"Evaluasi model selesai - ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
        
        # Membuat ringkasan evaluasi dalam format teks
        evaluation_summary = self._create_evaluation_summary(class_report, conf_matrix, roc_auc, pr_auc)
        
        with open('reports/evaluation_summary.txt', 'w') as f:
            f.write(evaluation_summary)
        
        return evaluation
    
    def _create_evaluation_plots(self, y_proba):
        """
        Membuat plot visualisasi untuk evaluasi model
        """
        # ROC Curve
        plt.figure(figsize=(10, 8))
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay.from_predictions(
            self.y_test, 
            y_proba,
            name="ROC Curve",
            color="darkorange",
        )
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.savefig('visualizations/roc_curve.png')
        plt.close()
        
        # Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        from sklearn.metrics import PrecisionRecallDisplay
        PrecisionRecallDisplay.from_predictions(
            self.y_test,
            y_proba,
            name="Precision-Recall Curve",
            color="darkorange",
        )
        plt.axis("square")
        plt.title("Precision-Recall Curve")
        plt.savefig('visualizations/precision_recall_curve.png')
        plt.close()
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_predictions(
            self.y_test,
            self.best_model.predict(self.X_test),
            cmap=plt.cm.Blues,
            normalize='true'
        )
        plt.title("Confusion Matrix (Normalized)")
        plt.savefig('visualizations/confusion_matrix.png')
        plt.close()
    
    def _create_evaluation_summary(self, class_report, conf_matrix, roc_auc, pr_auc):
        """
        Membuat ringkasan evaluasi model dalam format teks
        """
        summary = "RINGKASAN EVALUASI MODEL\n"
        summary += "=" * 50 + "\n\n"
        
        summary += "1. METRIK PERFORMA:\n"
        summary += f"   - Accuracy: {class_report['accuracy']:.4f}\n"
        summary += f"   - ROC-AUC: {roc_auc:.4f}\n"
        summary += f"   - PR-AUC: {pr_auc:.4f}\n\n"
        
        summary += "2. METRIK PER KELAS:\n"
        summary += f"   - Class 0 (Tidak Berlangganan):\n"
        summary += f"     * Precision: {class_report['0']['precision']:.4f}\n"
        summary += f"     * Recall: {class_report['0']['recall']:.4f}\n"
        summary += f"     * F1-Score: {class_report['0']['f1-score']:.4f}\n"
        summary += f"     * Support: {class_report['0']['support']}\n\n"
        
        summary += f"   - Class 1 (Berlangganan):\n"
        summary += f"     * Precision: {class_report['1']['precision']:.4f}\n"
        summary += f"     * Recall: {class_report['1']['recall']:.4f}\n"
        summary += f"     * F1-Score: {class_report['1']['f1-score']:.4f}\n"
        summary += f"     * Support: {class_report['1']['support']}\n\n"
        
        summary += "3. CONFUSION MATRIX:\n"
        summary += f"   - True Negatives: {conf_matrix[0, 0]}\n"
        summary += f"   - False Positives: {conf_matrix[0, 1]}\n"
        summary += f"   - False Negatives: {conf_matrix[1, 0]}\n"
        summary += f"   - True Positives: {conf_matrix[1, 1]}\n\n"
        
        summary += "4. INTERPRETASI:\n"
        
        # Interpretasi accuracy
        if class_report['accuracy'] > 0.9:
            summary += "   - Accuracy sangat tinggi, model memiliki performa yang baik\n"
        elif class_report['accuracy'] > 0.8:
            summary += "   - Accuracy cukup baik\n"
        else:
            summary += "   - Accuracy perlu ditingkatkan\n"
        
        # Interpretasi recall kelas positif
        if class_report['1']['recall'] < 0.6:
            summary += "   - Recall untuk kelas positif rendah, model mungkin melewatkan banyak kasus positif\n"
        else:
            summary += "   - Recall untuk kelas positif cukup baik\n"
        
        # Interpretasi precision kelas positif
        if class_report['1']['precision'] < 0.6:
            summary += "   - Precision untuk kelas positif rendah, model memberikan banyak false positive\n"
        else:
            summary += "   - Precision untuk kelas positif cukup baik\n"
        
        return summary
    
    def save_model(self, filename='best_model.pkl'):
        """
        Menyimpan model terbaik ke file
        """
        if self.best_model is None:
            logger.warning("Model belum dilatih, tidak dapat menyimpan model")
            return
        
        logger.info(f"Menyimpan model terbaik ke {filename}")
        
        try:
            with open(f"models/{filename}", 'wb') as f:
                pickle.dump(self.best_model, f)
            
            # Juga simpan metadata
            metadata = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': {
                    'roc_auc': roc_auc_score(
                        self.y_test, 
                        self.best_model.predict_proba(self.X_test)[:, 1]
                    )
                },
                'features': {
                    'numeric': self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                    'categorical': self.X_train.select_dtypes(include=['object']).columns.tolist()
                }
            }
            
            with open(f"models/{os.path.splitext(filename)[0]}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info("Model dan metadata disimpan sukses")
            return True
        
        except Exception as e:
            logger.error(f"Error saat menyimpan model: {str(e)}")
            return False
    
    def run_pipeline(self):
        """
        Menjalankan seluruh pipeline dari awal hingga akhir
        """
        logger.info("Menjalankan seluruh pipeline machine learning")
        
        try:
            # Load data
            self.load_data()
            
            # Perform EDA
            self.perform_eda()
            
            # Preprocess data
            self.preprocess_data()
            
            # Train and evaluate models
            self.train_models()
            
            # Save best model
            self.save_model()
            
            logger.info("Pipeline machine learning selesai dijalankan")
            
            return {
                'status': 'success',
                'best_model': self.model_metrics['best_model'],
                'test_roc_auc': self.model_metrics['models'][self.model_metrics['best_model']]['test_roc_auc']
            }
        
        except Exception as e:
            logger.error(f"Error saat menjalankan pipeline: {str(e)}")
            
            return {
                'status': 'error',
                'message': str(e)
            }

if __name__ == "__main__":
    pipeline = BankMarketingPipeline()
    result = pipeline.run_pipeline()
    print(result) 