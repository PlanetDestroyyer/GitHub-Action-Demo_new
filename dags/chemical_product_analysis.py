import os
import re
import warnings
import pandas as pd
import numpy as np
import pickle
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
import mlflow
import mlflow.sklearn
from airflow.utils.dates import days_ago

# Ignore warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Constants
EXCEL_FILE_PATH = "data/EHS_dupli_15.xlsx"
# TRACKING_URI = "/home/zuhair/lflow_tracking"
MLFLOW_UI_PORT = 5000
# OUTPUT_PATH = "/home/zuhair/airflow/dags/" # change the path if needed

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Instantiate the DAG
dag = DAG(
    'chemical_product_analysis',
    default_args=default_args,
    description='Chemical Product Analysis and Model Training with MLflow',
    schedule_interval='@daily',
    start_date=datetime(2024, 8, 18),
    catchup=False,
)

# Load and preprocess data
def load_and_preprocess_data(**kwargs):
    df = pd.read_excel(EXCEL_FILE_PATH)

    def chemical_info_break(value):
        # Extract chemical name and concentration from the 'Chemical Information' column
        if pd.isna(value):
            return None, None
        parts = value.split(',')
        chemical_name = parts[0].strip()
        chemical_concentration = parts[1].strip() if len(parts) > 1 else None
        return chemical_name, chemical_concentration

    # Apply the function to split chemical information into two columns
    df[['Chemical Name', 'Chemical Concentration']] = df['Chemical Information'].apply(chemical_info_break).apply(pd.Series)

    # Additional data cleaning and formatting logic can go here
    df_final = df.dropna(subset=['Chemical Name'])  # Example cleaning step

    # Save the processed data to a CSV file
    # output_path = os.path.join(OUTPUT_PATH, 'processed_data.csv')
    # df_final.to_csv(output_path, index=False)

    # Return the path of the processed file for downstream tasks
    return output_path

# Train models using the processed data
def train_models(**kwargs):
    # Load the processed data
    processed_data_path = kwargs['ti'].xcom_pull(task_ids='load_and_preprocess_data')
    df = pd.read_csv(processed_data_path)

    # Extract features and labels (example)
    X = df['Chemical Information']
    y = df['Waste_Code']  # Assume there's a label column for classification

    # Create embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X_embeddings = model.encode(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)

    # Initialize and train the SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics to MLflow
    # mlflow.set_tracking_uri(TRACKING_URI)
    with mlflow.start_run():
        mlflow.log_param("model_type", "SVM")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(svm_model, "svm_model")
    
    # Push metrics to XComs for viewing in Airflow UI
    kwargs['ti'].xcom_push(key='model_metrics', value={
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

# Print metrics
def print_metrics(**kwargs):
    metrics = kwargs['ti'].xcom_pull(task_ids='train_models', key='model_metrics')
    print("Model Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")


# Define the load and preprocess task
load_data_task = PythonOperator(
    task_id='load_and_preprocess_data',
    python_callable=load_and_preprocess_data,
    dag=dag,
)

# Define the train models task
train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

# Task to start MLflow server

# Task to print metrics
print_metrics_task = PythonOperator(
    task_id='print_metrics',
    python_callable=print_metrics,
    dag=dag,
)


# Set the task dependency: load_data_task >> train_models_task
load_data_task >> train_models_task >> print_metrics_task


