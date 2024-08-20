import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime , timedelta
import pandas as pd
import numpy as np
import logging
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
import mlflow
from mlflow import log_metric, log_param, log_artifact

# Set up logging
logging.basicConfig(filename='loan_approval.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'loan_approval_prediction_new',
    default_args=default_args,
    description='Loan Approval Prediction ',
    schedule_interval='@daily',
    start_date=datetime(2024, 8, 18),
    catchup=False,
)

def load_and_preprocess_data():
    # Data Loading
    data = pd.read_csv("/home/pranav/airflow/LoanApprovalPrediction.csv")
    logging.info("Data loaded successfully")

    # Drop unnecessary columns
    data.drop(['Loan_ID'], axis=1, inplace=True)

    # Encode categorical variables
    label_encoder = preprocessing.LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = label_encoder.fit_transform(data[col])
    logging.info("Categorical variables encoded")

    # Fill missing values
    for col in data.columns:
        data[col] = data[col].fillna(data[col].mean())
    logging.info("Missing values filled")

    return data
def train_and_evaluate_models(**kwargs):
    data = kwargs['ti'].xcom_pull(task_ids='load_data')

    # Split the data
    X = data.drop(['Loan_Status'], axis=1)
    Y = data['Loan_Status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
    logging.info(f"Data split into training and test sets with shapes {X_train.shape} and {X_test.shape}")

    # Initialize models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "SVC": SVC(),
        "LogisticRegression": LogisticRegression()
    }

    results = {}
    best_model = None
    best_accuracy = 0
    best_model_mlflow_run_id = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            model.fit(X_train, Y_train)
            Y_pred_test = model.predict(X_test)
            
            accuracy = metrics.accuracy_score(Y_test, Y_pred_test)
            f1_score = metrics.f1_score(Y_test, Y_pred_test)
            recall = metrics.recall_score(Y_test, Y_pred_test)
            
            mlflow.log_metric("Accuracy", accuracy * 100)
            mlflow.log_metric("F1 Score", f1_score * 100)
            mlflow.log_metric("Recall", recall * 100)
            
            mlflow.log_params(model.get_params())
            
            mlflow.sklearn.log_model(model, model_name)
            
            # Check if the artifact file exists before logging
            req_file = f"{model_name}_requirements.txt"
            if os.path.exists(req_file):
                mlflow.log_artifact(req_file)
            else:
                logging.warning(f"Artifact file {req_file} does not exist.")
            
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                feature_importance_file = f"{model_name}_feature_importance.csv"
                feature_importance.to_csv(feature_importance_file, index=False)
                mlflow.log_artifact(feature_importance_file)
            
            results[model_name] = {"accuracy": accuracy, "f1_score": f1_score, "recall": recall}
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
                best_model_mlflow_run_id = run.info.run_id

    with mlflow.start_run(run_name="Best Model"):
        mlflow.log_param("Best Model", best_model)
        mlflow.log_metric("Best Accuracy", best_accuracy * 100)
        mlflow.set_tag("Best Model Run ID", best_model_mlflow_run_id)
        
        summary_df = pd.DataFrame(results).T
        summary_df.to_csv("model_summary.csv")
        mlflow.log_artifact("model_summary.csv")
  
    return results, best_model


load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_and_preprocess_data,
    provide_context=True,
    dag=dag,
)

train_models = PythonOperator(
    task_id='train_and_evaluate',
    python_callable=train_and_evaluate_models,
    provide_context=True,
    dag=dag,
)

load_data >> train_models
