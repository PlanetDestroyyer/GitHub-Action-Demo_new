import os
import re
import pickle
import json
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mlflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

# Airflow DAG configuration
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 8, 12),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'hcpcs_classification',
    default_args=default_args,
    description='HCPCS Classification Workflow',
    schedule_interval=timedelta(days=1),
)

def get_temp_file_path(filename):
    temp_dir = Path("/tmp/hcpcs_classification")
    temp_dir.mkdir(parents=True, exist_ok=True)
    return str(temp_dir / filename)


def load_and_split_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        
        last_1000_rows = df.tail(300)
        remaining_rows = df.iloc[:-300]
        combined_df = pd.concat([remaining_rows, last_1000_rows], ignore_index=True)
        
        
        combined_df['Description'] = combined_df['Description'].str.lower()
        combined_df['Description'] = combined_df['Description'].str.replace('[^\w\s]', '', regex=True)
        
        X = combined_df["Description"].tolist()
        y = combined_df["Code"].tolist()
        
        
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        
        
        label_encoder_path = get_temp_file_path("label_encoder.pkl")
        with open(label_encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Save split data
        np.save(get_temp_file_path("X_train.npy"), X_train)
        np.save(get_temp_file_path("X_test.npy"), X_test)
        np.save(get_temp_file_path("y_train.npy"), y_train)
        np.save(get_temp_file_path("y_test.npy"), y_test)
        
        return "Data split and saved successfully"
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please check the file path.")
        return None

def vectorize_data():
    X_train = np.load(get_temp_file_path("X_train.npy"), allow_pickle=True)
    X_test = np.load(get_temp_file_path("X_test.npy"), allow_pickle=True)

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_vectorized)
    X_test_scaled = scaler.transform(X_test_vectorized)

    vectorizer_path = get_temp_file_path("vectorizer.pkl")
    scaler_path = get_temp_file_path("scaler.pkl")
    
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save vectorized and scaled data
    np.save(get_temp_file_path("X_train_scaled.npy"), X_train_scaled.toarray())
    np.save(get_temp_file_path("X_test_scaled.npy"), X_test_scaled.toarray())

    return "Data vectorized, scaled, and saved successfully"

def train_and_evaluate_models():
    X_train = np.load(get_temp_file_path("X_train_scaled.npy"))
    X_test = np.load(get_temp_file_path("X_test_scaled.npy"))
    y_train = np.load(get_temp_file_path("y_train.npy"))
    y_test = np.load(get_temp_file_path("y_test.npy"))

    classifiers = {
        'SVC': (SVC(random_state=42), {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'gamma': ['scale', 'auto']}),
        'KNeighborsClassifier': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance']}),
        'DecisionTreeClassifier': (DecisionTreeClassifier(random_state=42), {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30]}),
        'RandomForestClassifier': (RandomForestClassifier(random_state=42), {'n_estimators': [100, 200, 300], 'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30]})
    }
    best_accuracy = 0
    best_classifier_name = ""
    best_classifier = None
    mlflow.set_experiment("HCPCS_Classification")
    with mlflow.start_run(run_name="Model_Comparison"):
        for name, (clf, params) in classifiers.items():
            print(f"Training {name}...")
            grid_search = GridSearchCV(clf, params, cv=2, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            y_pred = grid_search.best_estimator_.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_params({f"{name}_{key}": val for key, val in grid_search.best_params_.items()})
            mlflow.log_metric(f"{name}_accuracy", test_accuracy)
            mlflow.log_metric(f"{name}_precision", precision_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric(f"{name}_recall", recall_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric(f"{name}_f1_score", f1_score(y_test, y_pred, average='weighted'))
            classifier_path = get_temp_file_path(f"{name}_classifier.pkl")
            with open(classifier_path, "wb") as f:
                pickle.dump(grid_search.best_estimator_, f)
            mlflow.log_artifact(classifier_path)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_classifier_name = name
                best_classifier = grid_search.best_estimator_
            print(f"{name}:")
            print(f"  Test accuracy: {test_accuracy*100:.2f}%")
            print(f"  Precision: {precision_score(y_test, y_pred, average='weighted')*100:.2f}%")
            print(f"  Recall: {recall_score(y_test, y_pred, average='weighted')*100:.2f}%")
            print(f"  F1 Score: {f1_score(y_test, y_pred, average='weighted')*100:.2f}%")
            print(f"  Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
            print()
        
        print(f"\nBest Model: {best_classifier_name}")
        print(f"Best Accuracy: {best_accuracy*100:.2f}%")
        
        # Feature importance for the best model (if applicable)
        if hasattr(best_classifier, 'feature_importances_'):
            feature_importance = best_classifier.feature_importances_
            mlflow.log_param("feature_importance", feature_importance.tolist())
            print("Top 10 Most Important Features:")
            for i in np.argsort(feature_importance)[-10:]:
                print(f"Feature {i}: {feature_importance[i]:.4f}")
    
    # Save best model info
    with open(get_temp_file_path("best_model_info.json"), "w") as f:
        json.dump({"name": best_classifier_name, "accuracy": best_accuracy}, f)

    return "Models trained, evaluated, and best model saved successfully"

def find_matching_description(input_str, descriptions):
    matches = {desc: re.search(re.escape(desc), input_str) for desc in descriptions}
    return {desc: match.group() for desc, match in matches.items() if match}

def predict_hcpcs_codes(input_string):
    # Load best model info
    with open(get_temp_file_path("best_model_info.json"), "r") as f:
        best_model_info = json.load(f)
    
    best_classifier_name = best_model_info["name"]
    best_accuracy = best_model_info["accuracy"]

    best_classifier_path = get_temp_file_path(f"{best_classifier_name}_classifier.pkl")
    with open(best_classifier_path, "rb") as f:
        best_classifier = pickle.load(f)
    
    label_encoder_path = get_temp_file_path("label_encoder.pkl")
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    
    vectorizer_path = get_temp_file_path("vectorizer.pkl")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    scaler_path = get_temp_file_path("scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Preprocess input string
    input_string = input_string.lower()
    input_string = re.sub(r'[^\w\s]', '', input_string)
    
    # Vectorize the input string
    X_input = vectorizer.transform([input_string])
    
    # Convert sparse matrix to dense array
    X_input_dense = X_input.toarray()
    
    # Scale the dense input
    X_input_scaled = scaler.transform(X_input_dense)
    
    # Predict the HCPCS code
    prediction = best_classifier.predict(X_input_scaled)
    hcpcs_code = label_encoder.inverse_transform(prediction)[0]
    
    print(f"Predicted HCPCS code: {hcpcs_code} with model {best_classifier_name} (Accuracy: {best_accuracy*100:.2f}%")

input_string = """Onc thyr 10 microrna seq alg Alright, she is here for an annual visit, has obesity, hypertension, hyperlipidemia and this is also a well visit. Hi, good morning. How are you today? Hey, good morning. Alright, nice haircut. Nice, I guess.
    Well, it looks good. You've got a big day today or not too bad. Not too bad, only 4 or 5. Oh, that's not bad. Something's doable. You're on a CPAP and is it working good? Yeah. Do you know if you miss a night?
    I mean, can you like, do you ever miss a night or you can tell? Oh yeah, I have. I almost can't sleep without it. Alright, yeah, that's what I've heard and so you should continue using that and then okay, so you're getting your pilot's license.
    What else have you been up to? Well, we had our trip to Southeast Asia. Oh yeah, how was it? It was great. Cool. Mm -hmm. We had time. I share some pictures. You can maybe show me a few. We might have time.
    You're getting over cold. Is that what's going on? I've had a rough month after Christmas I had when I'm assuming it was Norway where it's like 4 days our time is tough. Because I got a little better than that the weekend after New Year's taking over our garbage into the town hall.
    I slept and fell. Oh, yes. How's that? Lended flat on my back, busted my head off. Oh, that's bad. And my back still is horrible. Did you need staples or? No, it wasn't that bad just enough that when you rub it and oh, hands bloody.
    Alright, that's not good. But yeah, then then it hits you. Yeah, and then so probably about 10 or 10 -ish. Mm -hmm. Okay. I get some muscle spasm there and the pain comes all over the way around the ribs to the front.
    Is it still happening? Yeah, but it's much better than it was. Mm -hmm. Mm -hmm. I can actually get up and move and breathe around. Oh, does it interfere with your sleep at all or no? No, not really.
    Mm -hmm. But I think I have sinus infection currently because I've had this head cold for like 10 days and now you know. there's actually blood in tinging so it's not okay we can treat that yeah just take the whole 10 days because of the medicine because you know there's not a lot of blood vessels in your sinus so we gotta weld up the quantity exactly and and you know only fight the surface and you have bugs in the center it's a tough one to stay in the center right and so natural irrigation is a key part of the treatment so yeah that's not happening yeah all right so you know all this all right and then we'll tap on your sinus as well very good and then are you as healthy outside of being ill are you as healthy as you want to be i mean i could love some weight and be in better physical shape okay but it's less time for that okay so you're not completely satisfied but you're okay with where you are at and that sounds fine but movements is movement is very important along with a healthy diet i know you like project -based movement right yeah you have to you like to have something that the movement is accomplishing if you will yeah split a little five with other day with sledge but Kind of thing, you know, yeah, and I'm sending these downstairs.
    No, no, they're antibiotic. Sure Yes, you change the you know changes the smell of your urine Okay, then try I can't smell anything Anyways, try it again, you know once you're done with the antibiotic try to throw biotics in the form of plain yogurt If you like, so I love some maybe some Maybe some hot turmeric and potato salad that's about as well.
    Sure I mean, that's not bad for you. But yeah, that won't have the pre or probiotic, but it's okay the yogurt Yeah, maybe some hot turmeric and cream cream cheese Mm -hmm. Yeah cream cheese. Absolutely.
    You know in the same family as Soccer and the sense of the way it's prepared, but I'll look again chief. Do you do you didn't you go to South Korea though? No, mm -hmm All right. So you're really seen April 20 mg is working fine.
    Are you facing any side effects of it? No, it's working. Well, and I also started to monitor my blood pressure at home and it's a reduce of 125 to 85 That's very good So you can continue to take the same dose for now and your blood pressure and the office today is also under control Okay, what about the cholesterol medication?
    It seems fine. I'm taking 40 mg dosage of it okay, let me check your last lipid panel and Mm -hmm your HDL is 38 which is under normal range and your LDL is 90 cool So you can continue taking it to our state in with the same dose.
    Okay. Oh, that's great. You were what Thailand? Cool thing about you in Hong Kong for a few days Mm -hmm, that's safe way Taiwan and then three steps in Japan. Wow, that's right. Very good And did you get to a beach in Singapore?"""


# Airflow tasks
load_and_split_data_task = PythonOperator(
    task_id='load_and_split_data_task',
    python_callable=load_and_split_data,
    op_kwargs={'file_path': 'data/data.csv'},
    dag=dag,
)

vectorize_data_task = PythonOperator(
    task_id='vectorize_data_task',
    python_callable=vectorize_data,
    dag=dag,
)

train_and_evaluate_models_task = PythonOperator(
    task_id='train_and_evaluate_models_task',
    python_callable=train_and_evaluate_models,
    dag=dag,
)

predict_hcpcs_codes_task = PythonOperator(
    task_id='predict_hcpcs_codes_task',
    python_callable=predict_hcpcs_codes,
    op_kwargs={'input_string':input_string},
    dag=dag
)

load_and_split_data_task >> vectorize_data_task >> train_and_evaluate_models_task >> predict_hcpcs_codes_task
