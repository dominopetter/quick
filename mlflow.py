import os
import time
import numpy as np
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import mlflow
import warnings
warnings.filterwarnings("ignore")

def create_experiment(name_prefix="LENOVO_svm-digit-classifier"):
    timestamp = int(time.time())
    username = os.getenv('DOMINO_STARTING_USERNAME', 'default_user')
    experiment_name = f"{name_prefix}-{username}-{timestamp}"
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Experiment id: {experiment_id}")
    print(f"Experiment name: {experiment_name}")
    return experiment_id

def prepare_data():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return train_test_split(data, digits.target, test_size=0.20, shuffle=False)

def log_metrics_and_params(predicted, test_y, svc_params):
    report = metrics.classification_report(test_y, predicted, output_dict=True)
    mlflow.log_params(svc_params)
    for metric in ("precision", "recall", "f1-score"):
        for digit in range(10):
            mlflow.log_metric(f"{metric}_{digit}", report[str(digit)][metric])
        mlflow.log_metric(f"{metric}_macro_avg", report['macro avg'][metric])
        mlflow.log_metric(f"{metric}_weighted_avg", report['weighted avg'][metric])
    mlflow.log_metric("accuracy", report["accuracy"])

def run_svm_classifier(train_x, train_y, test_x, test_y, C, gamma):
    svc_params = {"kernel": "rbf", "C": C, "gamma": gamma, "random_state": 42}
    classifier = svm.SVC(**svc_params).fit(train_x, train_y)
    predicted = classifier.predict(test_x)
    log_metrics_and_params(predicted, test_y, svc_params)
    mlflow.sklearn.log_model(classifier, "model")

def main():
    experiment_id = create_experiment()
    train_x, test_x, train_y, test_y = prepare_data()

    # Define a small set of hyperparameters for demonstration.
    parameters = [(C, gamma) for C in [0.1, 1, 10] for gamma in [0.001, 0.01]]

    for C, gamma in parameters:
        with mlflow.start_run(experiment_id=experiment_id):
            print(f"Running SVM classifier with C={C}, gamma={gamma}")
            run_svm_classifier(train_x, train_y, test_x, test_y, C, gamma)

if __name__ == "__main__":
    main()