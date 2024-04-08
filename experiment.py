#!/usr/bin/env python
# coding: utf-8

import os
from copy import copy
from itertools import product
import pprint
import shutil
import time
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# create an experiment in MLflow
# we'll make the name unique in the project by appending a timestamp so that you and other users can run this cell more than once.
timestamp = time.time()
username = os.environ['DOMINO_STARTING_USERNAME']
experiment_name = f"Demo-svm-digit-classifier-{timestamp}"

# below, we'll use the returned experiment_id in calls to mlflow.start_run() to add data to the experiment.
experiment_id = mlflow.create_experiment(experiment_name)
print(f"Experiment id: {experiment_id}")
print(f"Experiment name: {experiment_name}")

# prepare the data
# this dataset has handwritten roman numerals represented as 8x8 pixel grayscale images
digits = datasets.load_digits()

# flatten each 2-D array of grayscale values from shape (8, 8) into shape (64,)
# data will have shape (n_samples = number of images, n_features = total number of pixels in each image)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# split the data into train and test subsets
train_x, test_x, train_y, test_y = train_test_split(
    data, digits.target, test_size=0.20, shuffle=False
)

# define a function that uses an SVM to classify the digits while recording relevant info as an MLflow run 
# within the given experiment.
def create_run_svm_classify(
    experiment_id: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    random_seed: int,
    svc_param_kwargs: Dict = None,
    run_name: str = None,
):
    """
    Record an MLflow run for experiment_id.
    Trains a Support Vector Machine classifier using train_x, train_y, and the given svm_param_kwargs.
    Then, predicts using test_x and test_y and logs params, metrics, artifacts, and model to MLflow.

    :param str experiment_id: id of the MLflow experiment in which to record the run
    :param np.ndarray train_x: x values for the training data
    :param np.ndarray train_y: y values for the training data
    :param np.ndarray test_x: x values for the test data
    :param np.ndarray test_y: y values for the test data
    :param int random_seed: used to set the random seed for all random number generators in this run 
        (e.g. numpy seed, sklearn model random state, etc.)
    :param Dict svc_param_kwargs: kwargs to use when creating the sklearn.svm.SVC (e.g. C, kernel, etc.)
    :param str run_name: name for the run in MLflow. If None, MLflow will generate a random name
    """
    if random_seed is None:
        random_seed = 42
    svc_param_kwargs = svc_param_kwargs or {}
    svc_param_kwargs["random_state"] = random_seed

    np.random.seed(random_seed)

    # passing experiment_id tells MLflow to associate the run data with the correct experiment.
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        if run_name is None:
            run_name = run.info.run_name
        pretty_params = pprint.pformat(svc_param_kwargs, width=1, indent=2)
        print("******************************************")
        print("******************************************")
        print(f"Starting run {run_name} in experiment {experiment_id} with SVC params:")
        print(f"{pretty_params}")

        print("Initializing model...")
        classifier = svm.SVC(**svc_param_kwargs)

        print("Training model...")
        classifier.fit(train_x, train_y)

        print("Predicting using trained model...")
        predicted = classifier.predict(test_x)
        prediction_report_str = metrics.classification_report(test_y, predicted, digits=3, output_dict=False)
        prediction_report_dict = metrics.classification_report(test_y, predicted, output_dict=True)

        # print some info about the run and also save as file for an MLflow artifact
        print("Finished predictions.")
        run_overview = (
            f"Run: {run_name}\n"
            f"Random seed: {random_seed}\n"
            "Classifier type: sklearn.svm.SVC\n"
            "Specified classifier params:\n"
            f"{pretty_params}"
            "\n"
            "Classification report:\n"
            f"{prediction_report_str}\n"
        )
        print(f"{run_overview}")
        run_overview_file_name = "run_overview.txt"
        with open(run_overview_file_name, "w") as f:
            f.write(run_overview)

        # visualize model performance and save visualization for an MLflow artifact
        output_dir = "/mnt/artifacts/results"
        disp = metrics.ConfusionMatrixDisplay.from_predictions(test_y, predicted)
        disp.figure_.suptitle(f"Confusion Matrix - Run {run_name}")
        confusion_matrix_file_name = f"confusion_matrix_run_{run_name}.png"
        output_path = os.path.join(output_dir, confusion_matrix_file_name)
        plt.savefig(output_path)
        plt.show()

        print("Recording run params, metrics, and artifacts to MLflow...")
        # record notable sklearn model params and performance metrics.
        # this enables visual comparison via the Experiments UI.
        for param, val in svc_param_kwargs.items():
            mlflow.log_param(param, val)
        # record metrics for each individual digit as well as overall averages
        for metric in ("precision", "recall", "f1-score"):
            for digit in {str(num) for num in range(0, 10)}:
                mlflow.log_metric(f"{metric}_{digit}", prediction_report_dict[digit][metric])
            for avg_type in ("macro", "weighted"):
                mlflow.log_metric(f"{metric}_{avg_type}_avg", prediction_report_dict[f"{avg_type} avg"][metric])
        mlflow.log_metric("overall_accuracy", prediction_report_dict["accuracy"])

        # record the data for reproducibility
        data_dir_name = "run_data"
        os.mkdir(data_dir_name)
        for data, data_name in ((train_x, "train_x"), (train_y, "train_y"), (test_x, "test_x"), (test_y, "test_y")):
            with open(os.path.join(data_dir_name, f"{data_name}.csv"), "w") as data_file:
                np.savetxt(data_file, data, delimiter=",")
        mlflow.log_artifact(data_dir_name)
        shutil.rmtree(data_dir_name)

        # record human-readable/interpretable overview and figures
        mlflow.log_artifact(run_overview_file_name)
        #mlflow.log_artifact(confusion_matrix_file_name)
        mlflow.log_artifact(output_path)

        # record the trained sklearn model
        mlflow.sklearn.log_model(classifier, "model")
        print("Finishing recording.")

# perform a hyperparam sweep to find the best regularization and gamma for an SVC with a radial basis function kernel
random_seed = 470
regularizations = [0.1, 1.0, 10.0, 100.0]
gammas = [0.001, 0.01, 0.1, 1, 10]
for regularization, gamma in product(regularizations, gammas):
    svc_param_kwargs = {
        "kernel": "rbf",
        "C": regularization,
        "gamma": gamma,
    }
    # each call creates a "run" in MLflow and records metrics/artifacts about that run
    create_run_svm_classify(
        experiment_id,
        train_x,
        train_y,
        test_x,
        test_y,
        random_seed,
        svc_param_kwargs=svc_param_kwargs,
        run_name=f"C_{regularization}_gamma_{gamma}",
    )

print("You created your first experiment!")
print(f"Go to the Experiments UI and click on the {experiment_name} experiment to compare runs.")
print("You can also explore and share run artifacts (e.g. overview and confusion matrix).")