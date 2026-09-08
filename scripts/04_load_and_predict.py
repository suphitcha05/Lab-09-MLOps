import mlflow
from sklearn.datasets import load_breast_cancer

def load_and_predict():
    """
    Loads the cancer classifier using the staging alias
    and predicts the first sample from each class.
    """

    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"

    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")

    # Load the model from the Model Registry using Alias URI
    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(
            f"Please make sure a model version has the alias "
            f"'@{MODEL_ALIAS}' in the MLflow UI."
        )
        return

    # Load Breast Cancer dataset
    X, y = load_breast_cancer(
        return_X_y=True,
        as_frame=True,
    )

    # Class names
    class_names = {
        0: "malignant",
        1: "benign",
    }

    # Select the first sample from each class
    sample_indices = [
        y[y == 0].index[0],
        y[y == 1].index[0],
    ]

    sample_data = X.loc[sample_indices]
    actual_labels = y.loc[sample_indices]

    # Make predictions
    predictions = model.predict(sample_data)

    # Display results
    print("-" * 50)
    print("Prediction Results")
    print("-" * 50)

    for actual, predicted in zip(actual_labels, predictions):
        actual_name = class_names[int(actual)]
        predicted_name = class_names[int(predicted)]
        correct = int(actual) == int(predicted)

        print(
            f"Actual: {actual_name:<10} "
            f"Predicted: {predicted_name:<10} "
            f"Correct: {correct}"
        )

    print("-" * 50)
 
if __name__ == "__main__":
    load_and_predict()

