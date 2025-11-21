import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train_and_log(stage: str = "staging") -> None:
    """
    Train a simple Linear Regression model on the diabetes dataset.
    Log params & metrics to MLflow and save the model to disk.

    stage: "staging" or "production"
    """
    # ----------------------------------------
    # 🔧 0. Set safe MLflow tracking directory (cross-platform)
    # ----------------------------------------
    # This ensures MLflow writes to ./mlruns on BOTH Windows & GitHub Actions (Linux)
    file_dir = os.path.dirname(os.path.abspath(__file__))       # .../src
    mlruns_dir = os.path.abspath(os.path.join(file_dir, "..", "mlruns"))

    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")

    # ----------------------------------------
    # 1. Load data
    # ----------------------------------------
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # ----------------------------------------
    # 2. Train-test split (same for staging & production)
    # ----------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train[:, :-2]
    X_test = X_test[:, :-2]

    # ----------------------------------------
    # 3. Start MLflow run
    # ----------------------------------------
    mlflow.set_experiment("diabetes_staging_vs_prod")
    with mlflow.start_run(run_name=f"diabetes_{stage}"):
        mlflow.log_param("stage", stage)
        mlflow.log_param("model_type", "LinearRegression")

        # ----------------------------------------
        # 4. Train Model
        # ----------------------------------------
        model = LinearRegression()
        model.fit(X_train, y_train)

        # ----------------------------------------
        # 5. Evaluate
        # ----------------------------------------
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)

        print(f"[{stage.upper()}] MSE: {mse:.2f}")

        # ----------------------------------------
        # 6. Log model in MLflow artifacts
        # ----------------------------------------
        mlflow.sklearn.log_model(model, artifact_path="model")

        # ----------------------------------------
        # 7. Save local Production/Staging model
        # ----------------------------------------
        models_dir = os.path.abspath(os.path.join(file_dir, "..", "models"))
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, f"diabetes_{stage}_model.pkl")
        joblib.dump(model, model_path)

        print(f"[{stage.upper()}] Saved model to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["staging", "production"],
        default="staging",
        help="Which stage to train for: 'staging' or 'production'",
    )
    args = parser.parse_args()

    train_and_log(stage=args.stage)
