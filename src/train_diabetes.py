import argparse
import os
import sys

import joblib
import mlflow
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 👉 simple quality threshold for demo
STAGING_MSE_THRESHOLD = 3000  # change this to be stricter/looser


def train_and_log(stage: str = "staging") -> None:
    """
    Train a simple Linear Regression model on the diabetes dataset.
    Log params & metrics to MLflow and save the model to disk.

    stage: "staging" or "production"
    """

    # 1. Load data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Start MLflow run
    mlflow.set_experiment("diabetes_demo")
    with mlflow.start_run(run_name=f"diabetes_{stage}"):
        mlflow.log_param("stage", stage)
        mlflow.log_param("model_type", "LinearRegression")

        # 4. Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 5. Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)

        print(f"[{stage.upper()}] MSE: {mse:.2f}")

        # 6. SIMPLE QUALITY GATE FOR STAGING
        if stage == "staging":
            print(
                f"[STAGING] Checking quality gate: "
                f"MSE <= {STAGING_MSE_THRESHOLD}"
            )
            if mse > STAGING_MSE_THRESHOLD:
                print(
                    f"[STAGING] ❌ Model FAILED quality gate "
                    f"(MSE={mse:.2f} > {STAGING_MSE_THRESHOLD})"
                )
                # 🔴 IMPORTANT: non-zero exit → GitHub Action step fails
                sys.exit(1)
            else:
                print(
                    f"[STAGING] ✅ Model PASSED quality gate "
                    f"(MSE={mse:.2f} ≤ {STAGING_MSE_THRESHOLD})"
                )

        elif stage == "production":
            print(
                "[PRODUCTION] Skipping quality gate here "
                "(we assume staging already checked it)."
            )

        # 7. Log model to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        # 8. Save model to local 'models/' folder
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"diabetes_{stage}_model.pkl")
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
