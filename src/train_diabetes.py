import argparse
import os

import joblib
import mlflow
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Simple quality gate threshold for staging
MSE_THRESHOLD = 3000.0


def train_and_log(stage: str = "staging") -> None:
    """
    Train a simple Linear Regression model on the diabetes dataset.
    Log params & metrics to MLflow and save the model to disk.

    stage: "staging" or "production"
    """

    # 1. Load data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # 2. Train-test split (same for both stages, to keep things simple)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # X_train = X_train[:, :-2]  # a bug
    # X_test = X_test[:, :-2]  # a bug

    # 3. Configure MLflow
    # Use a local folder in both local & CI runs
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes_demo")

    # Detect if we are running inside GitHub Actions
    is_ci = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

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

        # 6A. Apply quality gate in STAGING
        if stage == "staging":
            if mse <= MSE_THRESHOLD:
                print(
                    f"[STAGING] ✅ Model PASSED quality gate "
                    f"(MSE={mse:.2f} ≤ {MSE_THRESHOLD})"
                )
            else:
                print(
                    f"[STAGING] ❌ Model FAILED quality gate "
                    f"(MSE={mse:.2f} > {MSE_THRESHOLD})"
                )
                # Fail the CI job if staging is bad
                raise SystemExit(1)

        # 6B. Just informational print in PRODUCTION
        if stage == "production":
            print(f"[PRODUCTION] Using model with MSE={mse:.2f}")

        # 7. Log model to MLflow ONLY when running locally
        if not is_ci:
            # This is for local demo in MLflow UI
            mlflow.sklearn.log_model(model, artifact_path="model")
            print("[INFO] Logged model to MLflow (local run).")
        else:
            # In CI, skip this to avoid weird filesystem issues like '/C:' on Linux
            print(
                "[INFO] Running inside GitHub Actions – skipping mlflow.sklearn.log_model "
                "to avoid artifact permission issues."
            )

        # 8. Save model to local 'models/' folder (works both local & CI)
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
