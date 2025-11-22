# staging_vs_prod

This repository demonstrates the workflow to create a simple "staging vs prod" ML project structure, create a virtual environment, install dependencies, train a model in staging, run a prediction script, and initialize a git repository.

**Folder Structure**
- `src`: Application source code (examples: `train_diabetes.py`, `predict_diabetes.py`).
- `models`: Trained model artifacts (created by training script or MLflow artifacts).

**Setup & Workflow**
Follow these steps in order to set up and run the project (Windows / PowerShell commands):

1. Create the folder structure (if not already present):

```
mkdir src
mkdir models
```

2. Create a Python virtual environment named `venv1`:

```
python -m venv venv1
```

3. Activate the virtual environment (PowerShell):

```
venv1\Scripts\Activate.ps1
# or, if using the call operator to run the script explicitly:
& .\venv1\Scripts\Activate.ps1
```

4. Install required packages:

```
pip install -r requirements.txt
```

5. Train the model in the `staging` stage (example):

```
python src/train_diabetes.py --stage=staging
```

6. Run the prediction script:

```
python src/predict_diabetes.py
```

7. Initialize a new git repository and push to a remote (replace `<username>` and `<repo>`):

```
git init
git add .
git commit -m "Initial commit"
git remote remove origin
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

**Notes**
- The activation command above is for Windows PowerShell. If you use Command Prompt (cmd.exe) activate with `venv1\Scripts\activate.bat`.
- If `git remote remove origin` errors because no remote exists, it is safe to ignore the error and run the `git remote add origin ...` step.
- Replace `https://github.com/<username>/<repo>.git` with your actual GitHub repository URL.
