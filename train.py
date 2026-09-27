
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from huggingface_hub import hf_hub_download, HfApi, create_repo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


HF_USERNAME = "navtri12"
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"
MODEL_REPO = f"{HF_USERNAME}/tourism-model"


# Download dataset
data_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="tourism.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)

df = pd.read_csv(data_path)

print("Dataset loaded successfully.")
print("Shape:", df.shape)


# Features and target
X = df.drop(
    columns=["ProdTaken", "CustomerID"]
)

y = df["ProdTaken"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Identify columns
cat_cols = X_train.select_dtypes(
    include="object"
).columns.tolist()

num_cols = X_train.select_dtypes(
    include="number"
).columns.tolist()


# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            num_cols
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_cols
        )
    ]
)


# Models
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ),

    "gradient_boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# MLflow experiment
mlflow.set_experiment("tourism-package-prediction")

best_score = -1
best_pipeline = None
best_name = None


# Train and track models
for name, clf in models.items():

    with mlflow.start_run(run_name=name):

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(
                y_test,
                preds,
                zero_division=0
            ),
            "recall": recall_score(
                y_test,
                preds,
                zero_division=0
            ),
            "f1": f1_score(
                y_test,
                preds,
                zero_division=0
            )
        }

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            pipe,
            name="model",
            serialization_format="pickle"
        )

        print(f"{name}: {metrics}")

        # Select best model using F1 score
        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_pipeline = pipe
            best_name = name


print(
    f"Best model: {best_name} "
    f"(F1={best_score:.3f})"
)


# Save best model locally
os.makedirs(
    "tourism_project/model_building",
    exist_ok=True
)

model_path = (
    "tourism_project/model_building/"
    "best_model.joblib"
)

joblib.dump(
    best_pipeline,
    model_path
)

print("Model saved:", model_path)


# Upload best model to Hugging Face
api = HfApi(
    token=os.environ["HF_TOKEN"]
)

create_repo(
    repo_id=MODEL_REPO,
    repo_type="model",
    token=os.environ["HF_TOKEN"],
    exist_ok=True
)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_model.joblib",
    repo_id=MODEL_REPO,
    repo_type="model"
)

print(
    f"Model registered at: "
    f"https://huggingface.co/{MODEL_REPO}"
)
