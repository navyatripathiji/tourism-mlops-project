
import os
import pandas as pd

from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HF_USERNAME = "navtri12"
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"

# Download dataset from Hugging Face
data_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="tourism.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)

# Load dataset
df = pd.read_csv(data_path)

print("Dataset loaded successfully.")
print("Shape:", df.shape)

# Separate features and target
X = df.drop(
    columns=["ProdTaken", "CustomerID"]
)

y = df["ProdTaken"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# Identify categorical and numerical features
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("Categorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Fit preprocessing only on training data
preprocessor.fit(X_train)

# Transform train and test data
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing completed successfully.")

# Save processed datasets
os.makedirs("tourism_project/data", exist_ok=True)

X_train_processed_df = pd.DataFrame(
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed
)

X_test_processed_df = pd.DataFrame(
    X_test_processed.toarray()
    if hasattr(X_test_processed, "toarray")
    else X_test_processed
)

X_train_processed_df["ProdTaken"] = y_train.reset_index(drop=True)
X_test_processed_df["ProdTaken"] = y_test.reset_index(drop=True)

X_train_processed_df.to_csv(
    "tourism_project/data/train_processed.csv",
    index=False
)

X_test_processed_df.to_csv(
    "tourism_project/data/test_processed.csv",
    index=False
)

print("Processed datasets saved successfully.")
