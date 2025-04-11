import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path="data/Telco_customer_churn.csv"):
    df = pd.read_csv(path);
    return df;

def preprocess_data(df):
    # Drop customerID (not useful for modeling)
    df = df.drop("CustomerID", axis=1)

    # Reset index (important to keep X and y in sync)
    df = df.reset_index(drop=True)

    # Separate features and target
    X = df.drop('Churn Value', axis=1)
    y = df['Churn Value']

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y


def get_train_test_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

