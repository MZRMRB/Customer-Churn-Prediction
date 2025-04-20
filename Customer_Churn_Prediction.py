import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ------------------------------
# Load and cache data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# ------------------------------
# Preprocess data
def preprocess(df):
    df = df.copy()
    drop_cols = ['customerID', 'Phone', 'phone number']
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].replace({True: 1, False: 0, 'Yes': 1, 'No': 0})
        df['Churn'] = df['Churn'].astype(int)

    object_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=object_cols, drop_first=True)

    return df

# ------------------------------
# EDA
def show_eda(df):
    with st.expander("📊 Exploratory Data Analysis (Click to Expand)"):
        st.write("**Preview of Dataset**")
        st.dataframe(df.head())

        st.write("**Missing Values Per Column**")
        st.write(df.isnull().sum())

        if 'Churn' in df.columns:
            st.write("**Churn Distribution**")
            st.bar_chart(df['Churn'].value_counts())

        if 'tenure' in df.columns:
            st.write("**Tenure vs Churn**")
            fig, ax = plt.subplots()
            sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', ax=ax)
            st.pyplot(fig)

        st.write("**Correlation Matrix**")
        numeric_df = df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig)

# ------------------------------
# Train Model
def train_model(model_name, X_train, y_train, params):
    if model_name == "Logistic Regression":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train_scaled, y_train)
        return model, scaler

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=params.get("max_depth", 5))
        model.fit(X_train, y_train)
        return model, None

    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5)
        )
        model.fit(X_train, y_train)
        return model, None

# ------------------------------
# Evaluate Model
def evaluate_model(model, X_test, y_test, scaler=None):
    if scaler:
        X_test = scaler.transform(X_test)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("📈 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{prec:.4f}")
    col3.metric("Recall", f"{rec:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    st.subheader("📌 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5, linecolor='black')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

# ------------------------------
# Main Application
def main():
    st.set_page_config(page_title="Customer Churn Predictor", layout="wide", page_icon="📉")
    st.title("📉 Customer Churn Prediction App")
    st.markdown("⚙️ Use ML models to predict whether a customer will churn based on features.")

    # Model selection and parameters
    st.sidebar.header("🧠 Choose Model & Parameters")
    model_name = st.sidebar.selectbox(
        "Select ML Model", 
        ["Logistic Regression", "Decision Tree", "Random Forest"],
        help="Choose an ML model to train on the churn dataset."
    )

    params = {}
    if model_name in ["Decision Tree", "Random Forest"]:
        params["max_depth"] = st.sidebar.slider("Max Tree Depth", 1, 20, 5)
    if model_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("Number of Trees", 10, 200, 100)

    # Load data
    train_df = load_data("churn-bigml-80.csv")
    test_df = load_data("churn-bigml-20.csv")

    # Show raw data
    if st.checkbox("📂 Show Raw Training Data"):
        st.subheader("Raw Training Data")
        st.dataframe(train_df)

    # EDA Section
    show_eda(train_df)

    # Preprocess data
    X_train = preprocess(train_df.drop("Churn", axis=1))
    y_train = train_df["Churn"]
    X_test = preprocess(test_df.drop("Churn", axis=1))
    y_test = test_df["Churn"]

    # Align columns between train and test
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X_train.columns]

    # Train and evaluate model
    model, scaler = train_model(model_name, X_train, y_train, params)
    evaluate_model(model, X_test, y_test, scaler)

if __name__ == "__main__":
    main()