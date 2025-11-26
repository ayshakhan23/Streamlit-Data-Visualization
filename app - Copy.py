import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# ML Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

st.set_page_config(page_title="Advanced ML App", layout="wide")

# --------------------------------
# Load Data
# --------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

st.title("📊 Advanced Data Analysis & ML Prediction Dashboard")

# --------------------------------
# Show Data
# --------------------------------
st.header("🔍 Dataset Preview")
st.dataframe(df)

# --------------------------------
# Data Summary
# --------------------------------
st.header("📌 Summary")
c1, c2 = st.columns(2)

with c1:
    st.write("Shape:", df.shape)
with c2:
    st.write("Missing Values:", df.isnull().sum())

st.subheader("Statistics")
st.write(df.describe())

# --------------------------------
# Visualizations
# --------------------------------
st.header("📈 Visualizations")

numeric_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(exclude='number').columns

# Histogram
st.subheader("Histogram")
if len(numeric_cols) > 0:
    col_hist = st.selectbox("Select numeric column", numeric_cols)
    fig, ax = plt.subplots()
    ax.hist(df[col_hist], bins=20)
    st.pyplot(fig)

# Boxplot
st.subheader("Box Plot")
if len(numeric_cols) > 0:
    col_box = st.selectbox("Select column for box plot", numeric_cols)
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col_box], ax=ax)
    st.pyplot(fig)

# Bar chart for categorical
st.subheader("Bar Chart")
if len(cat_cols) > 0:
    col_bar = st.selectbox("Select categorical column", cat_cols)
    fig, ax = plt.subplots()
    df[col_bar].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
if len(numeric_cols) > 1:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    st.pyplot(fig)

# --------------------------------
# ML Section
# --------------------------------
st.header("🤖 Machine Learning Prediction")

target = st.selectbox("Choose Target Column", df.columns)

X = df.drop(columns=[target])
y = df[target]

# Encode categorical
for col in X.select_dtypes(include='object'):
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Determine if task is classification or regression
if y.dtype == 'object' or len(y.unique()) <= 10:
    problem_type = "classification"
else:
    problem_type = "regression"

# Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numeric
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Select model
st.subheader("Choose a Model")

if problem_type == "classification":
    model_name = st.selectbox("Model", [
        "Logistic Regression",
        "Random Forest Classifier",
        "Decision Tree Classifier",
        "KNN Classifier",
        "SVM Classifier"
    ])
else:
    model_name = st.selectbox("Model", [
        "Linear Regression",
        "Random Forest Regressor",
        "Decision Tree Regressor",
        "KNN Regressor",
        "SVM Regressor"
    ])

# Model selection
def get_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=250)
    if name == "Random Forest Classifier":
        return RandomForestClassifier()
    if name == "Decision Tree Classifier":
        return DecisionTreeClassifier()
    if name == "KNN Classifier":
        return KNeighborsClassifier()
    if name == "SVM Classifier":
        return SVC()

    if name == "Linear Regression":
        return LinearRegression()
    if name == "Random Forest Regressor":
        return RandomForestRegressor()
    if name == "Decision Tree Regressor":
        return DecisionTreeRegressor()
    if name == "KNN Regressor":
        return KNeighborsRegressor()
    if name == "SVM Regressor":
        return SVR()

model = get_model(model_name)
model.fit(x_train, y_train)
pred = model.predict(x_test)

# --------------------------------
# Model Evaluation
# --------------------------------
st.subheader("📊 Model Performance")

if problem_type == "classification":
    acc = accuracy_score(y_test, pred)
    st.success(f"Accuracy: **{acc:.4f}**")
else:
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    st.success(f"MSE: **{mse:.4f}**, R² Score: **{r2:.4f}**")

# --------------------------------
# Feature Importance (if available)
# --------------------------------
if model_name.startswith("Random Forest"):
    st.subheader("⭐ Feature Importance")
    fig, ax = plt.subplots(figsize=(6, 4))
    importance = model.feature_importances_
    plt.bar(X.columns, importance)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --------------------------------
# Prediction Form
# --------------------------------
st.header("🔮 Predict New Value")

user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"Enter {col}", 0.0)

if st.button("Predict"):
    df_user = pd.DataFrame([user_input])
    df_user = scaler.transform(df_user)
    result = model.predict(df_user)
    st.success(f"Prediction: **{result[0]}**")
