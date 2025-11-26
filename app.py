import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Advanced Data App", layout="wide")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()

st.title("📊 Advanced Data Analysis & Prediction App by Aysha")


# -----------------------------
# Dataset Preview
# -----------------------------
st.header("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Data Summary
# -----------------------------
st.header("📌 Data Summary")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Shape")
    st.write(df.shape)

with col2:
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

st.subheader("Descriptive Stats")
st.write(df.describe())

# -----------------------------
# Data Visualization Section
# -----------------------------
st.header("📈 Data Visualization")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# --- Histogram ---
st.subheader("Histogram")
if len(numeric_cols) > 0:
    selected_hist = st.selectbox("Select a numeric column:", numeric_cols)

    fig, ax = plt.subplots()
    ax.hist(df[selected_hist].dropna())
    ax.set_title(f"Distribution of {selected_hist}")
    st.pyplot(fig)
else:
    st.warning("No numeric columns available for histogram.")

# --- Pairplot ---
st.subheader("Pair Plot")
if st.checkbox("Show Pair Plot (may take time)"):
    if len(numeric_cols) > 1:
        fig = sns.pairplot(df[numeric_cols])
        st.pyplot(fig)
    else:
        st.warning("Need at least two numeric columns for pairplot.")

# --- Correlation Heatmap ---
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=['number'])

if numeric_df.shape[1] > 1:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(numeric_df.corr(), annot=True, ax=ax, cmap="coolwarm")
    st.pyplot(fig)
else:
    st.warning("Not enough numeric columns to show correlation heatmap.")

# -----------------------------
# Machine Learning Section
# -----------------------------
st.header("🤖 Machine Learning Prediction")

target = st.selectbox("Select Target Variable:", df.columns)

# Prepare features & target
X = df.drop(columns=[target])
y = df[target]

# Encode categorical columns
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model
model = LogisticRegression(max_iter=200)
model.fit(x_train_scaled, y_train)
pred = model.predict(x_test_scaled)

acc = accuracy_score(y_test, pred)

st.subheader("Model Accuracy")
st.write(f"✔ Accuracy: **{acc:.2f}**")

# -----------------------------
# Prediction Form
# -----------------------------
st.header("🔮 Predict New Data")

inputs = {}

for col in X.columns:
    # Use numeric input safely
    default_val = float(df[col].mean()) if col in numeric_cols else 0.0

    value = st.number_input(
        f"Enter {col}",
        value=default_val
    )
    inputs[col] = value

if st.button("Predict"):
    user_df = pd.DataFrame([inputs])
    user_scaled = scaler.transform(user_df)
    result = model.predict(user_scaled)
    st.success(f"Prediction: **{result[0]}**")
