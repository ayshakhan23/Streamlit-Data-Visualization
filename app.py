import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

st.set_page_config(page_title="Advanced Income Prediction App", layout="wide")

# ------------------------------------------
# LOAD & CLEAN DATASET
# ------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")

    # Clean typographical issues
    df = df.applymap(lambda x: str(x).replace("’","'").strip() if isinstance(x,str) else x)

    return df

df = load_data()

st.title("💼 Income Prediction · Streamlit App (Advanced)")
st.write("All charts now include Y-axis labels · Dataset auto-cleaned · Real category dropdowns")

# --------------------------------------------------------------------------------
# DETECT NUMERIC & CATEGORICAL COLUMNS
# --------------------------------------------------------------------------------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# ------------------------------------------
# ENCODE DATAFRAME FOR MODELING
# ------------------------------------------
df_encoded = df.copy()
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ------------------------------------------
# PREVIEW
# ------------------------------------------
st.header("📌 Dataset Preview")
st.dataframe(df.head())

# ------------------------------------------
# SUMMARY
# ------------------------------------------
st.subheader("Summary Statistics")
st.write(df.describe(include="all"))

# --------------------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------------------
st.header("📈 Data Visualization")

# ---------------- HISTOGRAM ---------------------
st.subheader("Histogram")
if len(num_cols) > 0:
    hist_col = st.selectbox("Select numerical column", num_cols)
    fig, ax = plt.subplots()
    ax.hist(df[hist_col], bins=20)
    ax.set_xlabel(hist_col)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {hist_col}")
    st.pyplot(fig)

# ---------------- BAR PLOT (categorical) ---------------------
st.subheader("Bar Chart")
if len(cat_cols) > 0:
    bar_col = st.selectbox("Select categorical column", cat_cols)
    fig, ax = plt.subplots()
    df[bar_col].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel(bar_col)
    ax.set_ylabel("Count")
    ax.set_title(f"Bar Chart of {bar_col}")
    st.pyplot(fig)

# ---------------- CORRELATION ---------------------
st.subheader("Correlation Heatmap (numerical only)")
num_df = df_encoded[num_cols]

if num_df.shape[1] > 1:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
else:
    st.warning("Not enough numeric columns.")

# --------------------------------------------------------------------------------
# MODEL TRAINING (Predict Income)
# --------------------------------------------------------------------------------
st.header("🤖 Predict Income")

target = "Income"  # You requested fixed target
if target not in df.columns:
    st.error("Dataset must contain an 'Income' column.")
else:
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_choice = st.selectbox(
        "Choose Model", 
        ["Linear Regression", "Polynomial Regression (Degree 2)", "Logistic Regression"]
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # ---------------- MODEL LOGIC ----------------------
    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(x_train_scaled, y_train)
        pred = model.predict(x_test_scaled)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        st.success(f"✔ MSE: {mse:.2f}")
        st.success(f"✔ R² Score: {r2:.2f}")

    elif model_choice == "Polynomial Regression (Degree 2)":
        poly = PolynomialFeatures(degree=2)
        x_poly_train = poly.fit_transform(x_train_scaled)
        x_poly_test = poly.transform(x_test_scaled)

        model = LinearRegression()
        model.fit(x_poly_train, y_train)
        pred = model.predict(x_poly_test)

        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        st.success(f"✔ MSE: {mse:.2f}")
        st.success(f"✔ R² Score: {r2:.2f}")

    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=200)
        model.fit(x_train_scaled, y_train)
        pred = model.predict(x_test_scaled)

        acc = accuracy_score(y_test, pred)
        st.success(f"✔ Accuracy: {acc:.2f}")

    # --------------------------------------------------------------------------------
    # NEW PREDICTION FORM (shows REAL categories)
    # --------------------------------------------------------------------------------
    st.header("🔮 Predict Income for a New Person")

    user_input = {}

    for col in X.columns:
        if col in cat_cols:
            # Show real categories
            options = df[col].unique().tolist()
            selected = st.selectbox(f"{col}", options)
            # Encode to number
            user_input[col] = encoders[col].transform([selected])[0]
        else:
            default = float(df[col].mean())
            user_input[col] = st.number_input(f"{col}", value=default)

    if st.button("Predict Income"):
        user_df = pd.DataFrame([user_input])
        user_scaled = scaler.transform(user_df)

        if model_choice == "Polynomial Regression (Degree 2)":
            user_scaled = poly.transform(user_scaled)

        result = model.predict(user_scaled)
        st.success(f"Predicted Income Value: **{result[0]}**")
