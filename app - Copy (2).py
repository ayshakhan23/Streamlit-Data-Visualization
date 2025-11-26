import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

st.title("📊 Advanced Data Visualization & ML Prediction App")
st.write("Upload your dataset and perform visualizations + predict **Income** using ML models.")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("File uploaded successfully!")

    st.subheader("🔍 Preview Data")
    st.write(df.head())

    # -----------------------------
    # Identify numeric + categorical
    # -----------------------------
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    st.info(f"Detected Numeric Columns: {num_cols}")
    st.info(f"Detected Categorical Columns: {cat_cols}")

    # ----------------------------------------------------
    # Encode while keeping original categories for UI
    # ----------------------------------------------------
    encoders = {}
    df_encoded = df.copy()

    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le

    # ------------------------------------------
    # 🔥 Data Visualization Section
    # ------------------------------------------
    st.header("📈 Data Visualization")

    # ==== Histogram ====
    st.subheader("Histogram")
    hist_col = st.selectbox("Select column for histogram", num_cols)
    if hist_col:
        fig, ax = plt.subplots()
        sns.histplot(df[hist_col], bins=30, ax=ax)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(hist_col)
        st.pyplot(fig)

    # ==== Boxplot ====
    st.subheader("Boxplot")
    box_col = st.selectbox("Select column for boxplot", num_cols, key="box")
    if box_col:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[box_col], ax=ax)
        ax.set_ylabel(box_col)
        st.pyplot(fig)

    # ==== Scatter Plot ====
    st.subheader("Scatter Plot")
    x_scatter = st.selectbox("Select X-axis (numeric)", num_cols, key="scatter_x")
    y_scatter = st.selectbox("Select Y-axis (numeric)", num_cols, key="scatter_y")
    if x_scatter and y_scatter:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_scatter], y=df[y_scatter], ax=ax)
        ax.set_xlabel(x_scatter)
        ax.set_ylabel(y_scatter)
        st.pyplot(fig)

    # ==== Heatmap ====
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_ylabel("Columns")
    st.pyplot(fig)

    # ----------------------------------------------------
    # PREDICTION SECTION
    # ----------------------------------------------------
    st.header("🤖 ML Model: Predict Income")

    if "Income" not in df.columns:
        st.error("Your dataset must contain an 'Income' column.")
    else:
        target = "Income"

        # Features selection
        st.subheader("Select Features")
        features = st.multiselect("Choose input features", df_encoded.columns.tolist(), default=[c for c in df_encoded.columns if c != target])

        if len(features) > 0:
            X = df_encoded[features]
            y = df[target]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # -------------------------
            # Select ML Model
            # -------------------------
            model_choice = st.radio("Select ML Model", 
                                    ["Linear Regression", "Polynomial Regression", "Logistic Regression"])

            # -------------------------
            # Linear Regression
            # -------------------------
            if model_choice == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                st.write("### R² Score:", r2_score(y_test, preds))

            # -------------------------
            # Polynomial Regression
            # -------------------------
            elif model_choice == "Polynomial Regression":
                degree = st.slider("Select Degree", 2, 5)

                poly = PolynomialFeatures(degree=degree)
                X_poly_train = poly.fit_transform(X_train)
                X_poly_test = poly.transform(X_test)

                model = LinearRegression()
                model.fit(X_poly_train, y_train)
                preds = model.predict(X_poly_test)

                st.write("### R² Score:", r2_score(y_test, preds))

            # -------------------------
            # Logistic Regression
            # -------------------------
            elif model_choice == "Logistic Regression":
                try:
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    st.write("### Accuracy:", accuracy_score(y_test, preds))
                except:
                    st.error("❌ Logistic Regression only works if 'Income' is categorical (e.g., low/medium/high).")

            # -------------------------
            # User Prediction
            # -------------------------
            st.subheader("🔮 Predict Income for Custom Input")

            user_input = {}
            for col in features:
                if col in cat_cols:  # dropdown should show categories
                    options = encoders[col].classes_
                    val = st.selectbox(f"{col}", options)
                    encoded_val = encoders[col].transform([val])[0]
                    user_input[col] = encoded_val
                else:
                    user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()))

            # Convert to DataFrame
            input_df = pd.DataFrame([user_input])

            # Polynomial transform if required
            if model_choice == "Polynomial Regression":
                input_df = poly.transform(input_df)

            # Make prediction
            if st.button("Predict Income"):
                result = model.predict(input_df)
                st.success(f"Predicted Income: **{result[0]:,.2f}**")
