import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier

st.set_page_config(page_title="Tabular AI System", layout="centered")

st.title("Tabular AI (TabNet)")

# -----------------------------
# UPLOAD DATA
# -----------------------------
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # SELECT TARGET
    # -----------------------------
    target_col = st.selectbox("Select Target Column", df.columns)

    if target_col:

        # -----------------------------
        # CLEAN DATA
        # -----------------------------
        df = df.dropna(subset=[target_col])

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Fill missing values
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = X[col].fillna("Unknown")
            else:
                X[col] = X[col].fillna(X[col].mean())

        # Encode categorical
        encoders = {}
        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders[col] = le

        # Encode target if needed
        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # -----------------------------
        # TRAIN MODEL
        # -----------------------------
        if st.button("Train Model"):

            with st.spinner("Training model..."):

                model = TabNetClassifier(verbose=0)
                model.fit(
                    X_train, y_train,
                    max_epochs=50,
                    patience=10,
                    batch_size=256
                )

                preds = model.predict(X_test)
                acc = (preds == y_test).mean()

            st.success("Model Ready")
            st.metric("Accuracy", f"{acc*100:.2f}%")

            # Save everything
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.columns = X.columns.tolist()
            st.session_state.raw_df = df
            st.session_state.encoders = encoders

# -----------------------------
# SMART PREDICTION UI
# -----------------------------
if "model" in st.session_state:

    st.subheader("Test Custom Input")

    inputs = []
    df_original = st.session_state.raw_df
    columns = st.session_state.columns
    encoders = st.session_state.encoders

    for col in columns:

        if df_original[col].dtype == "object":
            options = df_original[col].dropna().unique().tolist()
            val = st.selectbox(col, options)

        else:
            min_val = float(df_original[col].min())
            max_val = float(df_original[col].max())
            mean_val = float(df_original[col].mean())

            val = st.slider(col, min_val, max_val, mean_val)

        inputs.append(val)

    # -----------------------------
    # PREDICT
    # -----------------------------
    if st.button("Predict"):

        input_df = pd.DataFrame([inputs], columns=columns)

        # Encode categorical
        for col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        # Scale
        data = st.session_state.scaler.transform(input_df)

        pred = st.session_state.model.predict(data)[0]
        prob = st.session_state.model.predict_proba(data)[0].max()

        label = "Safe" if pred == 0 else "Risky"

        st.subheader("Result")
        st.write("Prediction:", label)
        st.progress(float(prob))
        st.write("Confidence:", round(prob, 2))