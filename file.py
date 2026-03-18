import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier

st.set_page_config(page_title="Credit Risk AI", layout="centered")

st.title("Credit Risk Prediction (TabNet)")

# -----------------------------
# UPLOAD DATA
# -----------------------------
file = st.file_uploader("Upload Credit Dataset (CSV)", type=["csv"])

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
        X = scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -----------------------------
        # TRAIN MODEL
        # -----------------------------
        if st.button("Train Model"):

            with st.spinner("Training..."):

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

            # Save
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.n_features = X.shape[1]

# -----------------------------
# PREDICT SECTION
# -----------------------------
if "model" in st.session_state:

    st.subheader("Test Custom Input")

    inputs = []

    for i in range(st.session_state.n_features):
        val = st.number_input(f"Feature {i+1}", value=0.0)
        inputs.append(val)

    if st.button("Predict"):

        data = np.array([inputs])
        data = st.session_state.scaler.transform(data)

        pred = st.session_state.model.predict(data)[0]
        prob = st.session_state.model.predict_proba(data)[0].max()

        label = "Safe" if pred == 0 else "Risky"

        st.subheader("Result")
        st.write("Prediction:", label)
        st.progress(float(prob))
        st.write("Confidence:", round(prob,2))