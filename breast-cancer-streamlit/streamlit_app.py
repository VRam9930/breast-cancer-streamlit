
# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
feature_names = data.feature_names

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model training
model = LogisticRegression(max_iter=10000)
model.fit(X_scaled, y)

# Streamlit UI
st.title("Breast Cancer Prediction App")
st.write("Provide the required features below to predict if the tumor is benign or malignant.")

user_input = [st.number_input(f"{feature}", value=0.0) for feature in feature_names]

if st.button("Predict"):
    input_np = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_np)
    prediction = model.predict(input_scaled)
    result = "Malignant" if prediction[0] == 0 else "Benign"
    st.success(f"The prediction is: {result}")
