import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# ==== Cấu hình giao diện ====
st.set_page_config(page_title="📊 Telco Churn Prediction", layout="centered")

st.title("📊 Telco Customer Churn Prediction & SHAP Explanation")
st.markdown("Nhập thông tin khách hàng để dự đoán khả năng **churn** và giải thích bằng SHAP.")

# ==== Load mô hình và preprocessor ====
pipeline, feature_names = joblib.load("churn_model.pkl")
preprocessor = pipeline.named_steps['processor']
model = pipeline.named_steps['classifier']
explainer = shap.Explainer(model)

# ==== Lưu lại lịch sử nếu có ====
if "churn_probs" not in st.session_state:
    st.session_state.churn_probs = []

# ==== Form người dùng ====
with st.form("form_input"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        seniorcitizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ['Yes', 'No'])
        dependents = st.selectbox("Dependents", ['Yes', 'No'])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phoneservice = st.selectbox("Phone Service", ['Yes', 'No'])
        multiplelines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])

    with col2:
        internetservice = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        onlinesecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        onlinebackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        deviceprotection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
        techsupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        streamingtv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
        streamingmovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
        contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        paperlessbilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
        paymentmethod = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        monthlycharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        totalcharges = st.number_input("Total Charges", min_value=0.0, value=2000.0)

    submitted = st.form_submit_button("🚀 Dự đoán")

# ==== Xử lý khi Submit ====
if submitted:
    input_df = pd.DataFrame([{
        "gender": gender,
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges
    }])

    # Dự đoán
    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]
    label = "❌ CHURN" if pred == 1 else "✅ Not Churn"
    color = "red" if pred == 1 else "green"

    st.subheader("🎯 Kết quả dự đoán")
    st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
    st.metric("Xác suất rời bỏ", f"{prob:.2%}")

    st.session_state.churn_probs.append(prob)

    # SHAP
    st.subheader("🧠 Giải thích bằng SHAP")
    st.markdown("Giá trị SHAP cho thấy những yếu tố nào ảnh hưởng đến kết quả dự đoán.")

    shap_input = preprocessor.transform(input_df)
    shap_values = explainer(shap_input)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

# ==== Lịch sử churn chart ====
if st.session_state.churn_probs:
    st.subheader("📈 Lịch sử xác suất churn")
    history_df = pd.DataFrame(st.session_state.churn_probs, columns=["Churn Probability"])
    st.line_chart(history_df)
