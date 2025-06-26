import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load pipeline & features
pipeline, feature_names = joblib.load("churn_model.pkl")
preprocessor = pipeline.named_steps['processor']
model = pipeline.named_steps['classifier']

# Tải SHAP explainer sẵn
explainer = shap.Explainer(model)

# ====== Streamlit UI ======
st.title("🔍 Telco Churn Prediction & Explainability (SHAP)")
st.markdown("Nhập thông tin khách hàng và xem khả năng churn cùng với giải thích bằng SHAP.")

# ==== Form người dùng nhập ====
with st.form("customer_form"):
    gender = st.selectbox("Gender", ['Male', 'Female'])
    seniorcitizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phoneservice = st.selectbox("Phone Service", ['Yes', 'No'])
    multiplelines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
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
    submit = st.form_submit_button("Dự đoán")

# ==== Dự đoán & SHAP Explain ====
if submit:
    input_data = pd.DataFrame([{
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
    prediction = pipeline.predict(input_data)[0]
    proba = pipeline.predict_proba(input_data)[0][1]
    label = "CHURN ❌" if prediction == 1 else "Not Churn ✅"

    st.subheader("🔮 Dự đoán:")
    st.markdown(f"**Kết quả:** {label}")
    st.markdown(f"**Xác suất churn:** `{proba:.2%}`")

    # SHAP explain
    input_transformed = preprocessor.transform(input_data)
    shap_value = explainer(input_transformed)

    st.subheader("🧠 Giải thích bằng SHAP")
    st.markdown("Biểu đồ dưới đây cho thấy các yếu tố nào đẩy quyết định về churn cao/lên hoặc thấp/xuống.")

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_value[0], max_display=10, show=False)
    st.pyplot(fig)
