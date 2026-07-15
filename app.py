from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "Telco_Customer_Churn_Cleaned.csv"


# ---------------------------------------------------------
# Load model, scaler, and expected feature columns
# ---------------------------------------------------------
@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_feature_columns():
    dataset = pd.read_csv(DATA_PATH)

    dataset["Churn"] = dataset["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    dataset = dataset.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    encoded_dataset = pd.get_dummies(
        dataset,
        drop_first=True
    )

    return encoded_dataset.columns.tolist()


try:
    model, scaler = load_model_files()
    feature_columns = load_feature_columns()

except FileNotFoundError as error:
    st.error(f"Required project file was not found: {error}")
    st.stop()

except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter the customer's service, contract, and billing details "
    "to estimate their probability of leaving the telecom company."
)

st.divider()


# ---------------------------------------------------------
# Customer input form
# ---------------------------------------------------------
with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Has Partner",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Has Dependents",
            ["No", "Yes"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

        tech_support = st.selectbox(
            "Technical Support",
            ["No", "Yes", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=800.0,
            step=10.0
        )

    predict_button = st.form_submit_button(
        "Predict Churn Risk",
        use_container_width=True
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if predict_button:

    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Convert categorical inputs using the same method used in training.
    encoded_customer = pd.get_dummies(
        customer_data,
        drop_first=True
    )

    # Add any missing training columns and preserve their original order.
    encoded_customer = encoded_customer.reindex(
        columns=feature_columns,
        fill_value=0
    )

    model_name = type(model).__name__

    # Logistic Regression was trained using scaled values.
    if model_name == "LogisticRegression":
        prediction_input = scaler.transform(encoded_customer)
    else:
        prediction_input = encoded_customer

    churn_probability = model.predict_proba(
        prediction_input
    )[0][1]

    churn_prediction = int(
        churn_probability >= 0.50
    )

    if churn_probability >= 0.75:
        risk_level = "High Risk"
    elif churn_probability >= 0.50:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    st.divider()
    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    result_col1.metric(
        "Churn Probability",
        f"{churn_probability * 100:.1f}%"
    )

    result_col2.metric(
        "Risk Level",
        risk_level
    )

    result_col3.metric(
        "Model",
        model_name
    )

    st.progress(
        min(max(float(churn_probability), 0.0), 1.0)
    )

    if churn_prediction == 1:
        st.error(
            f"⚠️ This customer is likely to churn. "
            f"Estimated probability: {churn_probability * 100:.1f}%"
        )

        st.subheader("Recommended Retention Actions")

        recommendations = []

        if contract == "Month-to-month":
            recommendations.append(
                "Offer an incentive to move to a one-year or two-year contract."
            )

        if tenure < 12:
            recommendations.append(
                "Provide additional onboarding and early customer support."
            )

        if monthly_charges >= 70:
            recommendations.append(
                "Review the customer's package and consider a loyalty discount."
            )

        if payment_method == "Electronic check":
            recommendations.append(
                "Encourage the customer to use an automatic payment method."
            )

        if tech_support == "No":
            recommendations.append(
                "Offer an affordable technical-support package."
            )

        if not recommendations:
            recommendations.append(
                "Contact the customer and offer a personalized retention package."
            )

        for recommendation in recommendations:
            st.write(f"• {recommendation}")

    else:
        st.success(
            f"✅ This customer is unlikely to churn. "
            f"Estimated probability: {churn_probability * 100:.1f}%"
        )

        st.write(
            "Continue maintaining service quality and customer engagement."
        )
