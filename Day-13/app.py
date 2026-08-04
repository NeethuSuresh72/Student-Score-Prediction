import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Student Score Prediction",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    .title-container {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .title-container h1 {
        color: #1f2937;
        font-size: 2.4rem;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(79, 70, 229, 0.35);
    }
    .result-card {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-top: 1.5rem;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.25);
    }
    .result-card h2 {
        margin: 0;
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.9;
    }
    .result-card h1 {
        margin: 0.3rem 0 0 0;
        font-size: 2.8rem;
    }
    div[data-testid="stNumberInput"] label {
        font-weight: 600;
        color: #374151;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("student_score_model.pkl")

model = load_model()

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This tool predicts a student's final score based on "
        "study hours, attendance, and previous performance."
    )
    st.markdown("---")
    st.caption("Model: Regression-based predictor")
    st.caption("Built with Streamlit")

# Title
st.markdown("""
    <div class="title-container">
        <h1>🎓 Student Score Prediction</h1>
    </div>
    <div class="subtitle">Enter student details below to predict the final score</div>
""", unsafe_allow_html=True)

# Input card
with st.container(border=True):
    st.subheader("📋 Student Details")

    col1, col2 = st.columns(2)

    with col1:
        hours = st.number_input(
            "📖 Hours Studied",
            min_value=0,
            max_value=24,
            value=5,
            help="Average hours studied per day"
        )
        previous_score = st.number_input(
            "📊 Previous Score",
            min_value=0,
            max_value=100,
            value=75
        )

    with col2:
        attendance = st.number_input(
            "🗓️ Attendance (%)",
            min_value=0,
            max_value=100,
            value=85
        )

    st.write("")
    predict_clicked = st.button("🔮 Predict Score")

# Prediction result
if predict_clicked:
    student_data = pd.DataFrame({
        "Hours_Studied": [hours],
        "Attendance": [attendance],
        "Previous_Score": [previous_score]
    })

    with st.spinner("Calculating prediction..."):
        prediction = model.predict(student_data)

    st.markdown(f"""
        <div class="result-card">
            <h2>PREDICTED FINAL SCORE</h2>
            <h1>{prediction[0]:.2f}</h1>
        </div>
    """, unsafe_allow_html=True)

    st.progress(min(int(prediction[0]), 100) / 100)