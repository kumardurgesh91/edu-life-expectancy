import streamlit as st
import pandas as pd
import pickle
import os

# Set Page Config
st.set_page_config(
    page_title="Life Expectancy Predictor", 
    layout="wide",
    page_icon="🌍"
)

# Custom CSS for background and styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    .disclaimer {
        background: linear-gradient(45deg, #FF9FF3, #F368E0);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .header-container {
        text-align: center;
        padding: 30px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# IMPORTANT DISCLAIMER
st.markdown("""
<div class="disclaimer">
    ⚠️ IMPORTANT DISCLAIMER: This is a TEST/DEMO Project Only ⚠️<br>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = 'life_expectancy_full_pipeline.pkl'
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except AttributeError as e:
        st.error(f"Version Mismatch Error: The scikit-learn version on Streamlit doesn't match your local version. {e}")
        return None
    except Exception as e:
        st.error(f"Loading Error: {e}")
        return None

pipeline = load_model()

# 2. Country List
countries = [
    "Afghanistan", "Albania", "Algeria", "Angola", "Antigua and Barbuda", "Argentina", 
    "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", 
    "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia (Plurinational State of)", 
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei Darussalam", "Bulgaria", 
    "Burkina Faso", "Burundi", "Côte d'Ivoire", "Cabo Verde", "Cambodia", "Cameroon", 
    "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", 
    "Congo", "Cook Islands", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czechia", 
    "Democratic People's Republic of Korea", "Democratic Republic of the Congo", "Denmark", 
    "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", 
    "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Fiji", "Finland", "France", 
    "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", 
    "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", 
    "India", "Indonesia", "Iran (Islamic Republic of)", "Iraq", "Ireland", "Israel", 
    "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", 
    "Kyrgyzstan", "Lao People's Democratic Republic", "Latvia", "Lebanon", "Lesotho", 
    "Liberia", "Libya", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", 
    "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", 
    "Micronesia (Federated States of)", "Monaco", "Mongolia", "Montenegro", "Morocco", 
    "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", 
    "Nicaragua", "Niger", "Nigeria", "Niue", "Norway", "Oman", "Pakistan", "Palau", 
    "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", 
    "Qatar", "Republic of Korea", "Republic of Moldova", "Romania", "Russian Federation", 
    "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", 
    "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", 
    "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", 
    "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", 
    "Swaziland", "Sweden", "Switzerland", "Syrian Arab Republic", "Tajikistan", "Thailand", 
    "The former Yugoslav republic of Macedonia", "Timor-Leste", "Togo", "Tonga", 
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", 
    "Ukraine", "United Arab Emirates", "United Kingdom of Great Britain and Northern Ireland", 
    "United Republic of Tanzania", "United States of America", "Uruguay", "Uzbekistan", 
    "Vanuatu", "Venezuela (Bolivarian Republic of)", "Viet Nam", "Yemen", "Zambia", "Zimbabwe"
]

# 3. UI Header
st.markdown("""
<div class="header-container">
    <h1 style="color: #2C3E50; margin-bottom: 10px;">🌍 Global Life Expectancy Predictor 🏥</h1>
    <p style="color: #34495E; font-size: 18px; margin-bottom: 20px;">
        🔬 Advanced AI-powered health analytics for predicting life expectancy worldwide
    </p>
    <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px;">
        <div style="text-align: center;">
            <span style="font-size: 40px;">📊</span><br>
            <small style="color: #7F8C8D;">Data-Driven</small>
        </div>
        <div style="text-align: center;">
            <span style="font-size: 40px;">🤖</span><br>
            <small style="color: #7F8C8D;">AI Powered</small>
        </div>
        <div style="text-align: center;">
            <span style="font-size: 40px;">🌐</span><br>
            <small style="color: #7F8C8D;">Global Insights</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📝 Enter country data below to see the predicted average life expectancy")

# 4. Input Layout
st.markdown("## 🎯 Prediction Parameters")


# Input fields in 4 columns to reduce scrolling
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 👥 Demographics")
    country_selected = st.selectbox("🌍 Country", countries, index=countries.index("United States of America"))
    status = st.radio("🏗️ Status", ["Developing", "Developed"], index=1)
    year = st.slider("📅 Year", 2000, 2026, 2015)
    schooling = st.number_input("🎓 Schooling", 0.0, 20.0, 13.0)

with col2:
    st.markdown("### 🏥 Health")
    adult_mortality = st.number_input("💀 Adult Mort.", 0, 300, 13)
    underfive_deaths = st.number_input("👶 Under-5 Deaths", 0, 200, 0)
    hiv_aids = st.number_input("🦠 HIV/AIDS", 0.0, 10.0, 0.1)
    hepatitis_b = st.slider("💉 Hep B %", 0, 100, 80)

with col3:
    st.markdown("### ⚖️ Health Cont.")
    thinness_119_years = st.number_input("⚖️ Thinness 1-19%", 0.0, 50.0, 0.8)
    bmi = st.slider("⚖️ BMI", 10.0, 100.0, 69.0)
    alcohol = st.number_input("🍺 Alcohol", 0.0, 25.0, 4.0)
    income_comp = st.number_input("💰 Income Comp.", 0.0, 1.0, 0.8)

with col4:
    st.markdown("### 💼 Economics")
    gdp = st.number_input("💵 GDP", 0, 150000, 60000)
    total_expenditure = st.number_input("💸 Total Exp %", 0.0, 100.0, 3.0)
    measles = st.number_input("🦠 Measles", 0, 1000, 188)
    polio = st.slider("💉 Polio %", 0, 100, 93)
    diphtheria = st.slider("💉 Diphtheria %", 0, 100, 93)

st.markdown("</div>", unsafe_allow_html=True)

# 5. Prediction
st.divider()
if st.button("Calculate Life Expectancy", type="primary"):
    # Map inputs to model structure
    input_data = {
        'country': country_selected,
        'year': year,
        'adult_mortality': adult_mortality,
        'alcohol': alcohol,
        'hepatitis_b': hepatitis_b,
        'measles': measles,
        'bmi': bmi,
        'underfive_deaths': underfive_deaths,
        'polio': polio,
        'total_expenditure': total_expenditure,
        'diphtheria': diphtheria,
        'hivaids': hiv_aids,
        'gdp': gdp,
        'thinness_119_years': thinness_119_years,
        'income_composition_of_resources': income_comp,
        'schooling': schooling,
        'status_Developing': 1 if status == "Developing" else 0
    }
    
    df_input = pd.DataFrame([input_data])
    prediction = pipeline.predict(df_input)[0]
    
    st.metric(label="Estimated Life Expectancy", value=f"{prediction:.1f} Years")
    
    if prediction > 75:
        st.success("This region has a high life expectancy!")
    elif prediction < 55:
        st.error("This region has a low life expectancy.")