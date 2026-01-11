import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(page_title="Life Expectancy Predictor", layout="wide")

# 1. Load the Model Pipeline
@st.cache_resource
def load_model():
    return joblib.load('life_expectancy_full_pipeline.pkl')

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
st.title("🌍 Global Life Expectancy Predictor")
st.markdown("Enter country data below to see the predicted average life expectancy.")

# 4. Input Layout
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Demographics")
        country_selected = st.selectbox("Select Country", countries, index=countries.index("United States of America"))
        status = st.radio("Economic Status", ["Developing", "Developed"], index=1)  # Developed
        year = st.slider("Year", 2000, 2026, 2015)
        schooling = st.number_input("Years of Schooling", 0.0, 20.0, 13.0)
        income_comp = st.number_input("Income Composition", 0.0, 1.0, 0.8)

    with col2:
        st.header("Health Factors")
        adult_mortality = st.number_input("Adult Mortality (per 1000)", 0, 300, 13)  # Low for developed country
        underfive_deaths = st.number_input("Under-five Deaths (per 1000)", 0, 200, 0)
        hiv_aids = st.number_input("HIV/AIDS Rate", 0.0, 10.0, 0.1)
        hepatitis_b = st.slider("Hepatitis B Immunization %", 0, 100, 80)
        thinness_119_years = st.number_input("Thinness 1-19 years (%)", 0.0, 50.0, 0.8)
        bmi = st.slider("Average BMI", 10.0, 100.0, 69.0)
        alcohol = st.number_input("Alcohol Consumption", 0.0, 25.0, 4.0)

    with col3:
        st.header("Economics & Immunization")
        gdp = st.number_input("GDP (USD)", 0, 150000, 60000)  # Approximate US GDP per capita
        total_expenditure = st.number_input("Total Expenditure %", 0.0, 100.0, 3.0)
        measles = st.number_input("Measles cases", 0, 1000, 188)
        polio = st.slider("Polio Immunization %", 0, 100, 93)
        diphtheria = st.slider("Diphtheria Immunization %", 0, 100, 93)

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