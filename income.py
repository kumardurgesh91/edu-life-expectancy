import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re

# --- CONFIGURATION & LOADING ---
st.set_page_config(page_title="Income Predictor", page_icon="💡", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_assets():
    with open('gboost_income_model.pkl', 'rb') as f:
        return pickle.load(f)

# --- YOUR EXACT CLEANING FUNCTION ---
def clean_column_names_custom(df):
    df.columns = (df.columns
                    .str.strip()                
                    .str.lower()               
                    .str.replace(r'[^\w\s]', '', regex=True)
                    .str.replace(r'\s+', '_', regex=True)
                    .str.strip('_'))
    return df

assets = load_assets()
model = assets["model"]
scaler = assets["scaler"]
ohe = assets["encoder"]
le = assets["label_encoder"]
features_list = assets["features_list"]
cat_cols = assets["categorical_columns"]

# --- Sidebar instructions and educational notice ---
st.sidebar.header("How to use")
st.sidebar.write(
    "Provide demographic and work information, then press **Run Prediction**."
)
st.sidebar.markdown("---")
st.sidebar.header("About this app")
st.sidebar.info(
    "This is an educational demo app to show a toy income prediction model. "
    "Do not use predictions from this app for real-world decisions."
)

# Small styling touches
st.markdown(
    """
    <style>
    .stApp h1{font-size:30px}
    .stApp .big-card{background:#f8fafc;padding:18px;border-radius:8px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💰 Census Income Prediction")
st.info("Educational demo — predictions are illustrative only.")

# --- FORM INPUTS ---
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 17, 90, 40)
        workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
        edu_num = st.slider("Education Num", 1, 16, 15)
        marital = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    
    with col2:
        occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
        race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
        sex = st.radio("Sex", ['Male', 'Female'])
        hours = st.number_input("Hours per Week", 1, 99, 45)

    gain = st.number_input("Capital Gain", 0, 100000, 0)
    loss = st.number_input("Capital Loss", 0, 5000, 0)
    country = st.text_input("Native Country", "United-States")
    
    submit = st.form_submit_button("Run Prediction")

if submit:
    # 1. Create Raw DataFrame
    input_df = pd.DataFrame([{
        'age': age, 'workclass': workclass, 'educationnum': edu_num,
        'maritalstatus': marital, 'occupation': occupation, 'race': race,
        'sex': sex, 'capitalgain': gain, 'capitalloss': loss,
        'hoursperweek': hours, 'nativecountry': country
    }])

    # 2. Feature Engineering
    input_df['edu_age_interaction'] = input_df['educationnum'] * input_df['age']
    input_df['is_overtime'] = (input_df['hoursperweek'] > 40).astype(int)
    input_df['has_capital_stats'] = ((input_df['capitalgain'] > 0) | (input_df['capitalloss'] > 0)).astype(int)

    # 3. Country Grouping (Strip spaces because Census data has them)
    input_df['nativecountry'] = input_df['nativecountry'].str.strip()
    input_df['nativecountry'] = input_df['nativecountry'].apply(lambda x: 'United-States' if x == 'United-States' else 'Other')

    # 4. Encoding
    cat_features = ohe.transform(input_df[cat_cols])
    
    # Cleaning the OHE column names using your logic
    ohe_cols_raw = pd.DataFrame(columns=ohe.get_feature_names_out(cat_cols))
    ohe_cols_clean = clean_column_names_custom(ohe_cols_raw).columns
    
    cat_df = pd.DataFrame(cat_features, columns=ohe_cols_clean)

    # 5. Final Merge & Cleaning
    final_df = input_df.drop(cat_cols, axis=1).reset_index(drop=True)
    final_df = pd.concat([final_df, cat_df], axis=1)
    final_df = clean_column_names_custom(final_df)

    # 6. Alignment, Scaling, and Prediction
    try:
        final_df = final_df[features_list] # This reorders everything to match training
        final_scaled = scaler.transform(final_df)
        
        prediction = model.predict(final_scaled)[0]
        prob = model.predict_proba(final_scaled)[0][1]
        label = le.inverse_transform([prediction])[0]

        st.divider()
        if prediction == 1:
            st.success(f"### Predicted Income: {label}")
        else:
            st.warning(f"### Predicted Income: {label}")
        
        st.metric("Probability of >50K", f"{prob:.2%}")

    except KeyError as e:
        st.error(f"Feature Mismatch: {e}")
        st.write("Check if your column cleaning logic changed between training and deployment.")