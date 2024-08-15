import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('framingham.csv')
    return data

data = load_data()

# Preprocess the data
def preprocess_data(data):
    # Handle missing values
    data = data.fillna(data.mean())
    
    # Encode categorical variables
    data = pd.get_dummies(data, columns=['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes'])
    
    # Normalize continuous variables
    scaler = StandardScaler()
    continuous_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    data[continuous_cols] = scaler.fit_transform(data[continuous_cols])
    
    return data, scaler

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Preprocess the data and train the model
preprocessed_data, scaler = preprocess_data(data)
X = preprocessed_data.drop('TenYearCHD', axis=1)
y = preprocessed_data['TenYearCHD']
model = train_model(X, y)

# Save the model and scaler
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Streamlit app
st.title('Cardiovascular Risk Prediction')

st.write("""
This app predicts the 10-year risk of coronary heart disease (CHD) based on patient data.
Please enter the patient's information below:
""")

# Input fields
age = st.number_input('Age', min_value=20, max_value=100)
male = st.selectbox('Sex', ['Male', 'Female']) == 'Male'
current_smoker = st.checkbox('Current Smoker')
cigs_per_day = st.number_input('Cigarettes per Day', min_value=0, max_value=100)
bp_meds = st.checkbox('On Blood Pressure Medication')
prevalent_stroke = st.checkbox('Prevalent Stroke')
prevalent_hyp = st.checkbox('Prevalent Hypertension')
diabetes = st.checkbox('Diabetes')
tot_chol = st.number_input('Total Cholesterol', min_value=100, max_value=600)
sys_bp = st.number_input('Systolic Blood Pressure', min_value=80, max_value=300)
dia_bp = st.number_input('Diastolic Blood Pressure', min_value=40, max_value=200)
bmi = st.number_input('BMI', min_value=15, max_value=50)
heart_rate = st.number_input('Heart Rate', min_value=40, max_value=200)
glucose = st.number_input('Glucose', min_value=40, max_value=400)

# Prediction button
if st.button('Predict CHD Risk'):
    # Prepare input data
    input_data = pd.DataFrame({
        'age': [age],
        'male': [male],
        'currentSmoker': [current_smoker],
        'cigsPerDay': [cigs_per_day],
        'BPMeds': [bp_meds],
        'prevalentStroke': [prevalent_stroke],
        'prevalentHyp': [prevalent_hyp],
        'diabetes': [diabetes],
        'totChol': [tot_chol],
        'sysBP': [sys_bp],
        'diaBP': [dia_bp],
        'BMI': [bmi],
        'heartRate': [heart_rate],
        'glucose': [glucose]
    })
    
    # Perform one-hot encoding
    input_data = pd.get_dummies(input_data, columns=['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes'])
    
    # Ensure all columns from training are present
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match training data
    input_data = input_data[X.columns]
    
    # Scale the input data
    continuous_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    input_data[continuous_cols] = scaler.transform(input_data[continuous_cols])
    
    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    # Display result
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.write('High risk of developing CHD in the next 10 years.')
    else:
        st.write('Low risk of developing CHD in the next 10 years.')
    
    st.write(f'Probability of developing CHD: {probability:.2%}')

st.write("""
Note: This app is for educational purposes only. Always consult with a healthcare professional for medical advice.
""")

#Footer: 

# Custom CSS for footer
footer_css = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    padding: 10px 0;
    font-size: 14px;
    height : 10vh
}
h5{
position : relative;
top : 35px;
}

.social-icons {
    display: flex;
    justify-content: center;
    gap: 20px;
}
.social-icons a {
    color: black;
    text-decoration: none;
}
.social-icons img {
    width: 30px;
    height: 30px;
}
</style>
"""

# Inject custom CSS
st.markdown(footer_css, unsafe_allow_html=True)

# Footer content for my social profiles and about us.
footer_content = """
<div class="footer">
    <div class="social-icons">
        <a href="https://in.linkedin.com/in/hrithik-kumar-singh-0a4127301" target="_blank"><img src="https://img.icons8.com/color/48/000000/linkedin.png"/></a>
        <a href="https://github.com/hrithikksingh3" target="_blank"><img src="https://img.icons8.com/color/48/000000/github--v1.png"/></a>
        <a href="https://www.instagram.com/codersvoice/" target="_blank"><img src="https://img.icons8.com/color/48/000000/instagram-new.png"/></a>
        <a href="https://medium.com/@hrithikkumarsingh" target="_blank"><img src="https://img.icons8.com/color/48/000000/medium-monogram.png"/></a>
        <a href="https://www.youtube.com/@codersvoicehrithik" target="_blank"><img src="https://img.icons8.com/color/48/000000/youtube-play.png"/></a>
        <a href="https://x.com/Codersvoice_" target="_blank"><img src="https://img.icons8.com/color/48/000000/twitter.png"/></a>
    </div>
    <p>Copyright ©2024 Hrithik Kumar Singh All rights reserved.</p>
</div>
"""

# Create a container for the footer
footer_container = st.container()

# Push the footer to the bottom
st.markdown('<div style="margin-bottom:100px;"></div>', unsafe_allow_html=True)

# Render the footer in the container
with footer_container:
    st.markdown(footer_content, unsafe_allow_html=True)
