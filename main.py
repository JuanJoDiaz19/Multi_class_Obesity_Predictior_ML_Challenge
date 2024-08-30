import streamlit as st 
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

loaded_model=joblib.load('XGB.joblib')

st.title("👨🏻‍⚕️ Multi-Class Prediction of Obesity")

# Test case

# {
#     "Age": 26.899886,
#     "Height": 1.848294,
#     "Weight": 120.644178,
#     "FCVC": 2.938616,
#     "NCP": 3.0,
#     "CAEC": 0,
#     "CH2O": 2.825629,
#     "FAF": 0.8554,
#     "TUE": 0.0,
#     "CALC": 0,
#     "MTRANS": 4,
#     "Gender_Female": false,
#     "family_history_with_overweight_yes": true,
#     "FAVC_yes": true,
#     "SMOKE_yes": false,
#     "SCC_yes": false
# }


# Making the mapping from the model answer to a normal answer

NObeyesdad_inverse_mapping = {
    '0': 'Insufficient_Weight',
    '1': 'Normal_Weight',
    '2': 'Overweight_Level_I',
    '3': 'Overweight_Level_II',
    '4': 'Obesity_Type_I',
    '5': 'Obesity_Type_II',
    '6': 'Obesity_Type_III'
}


transportation_mapping = {
    'Bike': 0,
    'Walking': 1,
    'Motorbike': 2,
    'Automobile': 3,
    'Public_Transportation': 4,
}

frequency_mapping = {
    'No': 1,
    'Sometimes': 0,
    'Frequently': 2,
    'Always': 3
}

boolean_mapping = {
    'Yes': 1,
    'No': 0
}

gender_mapping = {
    'Female': 0,
    'Male': 1,
}

st.write("The project is an Obesity Risk Prediction System that uses a neural network to assess an individual's likelihood of being obese based on various personal, lifestyle, and dietary factors. By inputting data such as age, height, weight, eating habits, physical activity, and other relevant characteristics, the model generates a prediction that helps identify the risk of obesity. This tool can be used for early detection and prevention, enabling users to make informed decisions about their health and lifestyle.")

st.write("""### **Inputs:**
- **Age:** Numerical value representing the person's age.
- **Height and Weight:** Numerical values used to calculate the Body Mass Index (BMI), which is a key indicator in obesity prediction.
- **FCVC (Frequency of Consumption of Vegetables):** Average frequency of vegetable intake.
- **NCP (Number of Main Meals per Day):** Indicates how often the person eats full meals each day.
- **CAEC (Consumption of Food Between Meals):** Indicates whether the person frequently eats between main meals.
- **CH2O (Daily Water Consumption):** Amount of water consumed daily.
- **FAF (Physical Activity Frequency per Week):** How often the person engages in physical activity each week.
- **TUE (Time Using Technology Devices per Day):** Average daily time spent using electronic devices, which might indicate sedentary behavior.
- **CALC (Alcohol Consumption Frequency):** Frequency of alcohol consumption.
- **MTRANS (Primary Mode of Transportation):** Indicates the primary mode of transportation, which could affect physical activity levels.
- **Gender_Female:** Boolean value indicating the gender of the individual.
- **Family History of Overweight:** Boolean value indicating whether there is a family history of overweight/obesity.
- **FAVC (Frequent Consumption of High-Calorie Food):** Boolean value indicating if the person frequently consumes high-calorie foods.
- **SMOKE:** Boolean value indicating if the person smokes.
- **SCC (Monitoring of Caloric Intake):** Boolean value indicating if the person monitors their caloric intake.
""")

st.write("In order to make a prediction enter the following fields into the system:")

age = st.number_input("Age")
height = st.number_input("Height (mt)")
weight = st.number_input("Weight (kg)")
fcvc = st.number_input("Frequency of Consumption of Vegetables (FCVC)")
ncp = st.number_input("Number of Main Meals per Day (NCP)")
caec = st.selectbox("Consumption of Food Between Meals (CAEC)", ["No", "Sometimes", "Frequently", "Always"])
ch20 = st.number_input("Daily Water Consumption (CH2O)")
faf = st.number_input("Physical Activity Frequency per Week (FAF)")
tue = st.number_input("Time Using Technology Devices per Day (TUE)")
calc = st.selectbox("Alcohol Consumption Frequency (CALC)", ["No", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Primary Mode of Transportation (MTRANS)", ["Bike", "Walking", "Motorbike", "Automobile", "Public Transportation"])

gender = st.selectbox("Gender", ["Male", "Female"])
family_overweight = st.selectbox("Family history with overweight", ["Yes", "No"])
favc = st.selectbox("Frequent consumption of high-caloric food (FAVC)", ["Yes", "No"])
smoke = st.selectbox("Smoke", ["Yes", "No"])
scc = st.selectbox("SCC", ["Yes", "No"])



print(frequency_mapping[caec])

query_map = {
    'Age': age,
    'Height': height, 
    'Weight':weight, 
    'FCVC':fcvc, 
    'NCP':ncp, 
    'CAEC':frequency_mapping[caec], 
    'CH20':ch20,
    'FAF':faf,
    'TUE':tue,
    'CALC':frequency_mapping[calc],
    'MTRANS':transportation_mapping[mtrans],
    'Gender_Female':gender_mapping[gender],
    'family_history_with_overweight_yes':boolean_mapping[family_overweight],
    'FAVC_yes':boolean_mapping[favc],
    'SMOKE_yes':boolean_mapping[smoke],
    'SCC_yes':boolean_mapping[scc]
}


if st.button("Make the prediction"):
    
    one_row_test = pd.Series(query_map)

    # Colocar la probabilidad con lo que puede pasar, dentro del mismo metodo esta
    result = loaded_model.predict(one_row_test.values.reshape(1, -1))
    st.header("Test results")
    st.write("According to our model it it likely that the patient has:", )
    st.write("####", NObeyesdad_inverse_mapping[str((result[0]))])
    probabilites = loaded_model.predict_proba(one_row_test.values.reshape(1, -1)).tolist()[0]
    mapped_probabilities = {}

    # Assuming probabilites is a list of probabilities
    probabilites = loaded_model.predict_proba(one_row_test.values.reshape(1, -1)).tolist()[0]

    # Iterate over the list with index
    for i, probability in enumerate(probabilites):
        mapped_probabilities[NObeyesdad_inverse_mapping[str(i)]] =  f"{round(probability*100, 3)} %"
    
    data = pd.Series(mapped_probabilities)
    data.name = "Probability"

    st.write("Here's is also a chart with the odds of having the other types of obesity risk:")
    st.write(data)