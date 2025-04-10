import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE,SMOTENC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



#Load data from csv file
def load_telecom():
    telecom = pd.read_csv('tele-churn.csv')
    st.dataframe(telecom)
    return telecom
 
#Preprocess data
def preprocess_data(telecom):
   # if 'churn' in telecom.columns:
        telecom = telecom.drop(['phone number', 'state', 'area code'], axis=1) 
        y = telecom['churn']
        X = telecom.drop(['churn'], axis=1)
        #Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        #Apply smote to the training data
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test


st.title("Predicting Mobile Churn Analysis and Mitigation Strategies")
#Train model
def train_model(X_train, y_train):
    gbc_model = GradientBoostingClassifier()
    gbc_model.fit(X_train_resampled, y_train_resampled)
    return gbc_model

#Evaluate model
def evaluate_model(gbc_model, X_test, y_test):
    predictions = gbc_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    return accuracy, report, matrix

#Create streamlit app
def main():
    st.write("This app predicts customer churn using the best-performing model.")
    st.title("PREDICTION OF MOBILE CHURN IN TELECOMMUNICATION INDUSTRY")

#Load and preprocess data
telecom= load_telecom()
#telecom = telecom.drop(['phone number', 'state', 'area code'], axis=1)
telecom['international plan'] = telecom['international plan'].map({'no': 0, 'yes': 1})
telecom['voice mail plan'] = telecom['voice mail plan'].map({'no': 0, 'yes': 1}) 
X_train_resampled, X_test, y_train_resampled, y_test = preprocess_data(telecom)

#Train and Evaluate model
gbc_model = train_model(X_train_resampled, y_train_resampled)
accuracy, report, matrix = evaluate_model(gbc_model, X_test, y_test)


#Display results
#st.write("Model Evaluation:")
st.write("Accuracy:", accuracy)
#st.write("Classification Report:")
#st.write(report)
#st.write("Confusion Matrix:")
#st.write(matrix)


#Create a form for users to input their data
st.write("Make a prediction:")
account_length = st.number_input("Account Length", min_value=0.0, max_value=500.0)
international_plan = st.selectbox("International Plan", [ 'Yes', 'No'])
voice_mail_plan = st.selectbox("Voice Mail Plan", [ 'Yes', 'No'])
number_vmail_messages = st.number_input("Number Voicemail Messages", min_value=0.0, max_value=500.0)
total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, max_value=500.0)
total_day_calls = st.number_input("Total Day Calls", min_value=0.0, max_value=500.0)
total_day_charge = st.number_input("Total Day Charge", min_value=0.0, max_value=500.0)
total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, max_value=500.0)
total_eve_calls = st.number_input("Total Evening Calls", min_value=0.0, max_value=500.0)
total_eve_charge= st.number_input("Total Evening Charge", min_value=0.0, max_value=500.0)
total_night_minutes = st.number_input("Total Night Minute", min_value=0.0, max_value=500.0)
total_night_calls  = st.number_input("Total Night Calls", min_value=0.0, max_value=500.0)
total_night_charge  = st.number_input("Total Night Charge", min_value=0.0, max_value=500.0)
total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, max_value=500.0)
total_intl_calls = st.number_input("Total International Calls", min_value=0.0, max_value=500.0)
total_intl_charge = st.number_input("Total International Charge", min_value=0.0, max_value=500.0)
customer_service_calls = st.number_input("Customer Service Calls", min_value=0.0, max_value=500.0)



#Create input features dataframe
if st.button("Submit"):
    input_features = pd.DataFrame({
    'account_length': [account_length],
    'international_plan': [1 if international_plan == 'yes' else 0],
    'voice_mail_plan': [1 if voice_mail_plan == 'yes' else 0],
    'number_vmail_messages': [number_vmail_messages],
    'total_day_minutes': [total_day_minutes],
    'total_day_calls': [total_day_calls],
    'total_day_charge': [total_day_charge],
    'total_eve_minutes':[total_eve_minutes],
    'total_eve_calls': [total_eve_calls],
    'total_eve_charge': [total_eve_charge],
    'total_night_minutes': [total_night_minutes],
    'total_night_calls': [total_night_calls],
    'total_night_charge': [total_night_charge],
   'total_intl_minutes': [total_intl_minutes],
   'total_intl_calls': [total_intl_calls],
   'total_intl_charge': [total_intl_charge],
   'customer_service_calls': [customer_service_calls]
   })


#Create prediction button
    st.write("Values Entered are:")
    st.write(input_features)
 
    prediction_proba = gbc_model.predict_proba(input_features.values)[:, 1][0]
    if prediction_proba > 0.5:
        prediction = "True"
    else:
        prediction = "False"
    st.write(f"Predicting Mobile Churn Analysis is:", prediction_proba)
 
    st.write("prediction:", prediction)
    st.success(f'predicted value:  {prediction}')

    
