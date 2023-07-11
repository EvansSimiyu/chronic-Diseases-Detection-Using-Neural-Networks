# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importing libraries
import pickle
import streamlit as st     
import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

#saved models
diabetes_model =load_model("/home/evans/diseases_pred_cnn/diabetes_detection_cnn.h5")

heart_disease_model =load_model("/home/evans/diseases_pred_cnn/heart_disease_prediction_cnn.h5")
parkinson_disease_model =load_model("/home/evans/diseases_pred_cnn/parkinson_disease_pred_cnn.h5")
breast_cancer_model =load_model("/home/evans/diseases_pred_cnn/breast_cancer_classification_cnn.h5")

#scaler
scaler = StandardScaler()

with st.sidebar:
    selected = st.selectbox('Diseases Prediction Using CNN',
                            ('Diabetes Disease Detection',
                             'Heart Disease Prediction',
                             "Parkinson's Disease Prediction",
                             'Breast Cancer Disease Prediction'))
if (selected =='Diabetes Disease Detection'):
    st.title('Diabetes Disease Prediction Using CNN')
    col1,col2,col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Level')
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    with col2:
        Age = st.text_input('Age of the Patient')
        
        #Prediction
    diabetes_diagnosis = ''
    if st.button('Diabetes Disease Detection'):
        
        input_vals = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
        input_numeric = pd.to_numeric(list(input_vals),errors= 'coerce')
        input_np = np.asarray(input_numeric)
        input_reshaped = input_np.reshape(1,-1)
        input_standardized = scaler.fit_transform(input_reshaped)
        
        
        
        
        prediction = diabetes_model.predict(input_standardized)
        prediction_lbl = [np.argmax(prediction)][0]
        if prediction_lbl == 1:
            diabetes_diagnosis = 'The Patient is most likely to be DIABETIC'
        else:
            diabetes_diagnosis = 'The Patient most likely DIABETIC-FREE'
    st.success(diabetes_diagnosis)
            

#Heart Disease
elif (selected == 'Heart Disease Prediction'):
    st.title('Heart Disease Prediction Using CNN')
    col1,col2,col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Gender')
    with col3:
        cp = st.text_input('Chest Pain Type')
    with col1:
        trestbps = st.text_input('Rest Blood Pressure')
    with col2:
        chol = st.text_input('Cholesterol')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic Result')
    with col2:
        thalach = st.text_input('Maximum HeartRate/thalach')
    with col3:
        exang = st.text_input('Exang')
    with col1:
        oldpeak = st.text_input('Old Peak')
    with col2:
        slope = st.text_input('Heart Rate Slope')
    with col3:
        ca = st.text_input("Coranary Artery Calcium")
    with col1:
        thal = st.text_input('Thal')
        
    #Prediction
    heart_disease_diagnosis = ''
    if st.button('Heart Disease Prediction'):
        
        #data preparation
        input_vals = [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
        
        input_numeric = pd.to_numeric(list(input_vals),errors= 'coerce')
        input_np = np.asarray(input_numeric)
        input_reshaped = input_np.reshape(1,-1)
        input_standardized = scaler.fit_transform(input_reshaped)
        
        
        
        prediction = heart_disease_model.predict(input_standardized)
        pred_lbl = [np.argmax(prediction)][0]
        if pred_lbl == 1:
            heart_disease_diagnosis = 'Defective Heart'
        else:
            heart_disease_diagnosis = 'Healthy Heart'
    st.success(heart_disease_diagnosis)
    


elif (selected == "Parkinson's Disease Prediction"):
    st.title("Parkinson's Disease Prediction Using CNN")
    col1,col2,col3= st.columns(3)
    
    with col1:
        MDVP_to_Fo_Ration = st.text_input('MDVP_to_Fo(Hz)_Ration')    
    with col2:
        MDVP_to_Fhi_Ratio = st.text_input('MDVP_to_Fhi(Hz)_Ratio')
    with col3:
        MDVP_to_Flo_Ratio = st.text_input('MDVP_to_Flo(Hz)_Ratio')
    with col1:
        MDVP_jitter_Percentage = st.text_input('MDVP_jitter_Percentage')
    with col2:
         MDVP_to_Jitter_Ratio = st.text_input('MDVP_to_Jitter(Abs)_Ratio')
    with col3:
        MDVP_to_RAP_Ratio = st.text_input('MDVP_to_RAP_Ratio')
    with col1:
        MDVP_to_PPQ_Ratio = st.text_input('MDVP_to_PPQ_Ratio')
    with col2:
        Jitter_to_DDP_Ratio = st.text_input('Jitter_to_DDP_Ratio')
    with col3:
        MDVP_to_Shimmer_Ratio_Abs = st.text_input('MDVP_to_Shimmer_Ratio')
    with col1:
        MDVP_to_Shimmer_dB = st.text_input('MDVP_to_Shimmer(dB)')
    with col2:
        Shimmer_to_APQ3_Ratio = st.text_input('Shimmer_to_APQ3_Ratio')
    with col3:
        Shimmer_to_APQ5_Ratio = st.text_input('Shimmer_to_APQ5_Ratio')
    with col1:
        MDVP_to_APQ_Ratio = st.text_input('MDVP_to_APQ_Ratio')
    with col2:
        Shimmer_to_DDA_Ratio = st.text_input('Shimmer_to_DDA_Ratio')
    with col3:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col1:
        spread1 =st.text_input('spread1')
    with col2:
        spread2 = st.text_input('spread2')
    with col3:
        D2 = st.text_input('D2')
    with col1:
        PPE = st.text_input('PPE')
    
    
    #Prediction
    parkinson_diagnosis = ''
    if st.button("Parkinson's Disease Prediction"):
        
        #Data Preparation
        input_vals = [[MDVP_to_Fo_Ration,MDVP_to_Fhi_Ratio,MDVP_to_Flo_Ratio,MDVP_jitter_Percentage,MDVP_to_Jitter_Ratio,
                  MDVP_to_RAP_Ratio,MDVP_to_PPQ_Ratio,Jitter_to_DDP_Ratio,MDVP_to_Shimmer_Ratio_Abs,MDVP_to_Shimmer_dB,
                  Shimmer_to_APQ3_Ratio,Shimmer_to_APQ5_Ratio,MDVP_to_APQ_Ratio,Shimmer_to_DDA_Ratio,NHR,HNR,RPDE,DFA,spread1,
                  spread2,D2,PPE]]
        
        
        input_numeric = pd.to_numeric(list(input_vals),errors= 'coerce')
        input_np = np.asarray(input_numeric)
        input_reshaped = input_np.reshape(1,-1)
        input_standardized = scaler.fit_transform(input_reshaped)
        
        
        prediction = parkinson_disease_model.predict(input_standardized)
        
        pred_lbl = [np.argmax(prediction)][0]
        if pred_lbl == 1:
            parkinson_diagnosis = "Parkinson's Disease DETECTED"
        else:
            parkinson_diagnosis = "NO Parkinson's Disease Detected"
    st.success(parkinson_diagnosis)
else:
    st.title('Breast Cancer Classification Using CNN')
    
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        radius_mean = st.text_input('radius_mean')
    with col2:
        texture_mean = st.text_input('texture_mean')
    with col3:
        perimeter_mean = st.text_input('perimeter_mean')
    with col4:
        area_mean = st.text_input('area_mean')
    with col1:
        smoothness_mean = st.text_input('smoothness_mean')
    with col2:
        compactness_mean = st.text_input('compactness_mean')
    with col3:
        concavity_mean = st.text_input('concavity_mean')
    with col4:
        concave_pts_mean = st.text_input('concave_pts_mean')
    with col1:
        symmetry_mean = st.text_input('symmetry_mean')
    with col2:
        fractal_dim_mean = st.text_input('fractal_dim_mean')
    with col3:
        radius_se = st.text_input('radius_se')
    with col4:
        texture_se = st.text_input(' texture_se')
    with col1:
        perimeter_se = st.text_input('perimeter_se')
    with col2:
        area_se = st.text_input('area_se')
    with col3:
        smoothness_se = st.text_input('smoothness_se')
    with col4:
        compactness_se = st.text_input('compactness_se')
    with col1:
        concavity_se = st.text_input('concavity_se')
    with col2:
        concave_pts_se = st.text_input('concave_pts_se')
    with col3:
        symmetry_se = st.text_input('symmetry_se')
    with col4:
        fractal_dim_se = st.text_input('fractal_dim_se')
    with col1:
        radius_worst = st.text_input('radius_worst')
    with col2:
        texture_worst = st.text_input('texture_worst')
    with col3:
        perimeter_worst = st.text_input ('perimeter_worst')
    with col4:
        area_worst = st.text_input('area_worst')
    with col1:
        smoothness_worst = st.text_input('smoothness_worst')
    with col2:
        compactness_worst = st.text_input('compactness_worst')
    with col3:
        concavity_worst = st.text_input('concavity_worst')
    with col4:
        concave_pts_worst = st.text_input('concave_pts_worst')
    with col1:
        symmetry_worst = st.text_input('symmetry_worst')
    with col2:
        fractal_dim_worst = st.text_input('fractal_dim_worst')
        
        
        #Prediction
    breast_cancer_diagnosis =''
    if st.button('Breast Cancer Classification'):
        
        #Data Preparation
        
        input_vals = [[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, 
                       concavity_mean, concave_pts_mean, symmetry_mean, fractal_dim_mean, radius_se, 
                       texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, 
                       concave_pts_se, symmetry_se, fractal_dim_se, radius_worst, texture_worst, 
                       perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
                       concave_pts_worst, symmetry_worst, fractal_dim_worst]]
        input_vals_numeric = pd.to_numeric(list(input_vals),errors='coerce')
        input_np = np.asarray(input_vals_numeric)
        input_reshaped = input_np.reshape(1,-1)
        input_standardized = scaler.fit_transform(input_reshaped)
      
        
        prediction = breast_cancer_model.predict(input_standardized)
        pred_label = [np.argmax(prediction)][0]
        if pred_label == 0:
            breast_cancer_diagnosis = 'Benign Breast Cancer'
        else:
            breast_cancer_diagnosis = 'Malignant Breast Cancer'
            
    st.success(breast_cancer_diagnosis)
