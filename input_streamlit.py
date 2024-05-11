# -*- coding: utf-8 -*-
"""
Created on Sat May 11 07:37:49 2024

@author: SRI VAISHNAVI
"""



import pickle
import streamlit as st

import joblib, pickle


# loading the saved models

model1 = pickle.load(open('knn.pkl', 'rb'))
impute = joblib.load('meanimpute')
winzor = joblib.load('winsor')
minmax = joblib.load('minmax')
encode = joblib.load('encoding')



    st.title("Machine Fault Prediction")


    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Hydraulic_pressure = st.text_input('Hydraulic_pressure(bar)')
        
    with col2:
        Coolant_pressure = st.text_input('Coolant_pressure(bar)')
    
    with col3:
        Air_systemPressure = st.text_input('Air System Pressure(bar) ')
    
    with col1:
        Coolant_Temperature = st.text_input('Coolant temperature')
    
    with col2:
        HydraulicOiltemperature = st.text_input('Hydraulic oil Temperature')
    
    with col3:
        ProximitySensors = st.text_input('Proximity sensors')
    
    with col1:
        Torque = st.text_input('Torque')
    
    with col2:
        Voltage = st.text_input('Voltage(volts)')
    with col3:
        CuttingForce=st.text_input('Cutting_Force(kns)')

    
    
    # code for Prediction
    Machine_downtime_detection = ''
    
    # creating a button for Prediction
    
    if st.button('Predict'):
        Machine_failuredetection_prediction = model1.predict([[Hydraulic_pressure, Coolant_pressure, Air_systemPressure, Coolant_Temperature, HydraulicOiltemperature, ProximitySensors, Torque, Voltage, CuttingForce]])
        
        if (Machine_failuredetection_prediction[0] == 1):
          Machine_downtime_detection = 'Machine_downtime'
        else:
          Machine_downtime_detection = 'No_machine_downtime'
        
    st.success(Machine_downtime_detection)




















