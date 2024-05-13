# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:34:52 2024

@author: SRI VAISHNAVI
"""
#pip install sqlalchemy

import sqlalchemy
import pickle
import pandas as pd
import streamlit as st 



from sqlalchemy import create_engine
import joblib, pickle

model1 = pickle.load('knn.pkl')
impute = joblib.load('meanimpute')
winzor = joblib.load('winsor')
minmax = joblib.load('minmax')
encode = joblib.load('encoding')


def preprocess_data(data):

    columns_tranform=['Hydraulic_Pressure(bar)','Coolant_Pressure(bar)','Air_System_Pressure(bar)','Coolant_Temperature','	Hydraulic_Oil_Temperature(Â°C)','Spindle_Bearing_Temperature(Â°C)','Spindle_Vibration(Âµm)','Tool_Vibration(Âµm)','Spindle_Speed(RPM)','Voltage(volts)','Torque(Nm)','Cutting(kN)','Downtime']
    clean = pd.DataFrame(impute.transform(data), columns = columns_tranform)
    clean1 = pd.DataFrame(winzor.transform(clean), columns = columns_tranform)
    clean2 = pd.DataFrame(minmax.transform(clean1), columns = columns_tranform)
    
def predict_downtime(original, data, user, pw, db):
    
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

    prediction = pd.DataFrame(model1.predict(data), columns = ['machin_failure_pred'])
    
    final = pd.concat([prediction, original], axis = 1)
        
    final.to_sql('knn_test', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return final


def main():

    st.title("Machine Fault Prediction")
    st.sidebar.title("Machine Fault Prediction")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Machine Fault Prediction</h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Upload a file" , type = ['csv','xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
        
        
    else:
        st.sidebar.warning("Upload a CSV or Excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        preprocessed_data=preprocess_data()
        result = predict_downtime(data, preprocessed_data, user, pw, db)
                           
        import seaborn as sns
        cm = sns.light_palette("yellow", as_cmap = True)
        st.table(result)

                           
if __name__=='__main__':
 main()

