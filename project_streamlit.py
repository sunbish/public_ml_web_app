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
import pymysql

from sqlalchemy import create_engine
import joblib, pickle

model1 = pickle.load(open('knn.joblib'))

impute = joblib.load('meanimpute')
winzor = joblib.load('winsor')



def predict_Downtime( data, user, pw, db):

    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
    data= data.drop(columns = ['Date','Machine_ID', 'Assembly_Line_No'])

    
    clean = pd.DataFrame(impute.transform(data), columns = data.columns)
    clean1 = pd.DataFrame(winzor.transform(clean), columns = data.columns)
    clean2 = pd.DataFrame(minmax.transform(clean1), columns = data.columns)
    
    
    prediction = pd.DataFrame(model1.predict(clean), columns = ['Machine_Fault_Prediction'])

    final = pd.concat([prediction, data], axis = 1)
        
    final.to_sql('KNN_test', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

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
     

        result = predict_Downtime(data, user, pw, db)
                           
        import seaborn as sns
        cm = sns.light_palette("yellow", as_cmap = True)
        st.table(result)

     
                           
if __name__=='__main__':
    main()

