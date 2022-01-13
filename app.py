import streamlit as st
import numpy as np
import pandas as pd
import pickle
df=pd.read_csv('Cleaned_Car_data.csv')

st.title('Car Price Prediction')


Name = st.selectbox(
'Name of the Car',
(df['name'].values))

Company = st.selectbox(
'Manufacturing Company',
(df['company'].values))



model=pickle.load(open('LinearRegressionModel.pkl','rb'))

#Name = st.text_input("name")
#Company = st.text_input("company")
st.text('* make sure to use same manufacturer')
#Year = st.text_input("year")
Year = st.selectbox(
'year',
sorted((df['year'].unique()))
)





Kms_driven = st.text_input("kms_driven")
#Fuel_type = st.text_input("fuel_type")

Fuel_type = st.selectbox(
'fuel_type',
(df['fuel_type'].unique()))



#y=[str(Name),str(Company),Year,Kms_driven,str(Fuel_type)]

if st.button('Predict'):
    # preprocessing

    result = model.predict(
        pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],
                     data=np.array([str(Name),str(Company),Year,Kms_driven,str(Fuel_type)]).reshape(1,5)))
    st.header('Predicted Value in INR')
    st.header(result[0])


st.text('Project by Sandeep')


