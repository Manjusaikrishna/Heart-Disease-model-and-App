#!/usr/bin/env python
# coding: utf-8

# In[18]:


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import requests
import io

st.title("""
Heart Disease Prediction App
This app predicts whether the person having given values of parameters based on tests is having heart disease or not type
""")
st.write("""
Attribute documentation:\n
1. age: age in years\n
2. sex: sex (1 = male; 0 = female)\n
3. cp: chest pain type\n
-- Value 1: typical angina
-- Value 2: atypical angina
-- Value 3: non-anginal pain
-- Value 4: asymptomatic
4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)\n
5. chol: serum cholestoral in mg/dl\n
6. fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)\n
7. restecg: resting electrocardiographic results\n
-- Value 0: normal
-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
-- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
8. thalach: maximum heart rate achieved\n
9. exang: exercise induced angina (1 = yes; 0 = no)\n
10. oldpeak = ST depression induced by exercise relative to rest\n
11. slope: the slope of the peak exercise ST segment\n
-- Value 1: upsloping
-- Value 2: flat
-- Value 3: downsloping
12. ca: number of major vessels (0-3) colored by flourosopy\n
13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect\n
14. num: diagnosis of heart disease (angiographic disease status)
-- Value 0: < 50% diameter narrowing
-- Value 1: > 50% diameter narrowing
"""
)

st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("age",20,100,55)
    sex = st.sidebar.selectbox("Select the Gender",("Male", "Female"))
    if sex =='Male':
        gender = 1 
    else :
        gender = 0
    cp=st.sidebar.slider("cp",0,3,2)
    trestbps=st.sidebar.slider("trestbps",94,200,150)
    chol=st.sidebar.slider("chol",126,138,128)
    fbs=st.sidebar.slider("fbs",0,1,0)
    restecg=st.sidebar.slider("restecg",0,2,1)
    thalach=st.sidebar.slider("thalach",71,202,86)
    exang=st.sidebar.slider("exang",0,1,0)
    oldpeak=st.sidebar.slider("oldpeak",0.0,6.2,2.3)
    slope=st.sidebar.slider("slope",0,2,1)
    ca=st.sidebar.slider("ca",0,4,3)
    thal=st.sidebar.slider("thal",0,3,1)
    data = {"age": age,
            "sex": gender,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal}
    features = pd.DataFrame(data,index=[0])
    return features
df = user_input_features()

st.subheader("User Input Parameters")
st.write(df)

url = "https://raw.githubusercontent.com/Manjusaikrishna/Heart-Disease-model-and-App/master/heart_disease.csv" # Make sure the url is the raw version of the file on GitHub
download = requests.get(url).content
heart_disease = pd.read_csv(io.StringIO(download.decode('utf-8')))

# create X(features matrix)
x = heart_disease.drop("target",axis=1)

# create Y(labels)
y = heart_disease["target"]

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression(max_iter=4000).fit(x,y)

prediction = logreg.predict(df)
prediction_proba = logreg.predict_proba(df)

st.subheader("Class label if heart disease is present or not and its corresponding index number")
st.write(heart_disease.target.unique()[::-1])

st.subheader("Prediction")
st.write(heart_disease.target.unique()[::-1][prediction])
#st.write(prediction)

st.subheader("Prediction Probablity")
st.write(prediction_proba)


# In[ ]:




