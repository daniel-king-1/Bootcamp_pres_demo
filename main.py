import streamlit as st
import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import os
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
shap.initjs()

st.title("""
Daniel King Bootcamp Demo
  """)
st.subheader("Welcome!\n\n")

path = os.path.abspath('adult.data')
#st.file_uploader(path, type='.data')

df = pd.read_csv('adult.data', names=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_stat', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'captial-loss', 'hours-week', 'country', 'income'])

st.info("Census Income Data (Predict if individual makes less than or at least $50k a year)")
st.dataframe(df.head(10))

def encode_(cl):
    d = {}
    ret = cl
    num = 0
    for i in range(0,len(cl)):
        v = str(cl[i])
        if v in d.keys():
            ret[i] = d[v]            
        else:
            d[v] = num 
            ret[i] = num
            num+=1
    return ret

for i in df.columns:
    if i in ['age', 'fnlwgt', 'education_num', 'capital-gain', 'captial-loss', 'hours-week']:
        continue
    n = df[i].values
    encode_(n)
    df[i] = n

X = df.drop('income', axis=1)
Y = df['income']
X=X.astype('int')
Y=Y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, Y)

model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.info("Logistic Regression Model")
st.write("Accuracy of Logistic Regression: ", accuracy_score(y_test, y_pred))
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
st_shap(shap.plots.bar(shap_values))
st_shap(shap.plots.force(shap_values[0]))

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.info("XGBoost Model")
st.write("Accuracy of XGBoost Classification: ", accuracy_score(y_test, y_pred))
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
st_shap(shap.plots.bar(shap_values))
st_shap(shap.plots.force(shap_values[0]))