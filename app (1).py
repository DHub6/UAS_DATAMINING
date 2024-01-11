import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.datasets import make_friedman2
from scipy.sparse import dok_matrix
import seaborn as sns 
import pickle 

#import model
nb = pickle.load(open('GaussianNB.pkl','rb'))

#load dataset
data = pd.read_csv('Bank Customer Churn Dataset.csv')
data = data[['gender', 'tenure', 'balance', 'credit_card', 'active_member', 'churn']]
data['gender'].replace({'Male':1, 'Female':0}, inplace = True)

st.title('Aplikasi Bank Customer Churn')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Bank Customer Churn</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['GaussianNB']
option = st.sidebar.selectbox('Model',activities)
st.sidebar.header('Data Customer')

if st.checkbox("Tentang Dataset"):
  html_layout2 ="""
  <br>
  <p>Ini adalah dataset PIMA Indian</p>
  """
  st.markdown(html_layout2,unsafe_allow_html=True)
  st.subheader('Dataset')
  st.write(data.head(10))
  st.subheader('Describe dataset')
  st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
  pr =ProfileReport(data,explorative=True)
  st.header('**Input Dataframe**')
  st.write(data)
  st.write('---')
  st.header('**Profiling Report**')
  st_profile_report(pr)

#train test split
X = data.drop('churn',axis=1)
y = data['churn']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
  st.subheader('X_train')
  st.write(X_train.head())
  st.write(X_train.shape)
  st.subheader("y_train")
  st.write(y_train.head())
  st.write(y_train.shape)
  st.subheader('X_test')
  st.write(X_test.shape)
  st.subheader('y_test')
  st.write(y_test.head())
  st.write(y_test.shape)

def user_report():
  gender = st.sidebar.selectbox('gender', (0, 1))
  tenure = st.sidebar.slider('tenure',1,10, 7)
  balance = st.sidebar.slider('balance',3768,250898, 4000)
  credit_card = st.sidebar.slider('credit_card',0.7055,1.0000,0.9000)
  active_member = st.sidebar.slider('active_member',0.5151,1.0000,0.9000)
    
  user_report_data = {
    'gender':gender,
    'tenure':tenure,
    'balance':balance,
    'credit_card':credit_card,
    'active_member':active_member
  }
  report_data = pd.DataFrame(user_report_data,index=[0])
  return report_data

#Data Customer
user_data = user_report()
st.subheader('Data Customer')
st.write(user_data)

user_result = nb.predict(user_data)
svc_score = accuracy_score(y_test,nb.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
  output='Customer not Churn'
else:
  output ='Customer Churn'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(svc_score*100)+'%')