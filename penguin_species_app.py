import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

@st.cache
def prediction(model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex):
  species=model.predict([[island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]])
  species=species[0]
  if species==1:
    return "Adelie".upper()
  elif species==2:
    return "Chinstrap".upper()
  else:
    return "Gentoo".upper()

st.title("Penguin species prediction")
bill_length_mm=st.slider("Beak length",32.0,60.0)
bill_depth_mm=st.slider("Beak depth",13.0,22.0)
flipper_length_mm=st.slider("Fin length",172.0,231.0)
body_mass_g=st.slider("Body mass",2700.0,6300.0)
pen_sex=st.sidebar.selectbox("Select the sex of the penguin : ",("male","female"))
sex=0
if pen_sex=="female":
  sex+=1
pen_island=st.sidebar.selectbox("Select the island the penguin comes from : ",('Biscoe','Dream','Torgersen'))
island=0
if pen_island=="Dream":
  island+=1
else:
  island+=2
classifier=st.sidebar.selectbox("Select the classifier you want to use : ",("SVC","LogisticRegression","RandomForestClassifier"))
if classifier=="SVC":
  if st.button("Predict"):
  	species_type_s=prediction(svc_model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex)
  	st.write("Species predicted : ",species_type_s)
  	st.write("Accuracy score of this model is : ", round(svc_score,3))
if classifier=="LogisticRegression":
  if st.button("Predict"):
  	species_type_l=prediction(log_reg,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex)
  	st.write("Species predicted : ",species_type_l)
  	st.write("Accuracy score of this model is : ", round(log_reg_score,3))
if classifier=="RandomForestClassifier":
  if st.button("Predict"):
  	species_type_r=prediction(rf_clf,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex)
  	st.write("Species predicted : ",species_type_r)
  	st.write("Accuracy score of this model is : ", round(rf_clf_score,3))