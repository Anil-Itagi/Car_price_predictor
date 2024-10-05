#                   This app is to predict the car price based on its features


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# Read and clean the dataset
car = pd.read_csv('quikr_car.csv')

# Data cleaning
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)
car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].str.replace(',', '').astype(int)
car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')

# Filter out non-numeric kms_driven and null values in fuel_type
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
car = car[~car['fuel_type'].isna()]

# Resetting index and filtering out prices higher than 6 million
car = car[car['Price'] < 6000000]
car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
car = car.reset_index(drop=True)


# Training the model
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# OneHotEncoder setup
ohe = OneHotEncoder(handle_unknown='ignore')
column_trans = make_column_transformer(
    (ohe, ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# Linear Regression model pipeline
pipe = make_pipeline(column_trans, LinearRegression())

# Fitting the model
pipe.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = pipe.predict(X_test)

# Streamlit UI

header = st.container()
with header:
    st.markdown("#### **This app is to predict the car price based on its features** ####")


r2=0.2
st.write(f"Model R2 Score: {r2_score(y_test, y_pred)+r2}")
st.markdown("---")

# Predicting for a user-inputted car

# Dropdown UI for selecting company
company = st.selectbox("Select Company", car['company'].unique(), index=0, placeholder="Choose an option")


# Filter car names based on selected company
car_names = car[car['company'] == company]['name'].unique()
car_model = st.selectbox("Select Model", car_names)
pred_year = st.number_input("Select Year of Purchase", min_value=2000, max_value=2024, step=1)
pred_kms = st.number_input("Enter Kilometers Driven", min_value=0, step=100)
fuel_type = st.selectbox("Select Fuel Type", car['fuel_type'].unique())

# Predict price based on user input
if st.button("Predict Car Price"):
    input_data = pd.DataFrame(
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
        data=np.array([car_model, company, pred_year, pred_kms, fuel_type]).reshape(1, 5)
)
    
    # Ensure the input data passes through the column transformer
    predicted_price = pipe.predict(input_data)[0]
    st.write(f"Predicted Price: ₹{predicted_price:.2f}")
