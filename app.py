import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Data Simulation (as the original CSV is not provided) ---
# Simulating the data structure and content based on the PDF
# In a real-world scenario, you would upload the 'Salary Data.csv' file.
@st.cache_data
def load_and_prepare_data():
    # Simulating a subset of the data based on the PDF's head() and info()
    data = {
        'Age': [32.0, 28.0, 45.0, 36.0, 52.0, 25.0, 30.0],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
        'Education Level': ["Bachelor's", "Master's", 'PhD', "Bachelor's", "Master's", "Bachelor's", "Master's"],
        'Job Title': ['Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate', 'Director', 'Junior Developer', 'Marketing Analyst'],
        'Years of Experience': [5.0, 3.0, 15.0, 7.0, 20.0, 1.0, 4.0],
        'Salary': [90000.0, 65000.0, 150000.0, 60000.0, 200000.0, 45000.0, 70000.0]
    }
    saldf = pd.DataFrame(data)

    # Introduce some nulls to simulate the PDF's null count for testing the dropna() part
    saldf.loc[6, 'Salary'] = np.nan
    saldf.loc[6, 'Years of Experience'] = np.nan
    
    # Selecting input (inp) and output (out) columns as done in the PDF
    inp = saldf[['Years of Experience']]
    out = saldf['Salary']

    # Combining and dropping rows with missing values (dropna) as shown in the PDF
    Train_data = pd.concat([inp, out], axis=1)
    train = Train_data.dropna()
    
    inpl = train[['Years of Experience']] # Input for training 
    outl = train['Salary']               # Output for training 
    
    return inpl, outl, train

# --- Model Training ---
@st.cache_resource
def train_linear_regression_model(inpl, outl):
    LR = LinearRegression() # Initialize the model
    LR.fit(inpl, outl)      # Train the model
    return LR

# --- Streamlit Application Layout ---
def main():
    st.title('Salary Prediction using Linear Regression')
    st.markdown('***Based on Analysis from PDF Document***')
    
    # Load and Train Data
    inpl, outl, train = load_and_prepare_data()
    
    # Add a safety check to prevent training on empty data
    if train.empty:
        st.error("Error: Training data is empty after removing missing values. Cannot proceed with prediction.")
        return
        
    LR = train_linear_regression_model(inpl, outl)
    
    # --- Prediction Sidebar ---
    st.sidebar.header('Salary Predictor')
    
    # User input for 'Years of Experience'
    # Default is 5, as it was the prediction example in the PDF
    years_experience = st.sidebar.slider(
        'Years of Experience (Input)', 
        min_value=0.0, 
        max_value=25.0, 
        value=5.0, 
        step=0.5
    )

    # --- Make Prediction ---
    try:
        # FIX: Create a DataFrame for prediction input to match training data structure.
        # FIX: Safely access the first (and only) element of the 1D prediction array using [0].
        input_data = pd.DataFrame({'Years of Experience': [years_experience]})
        predicted_salary = LR.predict(input_data)[0]
    
    except Exception as e:
        st.error(f"Prediction Error: The model failed to predict a salary. Check input data or model training. Error: {e}")
        return
        
    # --- Display Prediction ---
    st.sidebar.subheader('Predicted Salary')
    st.sidebar.markdown(f"**${predicted_salary:,.2f}**")
    
    st.sidebar.markdown('---')
    st.sidebar.caption(f'Predicted value for 5 years of experience in PDF: $66,143.77') # Based on Out[23]

    # --- Main Content: Visualization and Data ---
    
    st.header('Model Visualization')
    st.markdown('The trained Linear Regression model (line) fitted to the training data (dots).')
    
    # Generate the regression line for plotting
    X_plot = np.array([train['Years of Experience'].min(), train['Years of Experience'].max()]).reshape(-1, 1)
    Y_plot = LR.predict(X_plot)
    
    # Plotting the data and the regression line 
    fig, ax = plt.subplots()
    ax.scatter(train['Years of Experience'], train['Salary'], color='blue', label='Training Data')
    ax.plot(X_plot, Y_plot, color='red', label='Regression Line')
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel('Salary')
    ax.set_title('Linear Regression Model Fit')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    st.header('Training Data Sample')
    st.dataframe(train)

if __name__ == '__main__':
    main()
