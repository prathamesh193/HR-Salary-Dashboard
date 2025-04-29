# hr_salary_dashboard.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt

# ----------------------------
# 1. DATA CLEANING FUNCTION
# ----------------------------
def clean_data():
    """Load and clean HR dataset"""
    df = pd.read_csv('HRDataset_v14.csv', encoding='utf-8-sig')
    
    # Keep essential columns
    cols = ['Employee_Name', 'Age', 'Salary', 'Position', 'DateofHire', 'Department']
    df = df[cols].rename(columns={'Employee_Name': 'Name', 'Salary': 'Current_Salary'})
    
    # Handle missing data
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Current_Salary'].fillna(df['Current_Salary'].median(), inplace=True)
    
    # Calculate experience
    df['DateofHire'] = pd.to_datetime(df['DateofHire'], errors='coerce')
    df['Years_of_Experience'] = (datetime.now() - df['DateofHire']).dt.days / 365.25
    df['Years_of_Experience'] = df['Years_of_Experience'].round(1)
    
    # Remove outliers
    df = df[(df['Age'] >= 18) & (df['Age'] <= 70)]
    salary_cap = df['Current_Salary'].quantile(0.99)
    df['Current_Salary'] = np.where(df['Current_Salary'] > salary_cap, salary_cap, df['Current_Salary'])
    
    return df[['Name', 'Age', 'Years_of_Experience', 'Current_Salary', 'Position']]

# ----------------------------
# 2. MODEL TRAINING FUNCTION
# ----------------------------
def train_model(df):
    """Train and save salary prediction model"""
    X = df[['Years_of_Experience']]
    y = df['Current_Salary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save model for later use
    joblib.dump(model, 'salary_model.joblib')
    return model

# ----------------------------
# 3. STREAMLIT DASHBOARD
# ----------------------------
def main():
    st.title("HR Salary Prediction Dashboard")
    st.write("Predict salaries based on experience and role")
    
    # Load or clean data
    try:
        df = pd.read_csv('cleaned_hr_data.csv')
    except:
        df = clean_data()
        df.to_csv('cleaned_hr_data.csv', index=False)
    
    # Train or load model
    try:
        model = joblib.load('salary_model.joblib')
    except:
        model = train_model(df)
    
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        experience = st.slider("Years of Experience", 0.0, 30.0, 5.0, 0.5)
    with col2:
        current_salary = st.number_input("Current Salary ($)", 30000, 200000, 50000)
    
    # Prediction
    predicted_salary = model.predict([[experience]])[0]
    difference = predicted_salary - current_salary
    
    # Display results
    st.subheader("Prediction Results")
    st.metric("Predicted Salary", f"${predicted_salary:,.0f}")
    st.metric("Salary Difference", 
              f"${abs(difference):,.0f}", 
              "Higher" if difference > 0 else "Lower")
    
    # Visualization
    fig, ax = plt.subplots()
    ax.scatter(df['Years_of_Experience'], df['Current_Salary'], alpha=0.3)
    ax.plot([0, 30], 
            [model.intercept_, model.intercept_ + 30 * model.coef_[0]], 
            'r-', linewidth=2)
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary ($)")
    st.pyplot(fig)
    
    # Data table
    st.subheader("Sample HR Data")
    st.dataframe(df.head(10))

if __name__ == "__main__":
    main()