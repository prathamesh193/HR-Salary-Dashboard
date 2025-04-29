from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
import joblib
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load and preprocess data
def load_data():
    df = pd.read_csv(r'C:\VS CODE\tcs_rio\HRDataset_v14.csv')
    
    # Feature engineering
    df['YearsAtCompany'] = (pd.to_datetime('now') - pd.to_datetime(df['DateofHire'])).dt.days / 365
    df['Age'] = (pd.to_datetime('now') - pd.to_datetime(df['DOB'])).dt.days / 365
    
    # Encode categorical variables
    categorical_cols = ['Department', 'Position', 'State', 'MaritalDesc', 'RaceDesc', 'Sex', 'RecruitmentSource']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Save label encoders for prediction
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    return df, label_encoders

df, label_encoders = load_data()

# Prepare features and target
X = df[['Department', 'Position', 'State', 'YearsAtCompany', 'Age', 'EngagementSurvey', 
        'EmpSatisfaction', 'SpecialProjectsCount', 'MaritalDesc', 'RaceDesc', 'Sex']]
y = df['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'salary_predictor.pkl')

# Generate plots
def create_plots():
    plots = {}
    
    # Salary distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Salary'], kde=True)
    plt.title('Salary Distribution')
    plt.xlabel('Salary')
    plt.ylabel('Count')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['salary_dist'] = base64.b64encode(buf.read()).decode('ascii')
    plt.close()
    
    # Department vs Salary
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Department', y='Salary', data=df)
    plt.xticks(rotation=45)
    plt.title('Salary by Department')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['dept_salary'] = base64.b64encode(buf.read()).decode('ascii')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['corr_heatmap'] = base64.b64encode(buf.read()).decode('ascii')
    plt.close()
    
    # Performance metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='r2'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('R² Score')
    plt.title('Learning Curve')
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['learning_curve'] = base64.b64encode(buf.read()).decode('ascii')
    plt.close()
    
    # Feature importance
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Plot')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['feature_importance'] = base64.b64encode(buf.read()).decode('ascii')
    plt.close()
    
    return plots, {'mse': mse, 'r2': r2}

plots, metrics = create_plots()

@app.route('/')
def dashboard():
    return render_template('dashboard.html', plots=plots, metrics=metrics, 
                          data_head=df.head().to_html(classes='data-table'),
                          data_description=df.describe().to_html(classes='data-table'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Ensure label_encoders are available
    if not os.path.exists('label_encoders.pkl'):
        load_data()  # This will create the label_encoders.pkl file
    
    label_encoders = joblib.load('label_encoders.pkl')
    
    # Prepare default form data
    form_data = {
        'department': '',
        'position': '',
        'state': '',
        'years_at_company': '5',
        'age': '30',
        'engagement': '3.5',
        'satisfaction': '4',
        'projects': '3',
        'marital_status': '',
        'race': '',
        'gender': ''
    }
    
    if request.method == 'POST':
        try:
            # Get form data
            form_data = request.form.to_dict()
            
            # Convert to model input format
            input_data = {
                'Department': label_encoders['Department'].transform([form_data['department']])[0],
                'Position': label_encoders['Position'].transform([form_data['position']])[0],
                'State': label_encoders['State'].transform([form_data['state']])[0],
                'YearsAtCompany': float(form_data['years_at_company']),
                'Age': float(form_data['age']),
                'EngagementSurvey': float(form_data['engagement']),
                'EmpSatisfaction': int(form_data['satisfaction']),
                'SpecialProjectsCount': int(form_data['projects']),
                'MaritalDesc': label_encoders['MaritalDesc'].transform([form_data['marital_status']])[0],
                'RaceDesc': label_encoders['RaceDesc'].transform([form_data['race']])[0],
                'Sex': label_encoders['Sex'].transform([form_data['gender']])[0]
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure we have all expected columns (fill missing with 0)
            for col in X_train.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match training data
            input_df = input_df[X_train.columns]
            
            # Predict
            prediction = model.predict(input_df)[0]
            
            return render_template('prediction.html', 
                                 prediction=f"${prediction:,.2f}",
                                 form_data=form_data,
                                 label_encoders=label_encoders)
            
        except Exception as e:
            return render_template('prediction.html', 
                                 error=str(e),
                                 form_data=form_data,
                                 label_encoders=label_encoders)
    
    # For GET request, show form with default values
    return render_template('prediction.html',
                         form_data=form_data,
                         label_encoders=label_encoders)

@app.route('/data-analysis')
def data_analysis():
    # Create interactive plots with Plotly
    # Department vs Salary
    dept_fig = px.box(df, x='Department', y='Salary', 
                     title='Salary Distribution by Department')
    dept_plot = dept_fig.to_html(full_html=False)
    
    # Experience vs Salary
    exp_fig = px.scatter(df, x='YearsAtCompany', y='Salary', 
                        color='Department', 
                        title='Salary vs Years at Company')
    exp_plot = exp_fig.to_html(full_html=False)
    
    # Age vs Salary
    age_fig = px.scatter(df, x='Age', y='Salary', 
                        color='Position', 
                        title='Salary vs Age by Position')
    age_plot = age_fig.to_html(full_html=False)
    
    # Gender pay gap
    gender_fig = px.box(df, x='Sex', y='Salary', 
                       color='Department',
                       title='Gender Pay Gap Analysis by Department')
    gender_plot = gender_fig.to_html(full_html=False)
    
    return render_template('analysis.html',
                         dept_plot=dept_plot,
                         exp_plot=exp_plot,
                         age_plot=age_plot,
                         gender_plot=gender_plot)

if __name__ == '__main__':
    app.run(debug=True)