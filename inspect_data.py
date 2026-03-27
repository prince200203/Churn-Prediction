import pandas as pd
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
features = ['InternetService', 'Contract', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineBackup', 'DeviceProtection', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
for col in features:
    print(f"{col}: {df[col].unique().tolist()}")
