import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

with open('model.sav', 'rb') as pickle_file:  # Loading model
    model = pickle.load(pickle_file)

df = pd.read_csv('bankdata.csv')
df = df.drop(['customer_id', 'churn'], axis=1)

one = 0
zero = 0

for index, row in df.iterrows():
    credit_score = int(row['credit_score'])
    if row['country'] == 'France':
        country = 0
    elif row['country'] == 'Germany':
        country = 1
    else:
        country = 2
    gender = 1 if row['gender'] == 'Male' else 0
    age = int(row['age'])
    tenure = int(row['tenure'])
    balance = float(row['balance'])
    products_number = int(row['products_number'])
    credit_card = int(row['credit_card'])
    active_member = int(row['active_member'])
    estimated_salary = float(row['estimated_salary'])
    input_data = [[credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member,
                   estimated_salary]]
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)
    prediction = model.predict(input_data)
    if prediction == 1:
        one += 1
    else:
        zero += 1

print('zero=', zero, ' one=', one)
