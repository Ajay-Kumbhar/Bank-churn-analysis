import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('bankdata.csv')  # Loading dataset

with open('model.sav', 'rb') as pickle_file:  # Loading model
    model = pickle.load(pickle_file)

# Sidebar
st.sidebar.title("Bank Customer Churn Analysis")
st.sidebar.image('bank.jpg')
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Churn Analysis', 'Churn Prediction')
)

if user_menu == 'Churn Analysis':

    # Pie chart
    st.subheader('Percentage of churned and retained customers')
    labels = 'Churned', 'Retained'
    sizes = [df.churn[df['churn'] == 1].count(), df.churn[df['churn'] == 0].count()]
    explode = (0, 0.1)
    colors = ['#1faadb', '#80d4f2']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            startangle=90, colors=colors)
    ax1.axis('equal')
    st.pyplot(fig1)

    st.divider()

    # Histograms
    st.subheader('Churn vs credit score')
    fig2, ax2 = plt.subplots()
    sns.histplot(df, x='credit_score', hue='churn', kde=True, palette=['#1faadb', '#017336'])
    st.pyplot(fig2)

    st.divider()

    st.subheader('Churn vs Age')
    fig3, ax3 = plt.subplots()
    sns.histplot(df, x='age', hue='churn', kde=True, palette=['#1faadb', '#017336'])
    st.pyplot(fig3)

    st.divider()

    st.subheader('Churn vs Balance')
    fig4, ax4 = plt.subplots()
    sns.histplot(df, x='balance', hue='churn', kde=True, palette=['#1faadb', '#017336'])
    st.pyplot(fig4)

    st.divider()

    st.subheader('Churn vs Salary')
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.histplot(df, x='estimated_salary', hue='churn', kde=True, palette=['#1faadb', '#017336'])
    st.pyplot(fig5)

    st.divider()

    # Bar graphs
    st.subheader('Churn vs Country')
    fig6, ax6 = plt.subplots()
    sns.countplot(df, x='country', hue='churn', order=df['country'].value_counts().index,
                  palette=['#1faadb', '#80d4f2'])
    for label in ax6.containers:
        ax6.bar_label(label)
    st.pyplot(fig6)
    st.markdown('&#x2022; Germany has highest number of churners')

    st.divider()

    st.subheader('Churn vs Gender')
    fig7, ax7 = plt.subplots()
    sns.countplot(df, x='gender', hue='churn', order=df['gender'].value_counts().index,
                  palette=['#1faadb', '#80d4f2'])
    for label in ax7.containers:
        ax7.bar_label(label)
    st.pyplot(fig7)
    st.markdown('&#x2022; Female Churners are more than Male Churners')

    st.divider()

    st.subheader('Churn vs Credit card')
    fig8, ax8 = plt.subplots()
    sns.countplot(df, x='credit_card', hue='churn', order=df['credit_card'].value_counts().index,
                  palette=['#1faadb', '#80d4f2'])
    for label in ax8.containers:
        ax8.bar_label(label)
    st.pyplot(fig8)

    st.divider()

    st.subheader('Churn vs Active Member')
    fig9, ax9 = plt.subplots()
    sns.countplot(df, x='active_member', hue='churn', order=df['active_member'].value_counts().index,
                  palette=['#1faadb', '#80d4f2'])
    for label in ax9.containers:
        ax9.bar_label(label)
    st.pyplot(fig9)
    st.markdown('&#x2022; Most churners are not active members')

    st.divider()

    # Scatter plot
    st.header('Credit score vs Salary')
    fig10, ax10 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(df, x='estimated_salary', y='credit_score', hue='churn')
    st.pyplot(fig10)
    st.markdown('&#x2022; Customers with credit score < 400 are mostly churners')

    st.divider()

    st.header('Salary vs age')
    fig11, ax11 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(df, x='age', y='estimated_salary', hue='churn')
    st.pyplot(fig11)
    st.markdown('&#x2022; Most churners lie within the age group of 50-60 years')

if user_menu == 'Churn Prediction':

    st.subheader('Churn Prediction')
    st.subheader('')

    col1, col2, col3 = st.columns(spec=3, gap="large")

    with col1:

        country = st.selectbox(
            'Country',
            ['France', 'Germany', 'Spain'])

        st.header('')

        gender = st.radio(
            'Gender: ',
            ('Male', 'Female'), horizontal=True)

        st.header('')

        credit_score = st.text_input('Credit Score: ', value='', placeholder='Enter the credit score',
                                     label_visibility='visible')

        st.header('')

        salary = st.text_input('Salary: ', value='', placeholder='Enter your salary',
                               label_visibility='visible')

    with col2:

        age = st.text_input('Age: ', value='', placeholder='Please enter the age', label_visibility='visible')

        st.header('')

        credit_card = st.radio(
            'Credit Card: ',
            ('Yes', 'No'), horizontal=True)

        st.header('')

        product_number = st.text_input('Number of Products: ', value='', placeholder='Enter number of products',
                                       label_visibility='visible')

    with col3:

        tenure = st.text_input('Tenure: ', value='', placeholder='Enter tenure duration',
                               label_visibility='visible')

        st.header('')

        active_member = st.radio(
            'Active Member: ',
            ('Yes', 'No'), horizontal=True)

        st.header('')

        balance = st.text_input('Balance: ', value='', placeholder='Enter current balance',
                                label_visibility='visible')

    predict_button = st.button('Predict')

    if predict_button:

        credit_score = int(credit_score)

        if country == 'France':
            country = 0
        elif country == 'Germany':
            country = 1
        else:
            country = 2
        if gender == 'Male':
            gender = 1
        else:
            gender = 0

        age = int(age)
        tenure = int(tenure)
        balance = float(balance)
        product_number = int(product_number)

        if credit_card == 'Yes':
            credit_card = 1
        else:
            credit_card = 0

        if active_member == 'Yes':
            active_member = 1
        else:
            active_member = 0

        salary = float(salary)

        input_data = [[credit_score, country, gender, age, tenure, balance, product_number, credit_card, active_member,
                       salary]]
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)
        prediction = model.predict(input_data)

        if prediction[0] == 0:
            st.markdown(':green[Customer is not likely to churn]')
        else:
            st.markdown(':violet[Customer is likely to churn]')
