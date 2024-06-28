import pickle
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Load the models
with open('decision_tree_model.pickle', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('logistic_regression_model.pickle', 'rb') as file:
    logistic_regression_model = pickle.load(file)

with open('naive_bayes_model.pickle', 'rb') as file:
    naive_bayes_model = pickle.load(file)

# Apply custom CSS
def apply_custom_css():
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Define the pages
def home_page():
    st.title("🏠 Online Fraud Detection")
    st.image("contoh.jpg", use_column_width=True)
    st.write("""
    Welcome to the Online Fraud Payment Detection app. This application helps you predict whether a financial transaction is fraudulent or not.
    """)
    st.write("**Use the sidebar to navigate through the different sections of the app.**")
    
    


def prediction_page():
    st.title("🔍 Online Fraud Payment Detection")
    st.subheader('Predict whether a transaction is fraudulent')

    model_choice = st.selectbox("Choose the model for prediction", ["Decision Tree", "Logistic Regression", "Naive Bayes"])

    st.header("Select Transaction Type")
    type = st.selectbox("Transaction Type", ["Cash Out", "Payment", "Cash In", "Transfer", "Debit"], key='type')
    
    st.header("Input Transaction Details")
    amounts = st.number_input("Transaction Amount", min_value=0, max_value=10000000, value=0, step=1, key='amount')
    oldb_orig = st.number_input("Your Balance Before Transaction", min_value=0, max_value=10000000, value=0, step=1, key='old_balance_orig')
    oldb_dest = st.number_input("Recipient's Balance Before Transaction", min_value=0, max_value=10000000, value=0, step=1, key='old_balance_dest')
    newb_orig = st.number_input("Your Balance After Transaction", min_value=0, max_value=10000000, value=0, step=1, key='new_balance_orig')
    newb_dest = st.number_input("Recipient's Balance After Transaction", min_value=0, max_value=10000000, value=0, step=1, key='new_balance_dest')

    type_dict = {"Cash Out": 0, "Payment": 1, "Cash In": 2, "Transfer": 3, "Debit": 4}
    type_numeric = type_dict[type]

    input_data = pd.DataFrame({
        'type': [type_numeric],
        'amount': [amounts],
        'oldbalanceOrg': [oldb_orig],
        'newbalanceOrig': [newb_orig],
        'oldbalanceDest': [oldb_dest],
        'newbalanceDest': [newb_dest]
    })

    if st.button('Predict'):
        try:
            if model_choice == "Decision Tree":
                model = decision_tree_model
            elif model_choice == "Logistic Regression":
                model = logistic_regression_model
            elif model_choice == "Naive Bayes":
                model = naive_bayes_model

            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)[:, 1]

            st.write("### Prediction Results")
            st.write(f"**Prediction:** {'🟢 Not Fraud' if prediction[0] == 0 else '🔴 Fraud'}")
            st.write(f"**Prediction Probability:** {prediction_proba[0]:.2f}")

            st.progress(prediction_proba[0])

            fraud_image_url = "pngtree-fraud-alert-red-rubber-stamp-on-white-insecure-safety-id-vector-png-image_21878902.png"
            not_fraud_image_url = "images.jpeg"
            st.image(fraud_image_url if prediction[0] == 1 else not_fraud_image_url, width=100)

        except Exception as e:
            st.error(f"An error occurred: {e}")

def eda_page():
    st.title("📊 Exploratory Data Analysis")
    st.subheader('Exploring the transaction dataset')
    
    data = pd.read_csv('Sample-Dataset.csv)

    st.write("### Dataset Overview")
    st.write(data.head())

    st.write("### Basic Statistics")
    st.write(data.describe())

    st.write("### Distribution of Transaction Types")
    fig, ax = plt.subplots()
    sns.countplot(data['type'], ax=ax)
    st.pyplot(fig)

    non_numeric_columns = ['nameOrig', 'nameDest']
    data_numeric = data.drop(columns=non_numeric_columns)
    data_encoded = pd.get_dummies(data_numeric, columns=['type'])

    st.write("### Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(data_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("### Distribution of Transaction Amounts")
    fig, ax = plt.subplots()
    sns.histplot(data['amount'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

def about_page():
    st.title("About - Online Fraud Detection from kelompok 2")
    st.write("""
    This application helps to predict whether a financial transaction is fraudulent or not. It uses three different machine learning models:
    - **Decision Tree**: Known for its interpretability and ability to handle non-linear relationships.
    - **Logistic Regression**: A simple and effective baseline model providing probability estimates.
    - **Naive Bayes**: Computationally efficient and handles high-dimensional data well.
    - **Our Team**: 
    > Jimmie Henderson Gunawan (2602164685)
    >
    > Mellisa angeline (2602077862)
    >
    > tiara intan kusuma (2602172220)
    >
    > angel eodia (2602192140)
    >
    > benny strata wijaya (2540128682)

    """)

# Sidebar navigation with cards
st.sidebar.title("Navigation")
apply_custom_css()
page = st.sidebar.radio("Go to", ["Home", "Prediction", "EDA", "About"], format_func=lambda x: x)

# Display the selected page
if page == "Home":
    home_page()
elif page == "Prediction":
    prediction_page()
elif page == "EDA":
    eda_page()
elif page == "About":
    about_page()

# Add a watermark
st.markdown("""
<div class="watermark">
    Developed with ❤️ by Kelompok 2
</div>
""", unsafe_allow_html=True)

