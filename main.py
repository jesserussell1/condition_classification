import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from bs4 import BeautifulSoup
import re

st.title("Natural Language Processing Medical Conditions App")
st.header("Random Forest Model based on text for Medical Conditions")
st.subheader("Jesse Russell - June 2023")
st.text("Data Source: https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29")

#Load the model
@st.cache_resource
def load_model():
	  return joblib.load("my_random_forest.joblib")

loaded_rf = load_model()

@st.cache_data
def load_data(sheets_url):
    #csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    return pd.read_csv(sheets_url)

df = load_data(st.secrets["public_gsheets_url"])

# df_train = df_train[df_train['condition'].map(df_train['condition'].value_counts()) > 32]

# Filter for some conditions
df_train = df[df['condition'].isin(['Birth Control', 'Depression', 'Pain', 'Anxiety', 'Bipolar Disorder',
                                   'Insomnia', 'Weight Loss', 'Obesity', 'ADHD', 'Diabetes, Type 2',
                                   'Emergency Contraception', 'High Blood Pressure ', 'Vaginal Yeast Infection',
                                   'Abnormal Uterine Bleeding', 'Bowel Preparation', 'ibromyalgia',
                                   'Smoking Cessation', 'Migraine' 'Anxiety and Stress', 'Constipation',
                                    'Major Depressive Disorde', 'Chronic Pain', 'Panic Disorde', 'Migraine Prevention',
                                    'Urinary Tract Infection', 'Opiate Dependence', 'Osteoarthritis', 'Muscle Spasm',
                                    'Erectile Dysfunction', 'Generalized Anxiety Disorde', 'Allergic Rhinitis',
                                    'Irritable Bowel Syndrome', 'Rheumatoid Arthritis ', 'Bacterial Infection',
                                    'Cough', 'Sinusitis', 'Nausea/Vomiting', 'GERD', 'Hepatitis C', 'Hyperhidrosis',
                                    'Restless Legs Syndrome', 'Overactive Bladde', 'Multiple Sclerosis',
                                    'High Cholesterol', 'Psoriasis', 'Schizophrenia',
                                    'Hypogonadism, Male', 'Back Pain', 'Asthma, Maintenance', 'Bronchitis',
                                    'Not Listed / Othe', 'Headache', 'Underactive Thyroid',
                                    'Influenza', 'Cough and Nasal Congestion',
                                    'Opiate Withdrawal', 'Nasal Congestion', 'Pneumonia', 'Osteoporosis', 'Asthma',
                                    'Ulcerative Colitis', 'Diarrhea', 'Strep Throat', 'Allergies', 'COPD',
                                    'Diabetes, Type 1 ',  'Cold Symptoms', 'Asthma, acute', 'Atrial Fibrillation'])]

# Drop columns we won't be using for the model (including the dependent variable, drug)
X = df_train.drop(['drugName','rating','date','usefulCount'],axis=1)

# Remove double quotes
for i, col in enumerate(X.columns):
    X.iloc[:, i] = X.iloc[:, i].str.replace('"', '')

# Get list of stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

# Get the nltk wordnet
nltk.download('wordnet')

# Lemmatize
lemmatizer = WordNetLemmatizer()

# Function to clean text
def review_to_words(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))

# Apply the cleaning function to the text
X['review_clean'] = X['review'].apply(review_to_words)

# Set feature and dependent variable
X_feat = X['review_clean']
y = X['condition']

# Do train test split
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, stratify=y, test_size=0.2, random_state=0)

# Bag of words
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
#count_test = count_vectorizer.transform(X_test)

# Web page contents
st.subheader("Enter a couple of sentences about medical experiences and medications.")
st.caption("This might take a minute -- the model vectorizes a lot of data. If you don't get the "
           "expected result, try adding details.")

left_column, right_column = st.columns(2)

with left_column:
    inp_description = st.text_input('Description of your condition')

if st.button('Make Guess'):
    inp_description_raw = review_to_words(inp_description)

    # Get a prediction from some text
    input_array = count_vectorizer.transform([inp_description_raw]).toarray()
    prediction = loaded_rf.predict(input_array)

    st.write("Your guess is: " + str(prediction))

