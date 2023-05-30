import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from bs4 import BeautifulSoup
import re


# To show all the rows of pandas dataframe
pd.set_option('display.max_rows', None)

# Load the dataset from google sheets
df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQxg-eU4BXWaTJzHKlNW8pS3Jda-4v75jMrSWt0bOYJZga2Tsu8WJ8od5lvRrgTTy89qEiMNPDnid71/pub?gid=2072216167&single=true&output=csv")

# Look at the dataframe
df.head()

# Look at how many different conditions there are in the data
df.condition.value_counts().head(n = 200)

# df_train = df_train.loc[~df_train['condition'].isin(['Acne'])]

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
print(stopwords.words('english'))
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
count_test = count_vectorizer.transform(X_test)

# Fit a RF model
rf_model = RandomForestClassifier()
rf_model.fit(count_train, y_train)

# Save the model
joblib.dump(rf_model, "my_random_forest.joblib", compress=3)
