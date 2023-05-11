#!/usr/bin/env python
# coding: utf-8

# In[28]:


import os
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import LinearRegression
import pickle
import gensim
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from xgboost import XGBClassifier
import io

column_list = joblib.load('column_list.joblib')
lr_1 = joblib.load('APIlr_1.joblib')
lr_2 = joblib.load('APIlr_2.joblib')
xgb = joblib.load('APIxgb.joblib')
lgb = joblib.load('APIlgb.joblib')
scaler = joblib.load('scaler.joblib')
lda_model_emp = joblib.load('lda_model_emp.joblib')
lda_model_title = joblib.load('lda_model_title.joblib')
# Define the input schema
class LoanData(BaseModel):
    x: float
    # Add other columns as needed

class LoanDataList(BaseModel):
    loans: List[LoanData]

def emp_title_lda_transform(dataframe):
    samp_frame = dataframe.copy()
#    nltk.download('wordnet')
    samp_frame['emp_title'] = samp_frame['emp_title'].fillna('nan')

    docs = samp_frame['emp_title'].tolist()

# Tokenize the sentences
    tokenized_docs = [[word for word in document.lower().split() if word.isalpha()] for document in docs]

    # Remove stop words
    stop_words = set(gensim.parsing.preprocessing.STOPWORDS)
    tokenized_docs = [[word for word in document if word not in stop_words] for document in tokenized_docs]

    # Stem or lemmatize the words
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    tokenized_docs = [[lemmatizer.lemmatize(word) for word in document] for document in tokenized_docs]

    # Create a gensim Dictionary object
    dictionary = corpora.Dictionary(tokenized_docs)

    # Convert tokenized documents into a bag of words representation
    corpus = [dictionary.doc2bow(document) for document in tokenized_docs]

    lda_model_emp = joblib.load('lda_model_emp.joblib')

    # Transform the corpus into a matrix of topic probabilities
    topic_probs = np.zeros((len(corpus), 10))

    for i, doc in enumerate(corpus):
        for topic, prob in lda_model_emp[doc]:
            topic_probs[i][topic] = prob
            
    samp_frame['emp_title'] = samp_frame['emp_title'].fillna('nan')

# Tokenize the sentences
    tokenized_docs = [[word for word in document.lower().split() if word.isalpha()] for document in samp_frame['emp_title'].tolist()]

# Remove stop words
    stop_words = set(gensim.parsing.preprocessing.STOPWORDS)
    tokenized_docs = [[word for word in document if word not in stop_words] for document in tokenized_docs]

# Stem or lemmatize the words
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    tokenized_docs = [[lemmatizer.lemmatize(word) for word in document] for document in tokenized_docs]

    # Convert tokenized documents into a bag of words representation
    corpus = [dictionary.doc2bow(document) for document in tokenized_docs]

    # Transform the corpus into a matrix of topic probabilities
    topic_probs = np.zeros((len(corpus), 10))

    for i, doc in enumerate(corpus):
        for topic, prob in lda_model_emp[doc]:
            topic_probs[i][topic] = prob

    # Add topic probabilities as new columns to the DataFrame
    for i in range(10):
        samp_frame[f"Topic {i+1}"] = topic_probs[:, i]
        
    return samp_frame

def title_lda_transform(dataframe):
    samp_frame = dataframe.copy()
# nltk.download('wordnet')
    samp_frame['title'] = samp_frame['title'].fillna('nan')
    docs = samp_frame['title'].tolist()

# Tokenize the sentences
    tokenized_docs = [[word for word in document.lower().split() if word.isalpha()] for document in docs]

# Remove stop words
    stop_words = set(gensim.parsing.preprocessing.STOPWORDS)
    tokenized_docs = [[word for word in document if word not in stop_words] for document in tokenized_docs]

# Stem or lemmatize the words
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    tokenized_docs = [[lemmatizer.lemmatize(word) for word in document] for document in tokenized_docs]

# Create a gensim Dictionary object
    dictionary = corpora.Dictionary(tokenized_docs)

# Convert tokenized documents into a bag of words representation
    corpus = [dictionary.doc2bow(document) for document in tokenized_docs]

    lda_model_title = joblib.load('lda_model_title.joblib')

# Transform the corpus into a matrix of topic probabilities
    topic_probs = np.zeros((len(corpus), 10))

    for i, doc in enumerate(corpus):
        for topic, prob in lda_model_title[doc]:
            topic_probs[i][topic] = prob
        
    samp_frame['title'] = samp_frame['title'].fillna('nan')

# Tokenize the sentences
    tokenized_docs = [[word for word in document.lower().split() if word.isalpha()] for document in samp_frame['title'].tolist()]

# Remove stop words
    stop_words = set(gensim.parsing.preprocessing.STOPWORDS)
    tokenized_docs = [[word for word in document if word not in stop_words] for document in tokenized_docs]

# Stem or lemmatize the words
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    tokenized_docs = [[lemmatizer.lemmatize(word) for word in document] for document in tokenized_docs]

# Convert tokenized documents into a bag of words representation
    corpus = [dictionary.doc2bow(document) for document in tokenized_docs]

# Transform the corpus into a matrix of topic probabilities
    topic_probs = np.zeros((len(corpus), 10))

    for i, doc in enumerate(corpus):
        for topic, prob in lda_model_title[doc]:
            topic_probs[i][topic] = prob

# Add topic probabilities as new columns to the DataFrame
    for i in range(10):
        samp_frame[f"title_Topic {i+1}"] = topic_probs[:, i]
        
    return samp_frame

def data_process(dataframe):
    samp_frame = dataframe.copy()
    
    samp_frame["home_ownership"].replace({"ANY":"OTHER",
                             "NONE":"OTHER"},
                            inplace=True)
    
    samp_frame['emp_length'] = samp_frame['emp_length'].fillna('none')
    
    samp_frame['zip_code']= [address[-5:] for address in samp_frame['address']]

    samp_frame['term'] = samp_frame.term.map({' 36 months': 36, ' 60 months': 60})

    samp_frame['initial_list_status'] = samp_frame.initial_list_status.map({'w': 0, 'f': 1})

    samp_frame = samp_frame.drop(['emp_title','title','address','loan_status'], axis = 1)
    
    samp_frame['earliest_cr_line'] = pd.to_datetime(samp_frame['earliest_cr_line'])
    
    samp_frame['earliest_cr_line']  = (samp_frame['earliest_cr_line'] .dt.year - 1900) * 12 + samp_frame['earliest_cr_line'] .dt.month
    
    to_be_dummied = list(samp_frame.select_dtypes(include='object').columns)
    
    samp_frame = pd.get_dummies(samp_frame,columns=to_be_dummied,drop_first=True)
    
    column_list = joblib.load('column_list.joblib')
    
    samp_frame = samp_frame.reindex(columns = column_list, fill_value = 0)
    
    samp_frame = samp_frame.sort_index(axis=1)
    
    
    samp_frame = scaler.transform(samp_frame)
    
    return samp_frame

def predict_default(input_df: pd.DataFrame):
    xx = emp_title_lda_transform(input_df)
    xxx = title_lda_transform(xx)
    xxx=xxx.dropna()
    loans = xxx['loan_id']
    xxxx = data_process(xxx)

    # Make predictions
    lr1p = lr_1.predict_proba(xxxx)[:, 1]
    lr2p = lr_2.predict_proba(xxxx)[:, 1]
    xgbp = xgb.predict_proba(xxxx)[:, 1]
    lgbp = lgb.predict_proba(xxxx)[:, 1]
    proba_list = [lr1p, lr2p, xgbp, lgbp]

    # Stack the arrays along the columns axis
    proba_stacked = np.column_stack(proba_list)

    # Compute the mean of the columns along the rows axis
    proba_mean = np.mean(proba_stacked, axis=1)

    default_frame = pd.DataFrame({'loan_id': loans, 'default_prob': proba_mean}).sort_values('default_prob')
    default_frame['answer'] = 'loan' + default_frame['loan_id'].astype(str) + '  ' + default_frame['default_prob'].astype(str)

    return default_frame['answer'].tolist()


# FastAPI .
api_title = "LoanTap Default Prediction API"
api_description = """
This API allows you to predict the default rate of a loan based on the borrower's credit info and loan's info.
Parameters include:

'loan_amnt'：any float value

'term': ' 36 months' or ' 60 months'

'emp_title': any string

'emp_length': '10+ years', '4 years', '< 1 year', '6 years', '9 years',
       '2 years', '3 years', '8 years', '7 years', '5 years', '1 year',
       'nan'
       
'home_ownership':'RENT', 'MORTGAGE', 'OWN', 'OTHER', 'NONE', 'ANY'

'annual_inc': any float value

'verification_status': 'Not Verified', 'Source Verified', 'Verified'

'purpose': 'vacation', 'debt_consolidation', 'credit_card',
       'home_improvement', 'small_business', 'major_purchase', 'other',
       'medical', 'wedding', 'car', 'moving', 'house', 'educational',
       'renewable_energy'
       , 
'title': any text value

'dti': any float value

'earliest_cr_line'：month - year, such as'Jun-1990'. Month must be one of the following: 'Jun', 'Jul', 'Aug', 'Sep', 'Mar', 'Jan', 'Dec', 'May', 'Apr',
       'Oct', 'Feb', 'Nov'
       
'open_acc': any integer value

'pub_rec':any integer value

'revol_bal': revolving balance, any float value

'revol_util': revolving credit, any float value

'total_acc': any integer value

'initial_list_status': 'w', 'f'

'application_type':'INDIVIDUAL', 'JOINT', 'DIRECT_PAY'

'mort_acc': any integer value

'pub_rec_bankruptcies': any integer value

'address': the last 5 digits must be the zipcode.

"""
loan_default = FastAPI(title = api_title , description = api_description)

@loan_default.post("/predict")
async def predict_default_endpoint(file: UploadFile = File(...)):
    # Read the uploaded file content into a DataFrame
    file_content = await file.read()
    input_df = pd.read_csv(io.BytesIO(file_content))

    predictions = predict_default(input_df)
    return {"predictions": predictions}

