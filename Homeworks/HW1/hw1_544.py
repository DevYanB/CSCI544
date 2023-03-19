#!/usr/bin/env python
# coding: utf-8

# # DEVYAN BISWAS REPORT
# ---
# Some notes:
# - Python version is 3.7.5

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


# get_ipython().system(" pip install bs4 # in case you don't have it installed")
# get_ipython().system(' pip install contractions')

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz


# In[3]:


import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import contractions
 


# In[4]:


import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# In[5]:


nltk.download('wordnet')


# ## Read Data

# In[6]:


df = pd.read_csv('./amazon_reviews_us_Jewelry_v1_00.tsv', sep='\t', usecols = ['star_rating','review_body'], header=0) 


# In[7]:


df


# In[8]:


df.dtypes


# ## Keep Reviews and Ratings
# - Done already in read

# # Data Cleaning
# ---
# - NOTE: Regex expressions sourced from various online resoureces and documentations
# 
# 

# In[9]:


df = df.loc[df['star_rating'].isin([5, 4, 3, 2, 1])]
df


# In[10]:


# First, make sure all the datatypes are correct/consistent

# Convert ratings to int instead of double
df['star_rating'] = df['star_rating'].astype(int)

# Convert ratings to int instead of double
df['review_body'] = df['review_body'].astype(str)


# In[11]:


df.dtypes


# In[12]:


# Getting the average character length of review body BEFORE data cleaning
before_dataproc = df['review_body'].str.len().mean()


# In[13]:


# Lowercase the review bodies
df['review_body'] = df['review_body'].str.lower()


# In[14]:


# Remove links, html tags
df['review_body'] = df['review_body'].str.replace(r'<[^<>]*>', '', regex=True)
df['review_body']  = df['review_body'].str.replace(r's*https?://S+(s+|$)', ' ').str.strip()


# In[15]:


# Expand contractions
df['review_body'] = df['review_body'].astype(str)
df['review_body'] = df['review_body'].apply(lambda x: contractions.fix(x))


# In[16]:


# Remove punctuation, non-alpha
# df['review_body'] = df['review_body'].str.replace(r'[^\w\s]+', ' ')
df['review_body'] = df.review_body.str.replace('[^a-zA-Z\s]', ' ')


# In[17]:


# Remove extra spaces
df['review_body'] = df['review_body'].replace(r'\s+', ' ', regex=True)


# In[18]:


# Remove Blank lines after all data cleaning is done
df['review_body'].replace('', np.nan, inplace=True)
df['review_body'].dropna(inplace=True)


# In[19]:


df


# In[20]:


after_dataproc = df['review_body'].str.len().mean()
print("Before data proc: " + str(before_dataproc) + ",", "After data proc: " + str(after_dataproc))


#  ## We select 20000 reviews randomly from each rating class.
# 
# 

# In[21]:


# Figure out the different values for star rating column
cats = df['star_rating'].unique()


# In[22]:


# Get the integer ones from the dataframe
# Not very pythonic but hey she gets the job done lol
star_5_df = df[df['star_rating'] == 5]
star_4_df = df[df['star_rating'] == 4]
star_3_df = df[df['star_rating'] == 3]
star_2_df = df[df['star_rating'] == 2]
star_1_df = df[df['star_rating'] == 1]


# In[23]:


# CHOOSING 20k random entries from each
# Seeding them so that data is more consistent
df_20_5 = star_5_df.sample(n=20000, random_state=100)
df_20_4 = star_4_df.sample(n=20000, random_state=100)
df_20_3 = star_3_df.sample(n=20000, random_state=100)
df_20_2 = star_2_df.sample(n=20000, random_state=100)
df_20_1 = star_1_df.sample(n=20000, random_state=100)


# In[24]:


# Splitting them 16k and 4k to make new datasets for training and testing
training_5 = df_20_5.iloc[:16000,:]
testing_5 = df_20_5.iloc[16000:,:]
training_4 = df_20_4.iloc[:16000,:]
testing_4 = df_20_4.iloc[16000:,:]
training_3 = df_20_3.iloc[:16000,:]
testing_3 = df_20_3.iloc[16000:,:]
training_2 = df_20_2.iloc[:16000,:]
testing_2 = df_20_2.iloc[16000:,:]
training_1 = df_20_1.iloc[:16000,:]
testing_1 = df_20_1.iloc[16000:,:]


# In[25]:


# Merge all the ones above into one dataframe for training
# training_data = [training_5, training_4, training_3, training_2, training_1]
training_data = pd.concat([training_5, training_4])
training_data = pd.concat([training_data, training_3])
training_data = pd.concat([training_data, training_2])
training_data = pd.concat([training_data, training_1])
training_data=training_data.reset_index(drop=True)


# In[26]:


# Merge all the remaining ones above into one dataframe for testing
testing_data = pd.concat([testing_5, testing_4])
testing_data = pd.concat([testing_data, testing_3])
testing_data = pd.concat([testing_data, testing_2])
testing_data = pd.concat([testing_data, testing_1])
testing_data=testing_data.reset_index(drop=True)


# In[27]:


training_data


# In[28]:


testing_data


# # Pre-processing

# In[29]:


# This is a bit convoluted, but I concat the training and testing data
# For the sake of getting a more accurate measure of the average length of 
# the review body.
whole_dataset = pd.concat([training_data, testing_data])


# In[30]:


whole_dataset


# ## remove the stop words 

# In[31]:


# Average character length before pre-processing
before_preproc = whole_dataset['review_body'].str.len().mean()


# In[32]:


nltk.download('stopwords')
from nltk.corpus import stopwords


# In[33]:


stop_words = stopwords.words('english')
whole_dataset['review_body'] = whole_dataset['review_body'].apply(lambda x : ' '.join([word for word in str(x).split() if word not in (stop_words)]))


# In[34]:


whole_dataset


# ## perform lemmatization  

# In[35]:


from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
wnl = WordNetLemmatizer()


# In[36]:


whole_dataset['review_body'] = whole_dataset['review_body'].apply(lambda x: ' '.join(wnl.lemmatize(word, pos="n") for word in x.split()))


# In[37]:


whole_dataset


# In[38]:


# Average character length before pre-processing
after_preproc = whole_dataset['review_body'].str.len().mean()

# NOTE: Since this is being done on a new subset of the previous, the starting avg will be 
# different, but the idea that this demonstrates is still useful
print("Before pre proc: " + str(before_preproc) + ",", "After pre proc: " + str(after_preproc))


# # TF-IDF Feature Extraction

# In[39]:


# get_ipython().system(' pip install sklearn')


# In[40]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer() 


# In[41]:


whole_dataset_review = whole_dataset['review_body']


# In[42]:


x_whole_vectorized = vectorizer.fit_transform(whole_dataset_review)


# In[43]:


# Now, we can finally re-split the data back into training and testing.
# NOTE: I know there's a built in sklearn funciton to do this, but 
# I only learned about it later and I kinda wanna just stick with 
# what works tbh
X_train = x_whole_vectorized[:80000,:]
X_test = x_whole_vectorized[80000:,:]

y_train = training_data['star_rating']
y_test = testing_data['star_rating']


# # Perceptron

# In[44]:


# Import and training
from sklearn.linear_model import Perceptron
perc = Perceptron(max_iter=1)
perc.fit(X_train, y_train)


# In[45]:


# Testing and score calcs
from sklearn.metrics import recall_score, precision_score, f1_score

perc_y_pred = perc.predict(X_test)

print("METRICS FOR PERCEPTRON")
print("======================")

# Per class
recalls = recall_score(y_test, perc_y_pred, average=None)
precisions = precision_score(y_test, perc_y_pred, average=None)
f1s = f1_score(y_test, perc_y_pred, average=None)

for class_entry,value in enumerate(recalls):
    print(("Recall for class %s: " % str(class_entry+1)), value, end =", ")

print()

for class_entry,value in enumerate(precisions):
    print(("Precision for class %s: " % str(class_entry+1)), value, end =", ")

print()

for class_entry,value in enumerate(f1s):
    print(("F1 for class %s: " % str(class_entry+1)), value, end =", ")

print()

# Averages
recall_avg = recall_score(y_test, perc_y_pred, average='macro')
precision_avg = precision_score(y_test, perc_y_pred, average='macro')
f1_avg = f1_score(y_test, perc_y_pred, average='macro')

print("Recall Avg: ", recall_avg)
print("Precision Avg: ", precision_avg)
print("F1 Avg: ", f1_avg)


# # SVM

# In[46]:


from sklearn.svm import LinearSVC
lin_svc = LinearSVC(max_iter=2000)
lin_svc.fit(X_train, y_train)


# In[47]:


svc_y_pred = lin_svc.predict(X_test)

print("METRICS FOR LINEAR SVC (SVM)")
print("======================")

# Per class
recalls = recall_score(y_test, svc_y_pred, average=None)
precisions = precision_score(y_test, svc_y_pred, average=None)
f1s = f1_score(y_test, svc_y_pred, average=None)

for class_entry,value in enumerate(recalls):
    print(("Recall for class %s: " % str(class_entry+1)), value, end =", ")

print()

for class_entry,value in enumerate(precisions):
    print(("Precision for class %s: " % str(class_entry+1)), value, end =", ")

print()

for class_entry,value in enumerate(f1s):
    print(("F1 for class %s: " % str(class_entry+1)), value, end =", ")

print()

# Average
recall_avg = recall_score(y_test, svc_y_pred, average='macro')
precision_avg = precision_score(y_test, svc_y_pred, average='macro')
f1_avg = f1_score(y_test, svc_y_pred, average='macro')

print("Recall Avg: ", recall_avg)
print("Precision Avg: ", precision_avg)
print("F1 Avg: ", f1_avg)


# # Logistic Regression

# In[48]:


from sklearn.linear_model import LogisticRegression
log_regr = LogisticRegression(max_iter=2000)
log_regr.fit(X_train, y_train)


# In[49]:


log_regr_pred = log_regr.predict(X_test)

print("METRICS FOR LOGISTIC REGRESSION")
print("======================")

# Per class
recalls = recall_score(y_test, log_regr_pred, average=None)
precisions = precision_score(y_test, log_regr_pred, average=None)
f1s = f1_score(y_test, log_regr_pred, average=None)

for class_entry,value in enumerate(recalls):
    print(("Recall for class %s: " % str(class_entry+1)), value, end =", ")

print()

for class_entry,value in enumerate(precisions):
    print(("Precision for class %s: " % str(class_entry+1)), value, end =", ")

print()

for class_entry,value in enumerate(f1s):
    print(("F1 for class %s: " % str(class_entry+1)), value,  end =", ")

print()

# Average
recall_avg = recall_score(y_test, log_regr_pred, average='macro')
precision_avg = precision_score(y_test, log_regr_pred, average='macro')
f1_avg = f1_score(y_test, log_regr_pred, average='macro')

print("Recall Avg: ", recall_avg)
print("Precision Avg: ", precision_avg)
print("F1 Avg: ", f1_avg)


# # Naive Bayes

# In[50]:


from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)


# In[51]:


mnb_pred = mnb.predict(X_test)

print("METRICS FOR MULTINOMIAL NAIVE BAYES")
print("======================")

# Per class
recalls = recall_score(y_test, mnb_pred, average=None)
precisions = precision_score(y_test, mnb_pred, average=None)
f1s = f1_score(y_test, mnb_pred, average=None)

for class_entry,value in enumerate(recalls):
    print(("Recall for class %s: " % str(class_entry+1)), value,  end =", ")

print()

for class_entry,value in enumerate(precisions):
    print(("Precision for class %s: " % str(class_entry+1)), value,  end =", ")

print()

for class_entry,value in enumerate(f1s):
    print(("F1 for class %s: " % str(class_entry+1)), value,  end =", ")

print()

# Average
recall_avg = recall_score(y_test, mnb_pred, average='macro')
precision_avg = precision_score(y_test, mnb_pred, average='macro')
f1_avg = f1_score(y_test, mnb_pred, average='macro')

print("Recall Avg: ", recall_avg)
print("Precision Avg: ", precision_avg)
print("F1 Avg: ", f1_avg)

