# File of the class Utils, containing various helper functions
# File: utils.py
# Author: Atharva Kulkarni


import pandas as pd
import numpy as np
import re
import spacy
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn import naive_bayes
from sklearn.svm import LinearSVC
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')








class Utils():


    # -------------------------------------------- constructor --------------------------------------------
    
    def __init__(self):
        """ Class Constructor """
        self.stop_words = stopwords.words('english')
        unwanted_stopwords = {'no', 'nor', 'not', 'ain', 'aren', "aren't", 'couldn', 'what', 'which', 'who',
                                      'whom',
                                      'why', 'how', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                                      'hasn',
                                      "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                                      "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                                      "wasn't",
                                      'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'don', "don't"}

        self.stop_words = [ele for ele in self.stop_words if ele not in unwanted_stopwords]
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.nouns = ['NNP', 'NNPS']
        self.nlp = spacy.load('en_core_web_sm')
        self.label_encoder = LabelEncoder()


    # -------------------------------------------- Function to read data --------------------------------------------
    
    def read_data(self, path, sep=",", usecols=[]):
        """ Function to read the data
        @param path (str): path to the dataset
        @param df (pd.DataFrame): pandas DataFrame.
        """
        return pd.read_csv(path, 
                           sep=sep,  
                           usecols=usecols)
        



    # -------------------------------------------- Function to decide main categories --------------------------------------------
    
    def generate_label(self, df, top=5):
        """ Function to generate labels
        @param df (pd.DataFrame): input data.
        @param top (int): The number of labels to have.
        """
        df['label'] = df.product_category_tree.apply(lambda x: x.split('>>')[0][2:].strip())
        top_categories = list(df.groupby('label').count().sort_values(by='uniq_id',ascending=False).head(top).index)
        df = df[df['label'].isin(top_categories)]
        return df
        
        
        
        
        
       
    # -------------------------------------------- Function to clean text --------------------------------------------
    
    def clean_text(self, text, remove_stopwords=True, lemmatize=True):
        """ Function to clean text
        @param text (str): text to be cleaned
        @param remove_stopwords (bool): To remove stopwords or not.
        @param lemmatize (bool): to lemmatize or not.
        """

        # Remove emails 
        text = re.sub('\S*@\S*\s?', '', text)
        
        # Remove new line characters 
        text = re.sub('\s+', ' ', text) 
        
        # Remove distracting single quotes 
        text = re.sub("\'", '', text)

        # Remove puntuations and numbers
        text = re.sub('[^a-zA-Z]', ' ', text)

        # Remove single characters
        text = re.sub('\s+[a-zA-Z]\s+^I', ' ', text)
        
        # Remove accented words
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        # remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\s*|\s\s*', ' ', text).strip()
        text = text.lower()

        if not remove_stopwords and not lemmatize:
            return text

        # Remove unncecessay stopwords
        if remove_stopwords:
            text = word_tokenize(text)
            text = " ".join([word for word in text if word not in self.stop_words])
        
        # Word lemmatization
        if lemmatize:
            text = self.nlp(text)
            lemmatized_text = []
            for word in text:
                if word.lemma_.isalpha():
                    if word.lemma_ != '-PRON-':
                        lemmatized_text.append(word.lemma_.lower())
            text = " ".join([word.lower() for word in lemmatized_text])
                
        return text
        
        
        
        
        
    # -------------------------------------------- Function to prepare data for the model --------------------------------------------    
    
    def prepare_data(self, df, test_size=0.2):
        """ Function to prepare data for model.
        @param df (pd.DataFrame): input data.
        @param test_size (float): train-test split ratio.
        """
        df = df.sample(frac=1).reset_index(drop=True)
        description = df['description'].apply(lambda x: self.clean_text(str(x), remove_stopwords=False, lemmatize=False))
        labels = df['label'].values.tolist()
        labels = self.label_encoder.fit_transform(labels)
        x_train, x_test, y_train, y_test = train_test_split(description, 
                                                            labels, 
                                                            test_size=test_size, 
                                                            stratify=labels)
        return x_train, x_test, y_train, y_test
        
        
        
        
                
    



    # ------------------------------------- GENERATE COUNT FEATURES AS VECTORS---------------------------------

    def count_vectorize(self, X_train, X_test, analyzer='word', token_pattern=r'\w{1,}', max_features=10000, ngram_range=(1,1)):
        """ Function to count vectorize the text data.
        @param X_train (list): list of input train text data.
        @param X_val (list): list of input test text data.
        @param analyzer (string): Whether the feature should be made of word n-gram or character n-grams.(‘word’, ‘char’, ‘char_wb')
        @param token_pattern (string): Regular expression denoting what constitutes a “token”.
        @param max_features (int): Max no. of words to build your vocab.
        @param ngram_range (tuple): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.
        @return xtrain_count(list): count vectorized train data.
        @return xvalid_count(list): count vectorized test data.
        """
        if analyzer == 'word':
            count_vect = CountVectorizer(analyzer=analyzer, 
                                         token_pattern=token_pattern,
                                         max_features=max_features,
                                         ngram_range=ngram_range)
        else:
            count_vect = CountVectorizer(analyzer=analyzer, 
                                     max_features=max_features,
                                     ngram_range=ngram_range)
        count_vect.fit(X_train)

        # transform the training and validation data using count vectorizer object
        xtrain_count = count_vect.transform(X_train)
        xvalid_count = count_vect.transform(X_test)

        return xtrain_count, xvalid_count

    
    

    # ---------------------- GENERATE WORD LEVEL TF-IDF FEATURES AS VECTORS---------------------------------

    def tf_idf_vectorize(self, X_train, X_test, analyzer='word', token_pattern=r'\w{1,}', max_features=10000, ngram_range=(1,1)):
        """ Function to tf-idf vectorize the text data.
        @param X_train (list): list of input train text data.
        @param X_val (list): list of input test text data.
        @param analyzer (string): Whether the feature should be made of word n-gram or character n-grams.(‘word’, ‘char’, ‘char_wb')
        @param token_pattern (string): Regular expression denoting what constitutes a “token”.
        @param max_features (int): Max no. of words to build your vocab.
        @param ngram_range (tuple): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.
        @return xtrain_tfidf(list): tf-idf vectorized train data.
        @return xvalid_tfidf(list): tf-idf vectorized test data.
        """
        if analyzer == 'word':
            tfidf_vect = TfidfVectorizer(analyzer=analyzer, 
                                         token_pattern=token_pattern,
                                         max_features=max_features,
                                         ngram_range=ngram_range)
        else:
            tfidf_vect = TfidfVectorizer(analyzer=analyzer, 
                                         max_features=max_features,
                                         ngram_range=ngram_range)
        
        tfidf_vect.fit(X_train)

        xtrain_tfidf = tfidf_vect.transform(X_train)
        xvalid_tfidf = tfidf_vect.transform(X_test)

        return xtrain_tfidf, xvalid_tfidf
         
        
     
    
    # ------------------------------------------------------ TRAIN ML MODELS --------------------------------------------
    def train_ml_model(self, classifier, X_train, X_test, y_train, y_test):
        """ Function to train and evaluate the model
        @param classifier (sklearn model): The ML model to be used.
        @param X_train (list): list of input train text data.
        @param X_test (list): list of input test text data.
        @param y_train (list): training data label.
        @param y_test (list): test data label.
        @return accuracy (float): accuracy of the model.
        """
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = metrics.accuracy_score(predictions, y_test)      
        return accuracy
             

