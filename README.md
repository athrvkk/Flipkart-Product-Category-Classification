# MIDAS-IIITD-Task-2021
This repository contains the code to the tasks for MIDAS Summer Internship 2021.


# Task 3 Description:
* The task is to classify the product category based on the product's description.
* The product's description has to be extracted from the product category tree.
* The dataset at hand is the Flipkart e-commerce sales sample dataset containing about 20k samples.

# Solution:
* The textual description of each product is used to categories the product.
* The text data is preprocessed by removing emails, new line characters, distracting single quotes, digits, puntuations, single characters, accented words, and multiple spaces.
* Various feature engineering techniques (word count, tf-idf, ngrams, and character level input) are used to develop input representations for the ML models.
* The dataset is split using stratified 70:30, train:test ratio.
* The models are trained on the test data and its performance is measured on the validation data.
* Standard machine learning models (Multinomial Naive Bayes, Random Forest, and Linear SVC) are used as they give pretty good accuracy.
* Confusion matrix as well as the performance analysis of each model is provided in the notebook.

# Results (considering the to 10 product categories)

* 16,631 samples out of the total 20k samples cover the top 10 categories which is about 83% of the total data.
* The 70:30 stratified split results in 11,641 train samples and 4,990 test samples.

|Feature|Multinomial Naive Bayes|Random Forest|Linear SVC|
|:-------|:--------|:-------|:-------|
|Word frequency based representation (only unigrams)|97.35%|98.29%|98.93%|
|Word frequency based representation (unigrams and bigrams)|95.53%|98.39%|98.79%|
|Word frequency based representation (unigrams, bigrams, and trigrams)|95.39%|**98.43%**|98.71%|
|Word TF-IDF based representation (only unigrams)|95.13%|97.87%|98.21%|
|Word TF-IDF based representation (unigrams and bigrams)|**97.61%**|97.83%|**99.27%**|
|Word TF-IDF based representation (unigrams, bigrams, and trigrams)|97.17%|97.73%|99.17%|
|Character TF-IDF based representation (bigrams and trigrams)|87.27%|96.89%|98.95%|


# Results (considering the to 25 product categories)

* 19,619 samples out of the total 20k samples cover the top 10 categories which is about 98% of the total data.
* The 70:30 stratified split results in 13,733 train samples and 5,886 test samples.

|Feature|Multinomial Naive Bayes|Random Forest|Linear SVC|
|:-------|:--------|:-------|:-------|
|Word frequency based representation (only unigrams)|**93.84%**|96.55%|97.04%|
|Word frequency based representation (unigrams and bigrams)|93.30%|97.17%|97.19%|
|Word frequency based representation (unigrams, bigrams, and trigrams)|91.76%|**97.26%**|96.97%|
|Word TF-IDF based representation (only unigrams)|88.26%|96.00%|97.63%|
|Word TF-IDF based representation (unigrams and bigrams)|92.18%|96.29%|97.72%|
|Word TF-IDF based representation (unigrams, bigrams, and trigrams)|92.15%|96.29%|**97.77%**|
|Character TF-IDF based representation (bigrams and trigrams)|76.97%|94.75%|97.38%|

# Conclusion:

* The simple Machine learning models give quite a good performance for the task at hand.
* Thus, no deep learning models (DNN, CNN, LSTM) or other complex architectures (BERT, etc) are used.
* It can be infered that Linear SVC is better than Multinomial Naive Bayes and Random Forest in terms of performance as well as memory.
* The performance can be further improved by opting for a multimodal approach, extracting the information from the product's images.
