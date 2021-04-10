# MIDAS-IIITD-Task-2021
This repository contains the code to the tasks for MIDAS Summer Internship 2021.


# Task 3 Solution:
* The product's description has to be extracted from the product category tree.
* The dataset at hand is the Flipkart e-commerce sales sample dataset containing about 20k samples.
* Various feature engineering techniques (word count, tf-idf, ngrams, and character level input) are used to develop input representations for the ML models.
* Standard machine learning models (Multinomial Naive Bayes, Random Forest, and Linear SVC) are used as they give pretty good accuracy.
* Confusion matrix as well as the performance analysis of each model is provided at the end.

# Results (considering the to 10 product categories)

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

|Feature|Multinomial Naive Bayes|Random Forest|Linear SVC|
|:-------|:--------|:-------|:-------|
|Word frequency based representation (only unigrams)|**93.84%**|96.55%|97.04%|
|Word frequency based representation (unigrams and bigrams)|93.30%|97.17%|97.19%|
|Word frequency based representation (unigrams, bigrams, and trigrams)|91.76%|**97.26%**|96.97%|
|Word TF-IDF based representation (only unigrams)|88.26%|96.00%|97.63%|
|Word TF-IDF based representation (unigrams and bigrams)|92.18%|96.29%|97.72%|
|Word TF-IDF based representation (unigrams, bigrams, and trigrams)|92.15%|96.29%|**97.77%**|
|Character TF-IDF based representation (bigrams and trigrams)|76.97%|94.75%|97.38%|
