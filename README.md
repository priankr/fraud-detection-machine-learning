# fraud-detection-machine-learning

The data was downloaded from Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud

Note: The original csv file is too large for github but it can be downloaded directly through the kaggle link above.

The dataset contains transactions made by European credit cardholders credit cards in September 2013. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numeric input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, the original features and more background information about the data are unavailable. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependent cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.

## Goal

The objective is to classify the data transactions as either legitimate (class 0) or fradulent (class 1) using a machine learning model. Four machine learning models will be trained and tested to determine which will yield the best results:

- Decision Trees
- Random Forest
- K Nearest Neighbours
- K Means Clustering

#### See this machine learning project on Kaggle:
https://www.kaggle.com/priankravichandar/credit-card-fraud-detection-machine-learning
