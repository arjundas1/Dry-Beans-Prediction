# Dry Beans Prediction
The given dataset has been taken from the UC Irvine Machine Learning Repository.

This is a very clean dataset which has no missing values.
The common Machine Learning Algorithms have been used to predict the variety of Dry Bean based on various parameters.

The following types of Beans are present in the dataset:
- Seker
- Barbunya
- Bombay
- Cali
- Horoz
- Sira
- Dermason

Each Machine Learning Algorithm used has been iterated for a 100 times to get the train-test split which yields the highest prediction percentage. 
This combination has been stored as a pickle file for further use. 
Due to the fact that the algorithms have been iterated this many times, SVM algorithm consumes a lot of time for completing its execution.

K Nearest Neighbours Algorithm yields the highest prediction percentage. 
