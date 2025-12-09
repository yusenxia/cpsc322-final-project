# CPSC 322 Final Project

## 1. Project Overview
This project analyzes the **TMDB 5000 Movies dataset** to predict the **main genre** of each movie using only numeric features (budget, popularity, vote average, vote count, runtime, release year)

The classifiers implemented include:

- **Decision Tree** 
- **Random Forest** 
- **K-NN**
- **Naive Bayes**

## 2. How to Run the Project

### **Requirements**
Everything is run in Notebook

Make usre you have Python 3.8+ and the following packages:
- pandas
- numpy
- tabulate
- scikit-learn

## 3. How the Project Is Organized

We have dataset in folder: 

- input_data
- output_data

We have our code for DecisionTree and RandomForest in folder **Model**:

- MyDecisionTree.py
- MyRandomForestClassifier.py
- myutils.py

We have folder **mysklearn** to clean the dataset

All the notebooks are in one floder call **Notebooks**

Finally we have unit tests in folder **tests**


