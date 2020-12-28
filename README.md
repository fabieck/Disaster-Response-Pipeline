# Disaster Response Pipeline Project

You will find the [GitHub repo](https://github.com/fabieck/Disaster-Response-Pipeline.git) here.

1. [Project Overview](#overview)
2. [Requirements](#requirements)
3. [Files](#files)
4. [Instructions](#instructions)
5. [Acknowledgements](#acknowledgements)

<a id='overview'></a>

## 1. Project Overview

In this project, which is part of Udacity Data Science Nanodegree Program, a model for classification of different messages disaster is built. The dataset is provided by Figure Eight and contains pre-labeled real messages which were sent during real disaster events. It also contains 36 categories like Medical Help, Rescue and Aid related. As a final result we will build a Natural Language Processing model to categorize the messages that can be used in a web app. To achieve this we will build an ETL and a Machine Learning Pipline.  

<a id='requirements'></a>

## 2. Requirements

- flask
- plotly
- nltk
- numpy
- pandas
- scikit-learn
- sqlalchemy

<a id='files'></a>

## 3. Files
Basic Files:
- ETL Pipeline Preparation.ipynb: Basic for ETL Pipline for data/process_data.py
- ML Pipeline Preparation.ipynb: Basic for Machine Learning Pipeline for model/train_classifier.py 

Data:
- data/process_data.py: 
  - A data cleaning pipeline
  - Loads messages and categories datasets
  - Merges messages and categories datasets
  - Stores it in a SQLite database
- disaster_categories.csv: dataset contains categories
- disaster_messages.csv: dataset contains messages  
- DisasterResponse.db: SQLite database which conatins dataset with processes messages and categories

Model:
- model/train_classifier.py: machine learning pipeline, Loads data from the SQLite database, 
  - Splitting of dataset into training and test sets
  - Builds machine learning pipeline
  - Trains and fit a model using GridSearchCV
  - Get accurancy, f1-score and recall of trainset
  - Create pickle file including model
- classifier.pkl: saved model in pkl format
  
App:
- app/run.py: Integrate model and start web application
- templates: This contains html for web app

<a id='instructions'></a>
  
## 4. Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


<a id='acknowledgements'></a>

## 5. Acknowledgements

Thanks to [Figure Eight](https://www.figure-eight.com/) for providing the dataset.
