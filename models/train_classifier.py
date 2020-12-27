import sys
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import numpy as np
import pickle

def load_data(database_filepath):
    """
       Function:
       load data from database

       Args:
       database_filepath: path of the database
       Return:

       X (DataFrame) : Message dataframe
       Y (DataFrame) : target dataframe
       category_names (list) : Disaster category names
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Disaster_Messages_Table', engine)
    category_names = df.columns[4:]

    X = df['message']
    Y = df[category_names]

    return X, Y, category_names


def tokenize(text):
    """
    Function:
    Tokenize text (a disaster message).

    Args:
        text: (String) A disaster message.

    Returns:
        clean_tokens : (list) It contains tokens.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        # print(clean_token)
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    """
    Function:
    Build model.
    Returns:
        pipline: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
    """
    # creating pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    # choose paramter for gridsearch
    parameters = {
        'clf__estimator__learning_rate': [0.1, 0.3],
        'clf__estimator__n_estimators': [100, 200]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=5, n_jobs=4)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function:
    Evaluate a model and return the classificatio and accurancy score.

    Args:
        Model
        X_test
        y_test
        catgegory_names
    """

    try:
        y_pred = model.predict(X_test)
        for i, column in enumerate(Y_test):
            print(column)
            print(classification_report(Y_test[column], y_pred[:, i]))
        print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))
    except:
        pass




def save_model(model, model_filepath):
    """
    Function:
    Save the model as pickle
    Args:
        model
        model_filepath

    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()