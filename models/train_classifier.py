import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import string
import joblib

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt','wordnet','stopwords'])

def load_data(database_filepath):
    '''
    Loads a sqlite db, reads data, and returns the data listed below.

    in:
        database_filepath: database file path
    out:
        X: message feature dataframe
        Y: categories target dataframe
        category_names: list of category names
    '''

    engine = create_engine('sqlite:///' + database_filepath)

    df = pd.read_sql_table('DisasterResponse', engine)

    categories = df.columns[4:]

    X = df['message'].values
    Y = df[categories].values

    return X, Y, list(categories)

def tokenize(text):
    '''
    Sanitizes, tokenizes, and lemmatizes a string

    input:
        text: message string to tokenize
    output:
        tokens: list of tokens
    '''

    # remove URLs
    text = re.sub(r'https?:\/\/\S*[\s]*', ' ', text, flags=re.MULTILINE)

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords, special characters, and numerics
    tokens = [w.lower() for w in tokens if (not w in stopwords.words('english')) and (not w in string.punctuation) and (not w.isnumeric())]

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

def build_model():
    '''
    Builds and returns a GridSearchCV model using pipeline

    in:
        n/a
    out:
        gridSearchCv: GridSearchCV model
    '''

    #create a pipeline
    pipeline = Pipeline([
        ('count', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(SGDClassifier(n_jobs=4, verbose=10, early_stopping=True, warm_start=True))))
        ])

    parameters = {
    'count__ngram_range':((1,1),(1,2),(2,2)),
    'tfidf__smooth_idf':[True, False],
    'clf__estimator__estimator__loss':['hinge','log']
    }

    return GridSearchCV(pipeline, param_grid=parameters)

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints out an evaluation using the inputted model and test splits

    in:
        model: Model from build_model()
		X_test: X test split
		Y_test: Y test split
		category_names: list of categories from load_data()
    output:
        n/a
    '''
    y_pred = model.predict(X_test)

    for id in range(Y_test.shape[1]):
        print('\n\n')
        print('Category: ' + category_names[id])
        print(classification_report(Y_test[:,id], y_pred[:,id], output_dict=True))

def save_model(model, model_filepath):
    '''
    Saves inputted model to a file path

    in:
        model: Model from build_model()
		model_filepath: file path to save the model to
    out:
        n/a
    '''
    joblib.dump(model, model_filepath)

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
