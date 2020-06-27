import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    in:
        messages_filepath: messages file path
        categories_filepath: categories file path
    out:
        df_merged: merged dataframe
    '''

    # read messages file into a dataframe
    df_messages = pd.read_csv(messages_filepath)

    # read categories file into a dataframe
    df_categories = pd.read_csv(categories_filepath)

    # merge dataframes on id
    df_merged = pd.merge(df_messages, df_categories, on='id')

    return df_merged

def clean_data(df):
    '''
    in:
		df: mergerd dataframe from load_data()
	out:
		df_cleaned: cleaned dataframe
    '''

    df_categories = pd.DataFrame(df.categories.str.split(';', expand=True))
    df_categories.columns = [category.split('-')[0] for category in df_categories.iloc[0]]
    df_categories = df_categories.applymap(lambda x: int(x.split('-')[1]))

    df = df.drop(columns=['categories'])

    df_cleaned = pd.concat([df, df_categories], axis=1)
    df_cleaned.drop_duplicates(inplace=True, ignore_index=True)

    return df_cleaned

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename.split('.')[0], engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
