import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function:
    load messeage and categories data from path and merge them
    
    Args:
    messages_filepath (str): file path of messages csv file
    categories_filepath (str): file path of categories csv file
    
    Return:
    df (DataFrame): A merged dataframe of messages and categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id', how='outer')
    
    return df


def clean_data(df):
    """
    Function:
    Clean Dataframe

    Args:
    df (Dataframe): dataframe that will be cleaned

    Return:
    df (DataFrame): cleaned dataframe of messages and categories
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.map(lambda x: x[:-2]).tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0)
    for column in categories:
        categories[column] = categories[column].str[-1] # set each value to be the last character of the string
        categories[column] = categories[column].astype(int) # convert column from string to numeric

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Function:
    Save the clean dataset into an sqlite database.
    
    Args:
    df (Dataframe): dataframe that will be saved
    database_filename (str): The file name of the database
    """
    
    # Save the clean dataset into an sqlite database.
    engine = create_engine('sqlite:///{}'.format(database_filename)) # e.g. 'sqlite:///InsertDatabaseName.db'
    df.to_sql('Disaster_Messages_Table', engine, index=False)

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