import pandas as pd

def get_recommended_items(df, user, N=10):
    """
    Returns the top N recommended items for a given user based on the user's age and gender.
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing user metadata with columns for 'age' and 'gender', and a list of 'favorite_items' for each user.
    user (int): The user's index in the dataframe.
    N (int): The number of recommended items to return (default is 10).
    
    Returns:
    pandas.Series: A series of the recommended item IDs.
    """
    # Get the user's age and gender
    age = df['age'].loc[user]
    gender = df['gender'].loc[user]
    
    # Get the items recommended for users of the same age and gender
    recommended_items = df[(df['age'] == age) & (df['gender'] == gender)]['favorite_items']
    
    # Flatten the list of lists and get the top N items
    items = [item for sublist in recommended_items for item in sublist]
    top_items = pd.Series(items).value_counts().head(N).index
    
    return top_items

def main():
    # Import the data
    df = pd.read_csv('user-metadata.csv')

    # Get the recommendations for the user
    recommendations = get_recommended_items(df, 0)

    # Print the recommendations
    print(recommendations)

if __name__ == '__main__':
    main()
