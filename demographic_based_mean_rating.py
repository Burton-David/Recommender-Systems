import pandas as pd

def get_recommended_items(df, user_id, N=10):
    """
    Returns the top N recommended items for a given user based on their age and gender.
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing ratings data with columns for 'user_id', 'item_id', 'rating', 'age', and 'gender'.
    user_id (int): The ID of the user.
    N (int): The number of recommended items to return (default is 10).
    
    Returns:
    pandas.Series: A series of the recommended item IDs.
    """
    # Get the user's age and gender
    user_age = df[df['user_id'] == user_id]['age'].iloc[0]
    user_gender = df[df['user_id'] == user_id]['gender'].iloc[0]
    
    # Filter the dataframe to only include items that have a high average rating from users of the same age and gender
    recommended_items = df[(df['age'] == user_age) & (df['gender'] == user_gender)].groupby('item_id').rating.mean().sort_values(ascending=False)
    
    return recommended_items.index[:N]

def main():
    # Import the data
    df = pd.read_csv('ratings.csv')

    # Get the recommendations for the user
    recommendations = get_recommended_items(df, 1)

    # Print the recommendations
    print(recommendations)

if __name__ == '__main__':
    main()
