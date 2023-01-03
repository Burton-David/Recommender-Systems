import pandas as pd

def get_recommended_items(df, user_id, context, N=10):
    """
    Returns the top N recommended items for a given user and context.
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing ratings data with columns for 'user_id', 'item_id', 'rating', and 'context'.
    user_id (int): The ID of the user.
    context (str): The context in which the recommendations will be used.
    N (int): The number of recommended items to return (default is 10).
    
    Returns:
    pandas.Series: A series of the recommended item IDs.
    """
    # Filter the dataframe to only include items that have a high average rating in the given context
    context_items = df[df['context'] == context].groupby('item_id').rating.mean().sort_values(ascending=False).index
    
    # Filter the dataframe to only include items that the user has rated highly
    user_items = df[df['user_id'] == user_id].sort_values('rating', ascending=False)['item_id']
    
    # Combine the lists of items and sort them by rating
    combined_items = pd.concat([context_items, user_items]).sort_values(ascending=False)
    
    # Remove duplicates and return the top N items
    return combined_items.drop_duplicates().iloc[:N]

def main():
    # Import the data
    df = pd.read_csv('ratings.csv')

    # Get the recommendations for the user and context
    recommendations = get_recommended_items(df, 1, 'outdoor')

    # Print the recommendations
    print(recommendations)

if __name__ == '__main__':
    main()
