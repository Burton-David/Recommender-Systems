import pandas as pd

def get_recommended_items(df, context, N=10):
    """
    Returns the top N recommended items for a given context.
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing ratings data with columns for 'user_id', 'item_id', 'rating', and 'context'.
    context (str): The context in which the recommendations will be used.
    N (int): The number of recommended items to return (default is 10).
    
    Returns:
    pandas.Series: A series of the recommended item IDs.
    """
    # Filter the dataframe to only include items that have a high average rating in the given context
    recommended_items = df[df['context'] == context].groupby('item_id').rating.mean().sort_values(ascending=False)
    
    return recommended_items.index[:N]

def main():
    # Import the data
    df = pd.read_csv('ratings.csv')

    # Get the recommendations for the context
    recommendations = get_recommended_items(df, 'outdoor')

    # Print the recommendations
    print(recommendations)

if __name__ == '__main__':
    main()
