import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_recommended_items(df, user_id, N=10):
    """
    Returns the top N recommended items for a given user based on the cosine similarity of their ratings.
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing ratings data with columns for 'user_id', 'item_id', and 'rating'.
    user_id (int): The ID of the user.
    N (int): The number of recommended items to return (default is 10).
    
    Returns:
    pandas.Series: A series of the recommended item IDs.
    """
    # Get the user's ratings
    user_ratings = df[df['user_id'] == user_id]
    
    # Get the item IDs for the user's ratings
    user_items = user_ratings['item_id'].tolist()
    
    # Get the ratings matrix
    ratings_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    
    # Compute the cosine similarity matrix
    sim_matrix = cosine_similarity(ratings_matrix, ratings_matrix)
    
    # Get the user's row from the similarity matrix
    user_sim = sim_matrix[user_id]
    
    # Get the top N similar users
    top_users = user_sim.argsort()[::-1][:N]
    
    # Get the rating sums for the top users
    top_users_rating_sums = ratings_matrix.iloc[top_users].sum()
    
    # Get the rating counts for the top users
    top_users_rating_counts = ratings_matrix.iloc[top_users].astype(bool).sum()
    
    # Compute the weighted average rating for each item
    top_users_ratings = top_users_rating_sums / top_users_rating_counts
    
    # Sort the ratings in descending order
    top_items = top_users_ratings.sort_values(ascending=False).index
    
    # Filter out the items the user has already rated
    recommended_items = top_items[~top_items.isin(user_items)]
    
    return recommended_items[:N]

def main():
    # Import the data
    df = pd.read_csv('ratings.csv')

    # Get the recommendations for the user
    recommendations = get_recommended_items(df, 1)

    # Print the recommendations
    print(recommendations)

if __name__ == '__main__':
    main()
