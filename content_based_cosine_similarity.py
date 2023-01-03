import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_recommended_items(df, X, item, N=10):
    """
    Returns the top N recommended items for a given item based on the cosine similarity of their word counts.
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing item metadata with a column for 'item_id'.
    X (scipy.sparse.csr.csr_matrix): A matrix of word counts for each item.
    item (int): The index of the item in the dataframe and matrix.
    N (int): The number of recommended items to return (default is 10).
    
    Returns:
    pandas.Series: A series of the recommended item IDs.
    """
    # Calculate the cosine similarities
    sims = cosine_similarity(X, X[item, :]).flatten()
    
    # Sort the similarities in descending order
    sorted_sims = sims.argsort()[::-1]
    
    # Get the top N items with the highest similarities
    top_items = sorted_sims[:N]
    
    # Return the item IDs
    return df['item_id'].iloc[top_items]

def main():
    # Import the data
    df = pd.read_csv('item-metadata.csv')

    # Extract the item descriptions and create a list of descriptions
    descriptions = df['description'].tolist()

    # Create a TfidfVectorizer to convert the descriptions to a matrix of word counts
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(descriptions)

    # Get the recommendations for the item
    recommendations = get_recommended_items(df, X, 1)

    # Print the recommendations
    print(recommendations)

if __name__ == '__main__':
    main()
