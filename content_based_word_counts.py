import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def get_recommended_items(df, X, item, N=10):
    """
    Returns the top N recommended items for a given item based on the similarity of their word counts.
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing item metadata with a column for 'item_id'.
    X (scipy.sparse.csr.csr_matrix): A matrix of word counts for each item.
    item (int): The index of the item in the dataframe and matrix.
    N (int): The number of recommended items to return (default is 10).
    
    Returns:
    pandas.Series: A series of the recommended item IDs.
    """
    # Get the item's row from the matrix
    item_row = X[item, :]
    
    # Calculate the dot product between the item's row and each other item's row
    scores = X.dot(item_row.T).toarray().flatten()
    
    # Sort the scores in descending order
    sorted_scores = scores.argsort()[::-1]
    
    # Get the top N items with the highest scores
    top_items = sorted_scores[:N]
    
    # Return the item IDs
    return df['item_id'].iloc[top_items]

def main():
    # Import the data
    df = pd.read_csv('item-metadata.csv')

    # Extract the item descriptions and create a list of descriptions
    descriptions = df['description'].tolist()

    # Create a CountVectorizer to convert the descriptions to a matrix of word counts
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(descriptions)

    # Get the recommendations for the item
    recommendations = get_recommended_items(df, X, 1)

    # Print the recommendations
    print(recommendations)

if __name__ == '__main__':
    main()
