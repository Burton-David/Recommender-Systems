import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_recommended_items(df, X, item, N=10):
    """
    Returns the top N recommended items for a given item based on the presence of common words in their descriptions.
    
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
    
    # Create a binary filter for the common words
    common_words = item_row.astype(bool).toarray().flatten()
    
    # Filter the matrix to only include the common words
    filtered_X = X[:, common_words]
    
    # Get the sum of the filtered matrix
    sums = filtered_X.sum(axis=1).flatten()
    
    # Sort the sums in descending order and get the top N items
    top_items = sums.argsort()[::-1][:N]
    
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
