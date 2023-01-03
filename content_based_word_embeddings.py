import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

def get_recommended_items(df, X, item, N=10):
    """
    Returns the top N recommended items for a given item based on the similarity of their word embeddings.
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing item metadata with a column for 'item_id'.
    X (numpy.ndarray): A matrix of word embeddings for each item.
    item (int): The index of the item in the dataframe and matrix.
    N (int): The number of recommended items to return (default is 10).
    
    Returns:
    pandas.Series: A series of the recommended item IDs.
    """
    # Get the item's row from the matrix
    item_row = X[item, :]
    
    # Calculate the dot product between the item's row and each other item's row
    scores = X.dot(item_row)
    
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

    # Load the pre-trained word embeddings
    word_vectors = KeyedVectors.load_word2vec_format('word-vectors.bin', binary=True)

    # Create a matrix of word embeddings for each description
    X = np.zeros((len(descriptions), word_vectors.vector_size))
    for i, description in enumerate(descriptions):
        # Split the description into a list of words
        words = description.split()

        # Get the word embeddings for each word
        embeddings = [word_vectors[word] for word in words if word in word_vectors]

        # If the description contains at least one word embedding, average the embeddings
        if len(embeddings) > 0:
            X[i, :] = np.mean(embeddings, axis=0)

    # Get the recommendations for the item
    recommendations = get_recommended_items(df, X, 1)

    # Print the recommendations
    print(recommendations)

if __name__ == '__main__':
    main()
