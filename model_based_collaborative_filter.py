import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

# Import the data
df = pd.read_csv('user-item-data.csv')

# Create a pivot table of the data
df_pivot = df.pivot_table(index='user_id', columns='item_id', values='rating')

# Replace missing values with 0
df_pivot.fillna(0, inplace=True)

# Convert the pivot table to a matrix
X = df_pivot.values

# Create a TruncatedSVD model and fit it to the data
svd = TruncatedSVD(n_components=10, random_state=42)
X_transformed = svd.fit_transform(X)

# Normalize the data
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X_transformed)

# Create a function that returns the top N recommended items for a given user
def get_recommended_items(user, N=10):
    # Get the user's row from the matrix
    user_row = X_normalized[user-1, :]
    
    # Calculate the dot product between the user's row and each item's column
    scores = X_normalized.dot(user_row)
    
    # Sort the scores in descending order
    sorted_scores = scores.argsort()[::-1]
    
    # Get the top N items with the highest scores
    top_items = sorted_scores[:N]
    
    # Return the item IDs
    return df_pivot.columns[top_items]

# Get the recommendations for the user
recommendations = get_recommended_items(1)

# Print the recommendations
print(recommendations)
