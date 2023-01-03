# Recommender Systems Library
**This library contains a variety of different recommender systems implemented in Python. The following is a table of contents for the included files:**

1. user_based_cosine_similarity.py
2. item_based_dot_product.py
3. demographic_based_mean_rating.py
4. context_aware_mean_rating.py
5. hybrid_context_user.py
6. content_based_binary_filter.py
7. content_based_cosine_similarity.py
8. content_based_word_counts.py
9. content_based_word_embeddings.py
10. demographic_based_filtering.py

## Using the Library
To use any of the included files, simply import the file and call the get_recommended_items() function with the appropriate arguments.

For example, to use the user-based collaborative filtering recommender using cosine similarity:
```
import user_based_cosine_similarity

recommendations = user_based_cosine_similarity.get_recommended_items(df, 1)
```
The **df** parameter should be a pandas DataFrame containing ratings data with columns for 'user_id', 'item_id', and 'rating'. The second parameter is the ID of the user or item for which you want to get recommendations.

Each file has its own specific requirements for the data and parameters, so be sure to read the docstrings for more information.

### File Descriptions
user_based_cosine_similarity.py
This file contains a user-based collaborative filtering recommender that uses cosine similarity to compute the similarity

### item_based_dot_product.py
This file contains an item-based collaborative filtering recommender that uses the dot product of ratings to compute the similarity between items. Given an item's ID, the function returns the top N recommended items for that item based on the dot product of their ratings with other items.

### demographic_based_mean_rating.py
This file contains a demographic-based recommender that uses the age and gender of users to recommend items. Given a user's ID, the function returns the top N recommended items for that user based on the average rating of those items by users of the same age and gender.

### context_aware_mean_rating.py
This file contains a context-aware recommender that uses the context in which the recommendations will be used to recommend items. Given a context, the function returns the top N recommended items for that context based on the average rating of those items in that context.

### hybrid_context_user.py
This file contains a hybrid recommender that combines both context-aware and user-based collaborative filtering. Given a user's ID and a context, the function returns the top N recommended items for that user and context by combining the recommendations from both the context-aware and user-based collaborative filtering approaches.

### content_based_binary_filter.py
This file contains a content-based recommender that uses a binary filter to recommend items. Given a list of keywords, the function returns the top N recommended items that contain those keywords in their description.

### content_based_cosine_similarity.py
This file contains a content-based recommender that uses cosine similarity to recommend items. Given a list of keywords, the function returns the top N recommended items that are most similar to the keywords based on the cosine similarity of their descriptions.

### content_based_word_counts.py
This file contains a content-based recommender that uses word counts to recommend items. Given a list of keywords, the function returns the top N recommended items that contain the most occurrences of those keywords in their description.

### content_based_word_embeddings.py
This file contains a content-based recommender that uses word embeddings to recommend items. Given a list of keywords, the function returns the top N recommended items that are most similar to the keywords based on the cosine similarity of their word embeddings.

## demographic_based_filtering.py
This file contains a demographic-based recommender that uses the age and gender of users to recommend items. Given a user's ID, the function returns the top N recommended items for that user based on the average rating of those items by users of the same age and gender.
