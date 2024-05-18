import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate, train_test_split
from collections import defaultdict


ratings_path = 'D:/sjj/ratings.csv'
movies_path = 'D:/sjj/movies.csv'
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

# Data cleansing and creation of smaller subsets of data
ratings.dropna(inplace=True)
ratings = ratings[(ratings['rating'] >= 0.5) & (ratings['rating'] <= 5)]
subset_size = 5000
ratings = ratings[ratings['userId'] <= subset_size]

# Data consolidation
ratings = ratings.merge(movies[['movieId', 'title']], on='movieId', how='left')


reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# item-based KNN
model = KNNBasic(sim_options={'user_based': False})  # 设为基于物品的CF

# model
training_set = data.build_full_trainset()
model.fit(training_set)

# Cross-validation assessment
results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Segmented data sets
trainset, testset = train_test_split(data, test_size=0.25)
predictions = model.test(testset)

# Visualisation of error distribution
errors = [abs(true_r - est) for (_, _, true_r, est, _) in predictions]
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=30, color='blue', edgecolor='black')
plt.title('Error Distribution')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.show()

# display recommendation
def display_recommendations(user_id):
    user_ratings = [r for r in predictions if r[0] == user_id]
    if not user_ratings:
        print("No data available for user:", user_id)
        return
    recommended_movies = pd.DataFrame(user_ratings, columns=['userId', 'movieId', 'actual', 'est', 'details'])
    recommended_movies = recommended_movies.merge(movies, on='movieId')
    recommended_movies.sort_values(by='est', ascending=False, inplace=True)
    sns.barplot(x='est', y='title', data=recommended_movies.head(10), palette='viridis')
    plt.title(f'Top 10 Recommended Movies for User {user_id}')
    plt.xlabel('Estimated Rating')
    plt.ylabel('Movie Title')
    plt.show()

# user id 1
display_recommendations(1)

# recall
def calculate_recall(predictions, threshold):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_rel = sum((est >= threshold) and (true_r >= threshold) for (est, true_r) in user_ratings)
        recalls[uid] = n_rec_rel / n_rel if n_rel != 0 else 0
    return sum(rec for rec in recalls.values()) / len(recalls)

average_recall = calculate_recall(predictions, 3.5)
print(f"average recall: {average_recall:.2f}")
print(f"average RMSE: {results['test_rmse'].mean():.2f}")
print(f"average MAE: {results['test_mae'].mean():.2f}")
