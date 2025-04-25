from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import pandas as pd
import os

def build_dataset(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
    return data

def user_based_cf(data, sim_name='cosine', k=40):
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    sim_option = {
        'name': sim_name,
        'user_based': True
    }
    algo = KNNBasic(sim_options=sim_option, k=k)
    algo.fit(trainset)
    predictions = algo.test(testset)
    return algo, predictions

def item_based_cf(data, sim_name='cosine', k=40):
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    sim_option = {
        'name': sim_name,
        'user_based': False
    }
    algo = KNNBasic(sim_options=sim_option, k=k)
    algo.fit(trainset)
    predictions = algo.test(testset)
    return algo, predictions

def get_top_n_recommendations(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def save_recommendations(top_n, file_path="results/top_n_recommendations.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    rows = []
    for user_id, recommendations in top_n.items():
        for movie_id, score in recommendations:
            rows.append({'user_id': user_id, 'movie_id': movie_id, 'estimated_rating': score})
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)
