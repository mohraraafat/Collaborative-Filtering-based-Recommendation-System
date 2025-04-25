import pandas as pd
import os

def load_data(data_path='data/'):
    # Check required files exist
    required_files = ['u.data', 'u.item', 'u.user']
    for file in required_files:
        full_path = os.path.join(data_path, file)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"❌ File not found: {full_path}")

    # Load ratings
    ratings = pd.read_csv(
        os.path.join(data_path, 'u.data'),
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )

    # Convert timestamp to datetime
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    # Load movie info
    movies = pd.read_csv(
        os.path.join(data_path, 'u.item'),
        sep='|',
        encoding='latin-1',
        header=None,
        names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
               'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    )

    # Load user info
    users = pd.read_csv(
        os.path.join(data_path, 'u.user'),
        sep='|',
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
    )

    return ratings, movies, users

def preprocess_data(ratings):
    # Create user-item matrix
    user_item_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating')
    return user_item_matrix
