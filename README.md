# 🎬 MovieMate: Collaborative Filtering Recommender System

**MovieMate** is a movie recommendation engine powered by **Collaborative Filtering**. Built using the **MovieLens 100K** dataset, it predicts what users would like to watch based on their previous ratings and the behavior of similar users or items.

---

## 📦 Dataset
- **MovieLens 100K**: 100,000 ratings from 943 users on 1,682 movies
- [Dataset Link (Kaggle)](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)
- Files used: `u.data`, `u.item`, `u.user`

---

## 🚀 Features
- Load and preprocess the MovieLens dataset
- Build **User-Based** and **Item-Based Collaborative Filtering** models
- Evaluate models using:
  - RMSE
  - Precision@10
  - Recall@10
- Visualize:
  - Distribution of ratings
  - Top movies
  - Heatmaps of user-item matrix
  - True vs Predicted ratings
- Save:
  - Recommendations to CSV
  - Evaluation metrics to JSON

---

## 🧠 Project Structure
```
Collaborative-Filtering-based-Recommendation-System/
├── data/                         # Raw MovieLens data
├── results/                      # Output files (CSV/JSON)
├── src/                          # Source code (recommender, evaluation, loading)
│   ├── data_loader.py
│   ├── recommender.py
│   └── evaluation.py
├── notebooks/                    # Jupyter Notebooks for analysis and modeling
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_user_based_cf.ipynb
│   ├── 03_item_based_cf.ipynb
│   └── 04_evaluation_and_visualization.ipynb
└── README.md
```


---



