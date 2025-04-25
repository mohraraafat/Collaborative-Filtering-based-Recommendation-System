from surprise import accuracy
from collections import defaultdict
import json
import os

def compute_rmse(predictions):
    """Compute RMSE using Surprise's built-in function"""
    return accuracy.rmse(predictions, verbose=True)

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Compute precision and recall at K for each user
    """
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    return precisions, recalls

def evaluate_and_report(predictions, k=10, threshold=3.5, save_path="results/evaluation_metrics.json"):
    """
    Compute RMSE, average precision/recall, and save to file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("\n🔍 Evaluation Metrics:")
    rmse = compute_rmse(predictions)
    precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=threshold)

    avg_precision = sum(precisions.values()) / len(precisions)
    avg_recall = sum(recalls.values()) / len(recalls)

    print(f"📌 Average Precision@{k}: {avg_precision:.4f}")
    print(f"📌 Average Recall@{k}: {avg_recall:.4f}")

    metrics = {
        "RMSE": rmse,
        f"Precision@{k}": avg_precision,
        f"Recall@{k}": avg_recall
    }

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"📁 Metrics saved to: {save_path}")
    return metrics
