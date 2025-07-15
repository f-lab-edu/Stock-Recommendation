import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Dict, List

'''
현재는 책에 있는 코드를 그대로 옮겨 놓았는데 조금 수정할 예정입니다.
'''


def precision_at_k(true_items: List[int], pred_items: List[int], k: int) -> float:
    if k == 0:
        return 0.0

    p_at_k = (len(set(true_items) & set(pred_items[:k]))) / k
    return p_at_k

def recall_at_k(true_items: List[int], pred_items: List[int], k: int) -> float:
    if len(true_items) == 0 or k == 0:
        return 0.0

    r_at_k = (len(set(true_items) & set(pred_items[:k]))) / len(true_items)
    return r_at_k

def calc_rmse(true_rating: List[float], pred_rating: List[float]) -> float:
    return np.sqrt(mean_squared_error(true_rating, pred_rating))

def calc_recall_at_k(true_user2items: Dict[int, List[int]], pred_user2items: Dict[int, List[int]], k: int) -> float:
    scores = []
    for user_id in true_user2items.keys():
        r_at_k = recall_at_k(true_user2items[user_id], pred_user2items[user_id], k)
        scores.append(r_at_k)
    return np.mean(scores)

def calc_precision_at_k(true_user2items: Dict[int, List[int]], pred_user2items: Dict[int, List[int]], k: int) -> float:
    scores = []
    for user_id in true_user2items.keys():
        p_at_k = precision_at_k(true_user2items[user_id], pred_user2items[user_id], k)
        scores.append(p_at_k)
    return np.mean(scores)