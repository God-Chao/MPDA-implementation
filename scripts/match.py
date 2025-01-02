import random
import numpy as np
import sys
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# 引入项目根目录
home_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(home_dir))

import utils.movielens_data_util as movielens_data_util
import utils.amazon_data_util as amazon_data_util

# 随机匹配
def random_match(k, dataset):
    all_users = movielens_data_util.get_user_with_train() if dataset == 'movielens' else amazon_data_util.get_user_with_train()
    selected_users = random.sample(all_users, k)
    return selected_users

# 根据knn算法匹配相似用户
def knn(user, user_vectors, k):
    # 提取用户 ID 和对应向量
    user_ids = list(user_vectors.keys())
    vectors = list(user_vectors.values())
    
    # 找到目标用户的向量
    user_vector = np.array(user_vectors[user]).reshape(1, -1)
    
    # 初始化最近邻搜索器
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine')  # 使用余弦相似度
    nbrs.fit(vectors)
    
    # 查询最近邻
    distances, indices = nbrs.kneighbors(user_vector)
    
    # 跳过第一个结果（是自己），返回用户ID和对应距离
    neighbors = [user_ids[idx] for idx in indices[0][1:]]
    return neighbors