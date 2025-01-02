import os
import json
import sys
from pathlib import Path
import pandas as pd
import random

# 引入项目根目录
home_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(home_dir))

import utils.movielens_data_util as data_util
import utils.config_util as config_util
import utils.path_util as path_util


'''
制定生成云端召回物品对的规则
'''

# 设置随机种子
seed = config_util.get_random_seed()
random.seed(seed)

popular_item_pairs_ratio = config_util.get_popular_item_pairs_ratio()
non_popular_item_pairs_ratio = 1 - popular_item_pairs_ratio


def recall_item_pairs(data, num_pairs, popular_item_pairs_ratio=0.5):
    # 统计每个物品的交互次数
    item_popularity = data['item_id'].value_counts()
    
    # 计算平均交互次数
    threshold = item_popularity.mean()
    
    # 划分流行和不流行物品
    popular_items = item_popularity[item_popularity > threshold].index.tolist()
    non_popular_items = item_popularity[item_popularity <= threshold].index.tolist()
    
    # 校验是否足够生成物品对
    if len(popular_items) < 2 or len(non_popular_items) < 2:
        raise ValueError("物品数量不足，无法生成物品对")
    
    # 生成流行物品对
    popular_pairs = set()  # 使用集合来去重
    while len(popular_pairs) < num_pairs * popular_item_pairs_ratio:
        pair = tuple(random.sample(popular_items, 2))  # random.sample不会重复抽取相同的物品，即生成的两个物品总是不同的。
        popular_pairs.add(pair)
    
    # 生成不流行物品对
    non_popular_pairs = set()  # 使用集合来去重
    while len(non_popular_pairs) < num_pairs * (1 - popular_item_pairs_ratio):
        pair = tuple(random.sample(non_popular_items, 2))
        non_popular_pairs.add(pair)
    
    # 将集合转换为列表，并返回
    return list(popular_pairs) + list(non_popular_pairs)

data = data_util.get_raw_data()
num_recall_item_pairs = config_util.get_num_recall_item_pairs()

recall_item_pairs = recall_item_pairs(data, num_recall_item_pairs)

# 保存到json文件中
json_path = os.path.join(path_util.get_movielens_preprocess_fp(), 'recall_item_pairs.json')

data_util.save_json_file(recall_item_pairs, json_path)

# 保存为 JSON 格式
print(f"物品对已保存为 JSON 文件: {json_path}")