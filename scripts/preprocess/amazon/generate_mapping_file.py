import sys
from pathlib import Path
import os
import json
from sklearn.preprocessing import LabelEncoder


# 引入项目根目录
home_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(home_dir))

import utils.amazon_data_util as data_util
import utils.path_util as path_util

'''
生成用户id和物品id的映射文件，用于模型嵌入层训练
'''


# 读取数据
data = data_util.get_raw_data()
print(f'raw data has been loaded')

# 生成用户 ID 和物品 ID 映射
user_mapping = {user: idx for idx, user in enumerate(data["user_id"].unique())}
item_mapping = {item: idx for idx, item in enumerate(data["item_id"].unique())}

# 生成用户 ID 和物品 ID 映射
user_mapping = {user: idx for idx, user in enumerate(data["user_id"].unique())}
item_mapping = {item: idx for idx, item in enumerate(data["item_id"].unique())}

# 保存映射为 JSON 文件
user_path = os.path.join(path_util.get_amazon_preprocess_fp(), 'user_mapping.json')
with open(user_path, "w") as user_file:
    data_util.save_json_file(user_mapping, user_path)
print(f'user mapping json has been saved, path = {user_path}')

item_path = os.path.join(path_util.get_amazon_preprocess_fp(), 'item_mapping.json')
with open(item_path, "w") as user_file:
    data_util.save_json_file(item_mapping, item_path)
print(f'item mapping json has been saved, path = {item_path}')