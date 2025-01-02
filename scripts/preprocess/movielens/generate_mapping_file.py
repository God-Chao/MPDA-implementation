import sys
from pathlib import Path
import os
from sklearn.preprocessing import LabelEncoder


# 引入项目根目录
home_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(home_dir))

import utils.movielens_data_util as data_util
import utils.path_util as path_util

'''
生成用户id和物品id的映射文件，用于模型嵌入层训练
'''


# 读取数据
data = data_util.get_raw_data()
print(f'raw data has been loaded')

user_encoder = LabelEncoder()
data['user'] = user_encoder.fit_transform(data['user_id'])

item_encoder = LabelEncoder()
data['item'] = item_encoder.fit_transform(data['item_id'])

# 编码映射关系
user_mapping = dict(zip(user_encoder.classes_, user_encoder.transform(user_encoder.classes_)))
item_mapping = dict(zip(item_encoder.classes_, item_encoder.transform(item_encoder.classes_)))


user_mapping = {int(k): int(v) for k, v in user_mapping.items()}
user_path = os.path.join(path_util.get_movielens_preprocess_fp(), 'user_mapping.json')
data_util.save_json_file(user_mapping, user_path)
print(f'user mapping json has been saved, path = {user_path}')


item_mapping = {int(k): int(v) for k, v in item_mapping.items()}
item_path = os.path.join(path_util.get_movielens_preprocess_fp(), 'item_mapping.json')
data_util.save_json_file(item_mapping, item_path)
print(f'item mapping json has been saved, path = {item_path}')