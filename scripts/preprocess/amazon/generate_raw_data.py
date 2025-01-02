import os
import yaml
import sys
from pathlib import Path
import json
import csv

# 引入项目根目录
home_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(home_dir))

import utils.path_util as path_util

'''
将Amazon Electronics原始json文件取出用户id，物品id，评分，时间戳四列单独保存为一个csv文件
'''

# 读取原始数据
raw_data_path = path_util.get_amazon_raw_data_path()
with open(raw_data_path, 'r') as file:
    content = file.read()
json_array = '[' + content.replace('}\n{', '},{') + ']'
raw_data = json.loads(json_array)

# 保存为CSV文件
csv_file_path = os.path.join(path_util.get_amazon_preprocess_fp(), 'ratings.csv')
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(["user_id", "item_id", "rating", "timestamp"])
    # 写入数据
    for row in raw_data:
        writer.writerow([row["reviewerID"], row["asin"], row["overall"], row["unixReviewTime"]])

print('rating.csv has been successfully generated')