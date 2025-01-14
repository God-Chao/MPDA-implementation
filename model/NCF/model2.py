import torch
import torch.nn as nn
import os
import yaml
import sys
from pathlib import Path

# 引入项目根目录
home_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(home_dir))

import utils.movielens_data_util as data_util

# 定义NCF模型
class NCF(nn.Module): 
    def __init__(self, device):
        super(NCF, self).__init__()
        # 读取配置文件
        file_path = os.path.abspath(__file__) # 获取当前 Python 文件的绝对路径
        # 获取当前 Python 文件所在的目录
        current_directory = os.path.dirname(file_path)
        config_fp = os.path.join(current_directory, 'config.yml') # 配置文件位置
        with open(config_fp, "r") as file:
            config = yaml.safe_load(file)
        # 访问变量
        num_user_embedding = config['num_user_embedding']
        num_item_embedding = config['num_item_embedding']
        embedding_dim = config['embedding_dim']

        # 用户和物品的嵌入层
        self.user_embedding = nn.Embedding(num_user_embedding, embedding_dim)
        self.item_embedding = nn.Embedding(num_item_embedding, embedding_dim)

        # 加载用户和物品mapping文件
        self.user_mapping = data_util.get_user_mapping()
        self.item_mapping = data_util.get_item_mapping()
        
        self.device = device
        
        # MLP层
        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128), # 将用户和物品嵌入拼接作为输入
            nn.ReLU(),

            nn.Linear(128, 64), # 将用户和物品嵌入拼接作为输入
            nn.ReLU(),

            nn.Linear(64, 32), # 将用户和物品嵌入拼接作为输入
            nn.ReLU(),
        )

        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid() # 将输出值压缩到[0, 1]之间，用于概率预测
        )
        
    def forward(self, user_indices, item_indices):
        # 获取嵌入向量
        user_id = data_util.get_user_mapping_id(self.user_mapping, user_indices).to(self.device)
        item_id = data_util.get_item_mapping_id(self.item_mapping, item_indices).to(self.device)

        user_vector = self.user_embedding(user_id)
        item_vector = self.item_embedding(item_id)
        
        # 拼接用户和物品向量
        vector = torch.cat([user_vector, item_vector], dim=-1)
        
        # MLP层
        output = self.mlp_layers(vector)

        # 输出层
        predict = self.output_layers(output)

        return predict
    
    def get_item_embedding(self, item_indices):
        item_id = data_util.get_item_mapping_id(self.item_mapping, item_indices)
        item_vector = self.item_embedding(item_id.to(self.device))
        return item_vector
    
    def get_item_embedding(self, item_indices):
        item_id = data_util.get_item_mapping_id(self.item_mapping, item_indices)
        item_vector = self.item_embedding(item_id.to(self.device))
        return item_vector
    

