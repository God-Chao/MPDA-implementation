import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import os
import random
import warnings
from openpyxl import Workbook


# 引入项目根目录
home_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(home_dir))

import utils.config_util as config_util
import utils.train_util as train_util
import utils.path_util as path_util
from scripts.match import *
from model.MaskNet.model import Model as MaskNet

warnings.filterwarnings("ignore", category=FutureWarning)


def config_args():
    parser = argparse.ArgumentParser('train and test model for users in four models')

    parser.add_argument('-dataset', type=str, default='movielens', choices=['movielens', 'amazon'], help='训练集名称')
    parser.add_argument('-epochs', type=int, default=1, help='模型在每个用户训练集上微调的epoch')
    parser.add_argument('-device', type=str, default='cuda:2', help='训练模型的设备')
    parser.add_argument('-batch_size', type=int, default=64, help='batch大小')
    parser.add_argument('-task_index', type=int, default=0, help='任务并行工作下标')
    parser.add_argument('-num_task', type=int, default=config_util.get_num_task(), help='总并行任务个数')
    parser.add_argument('-lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('-model', type=str, default='NCF', help='模型名称')

    args = parser.parse_args()
    return args


def main():
    # 配置参数
    args = config_args()
    print(f'{vars(args)}')

    # 设置随机种子
    seed = config_util.get_random_seed()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'random seed has benn setted')

    # 设置超参数
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = torch.device(args.device)
    batch_size = args.batch_size
    task_index = args.task_index
    num_task = args.num_task
    dataset_name = args.dataset
    model_name = args.model

    # 导入对应的datautil
    if dataset_name == 'movielens':
        import utils.movielens_data_util as data_util
    elif dataset_name == 'amazon':
        import utils.amazon_data_util as data_util

    # 加载测试用户和划分task_index
    test_users = data_util.get_user_with_train_and_test()
    test_users = np.array_split(test_users, num_task)[task_index]
    print(f'test users have been loaded, len = {len(test_users)}')
    print(f'test users = {test_users}')
    
    # 初始化结果保存列表
    user_id_list =[]
    num_train_samples_list = []
    num_test_samples_list =[]
    cloud_list = []

    # 加载全局数据
    raw_data = data_util.get_raw_data()
    print(f'raw data has been loaded')

    start_time = datetime.now()
    print(f'[{start_time}] start test on test users')

    for index, user in enumerate(test_users):
        # 加载全局模型
        init_model = train_util.get_init_model_by_name(model_name, dataset_name, device)
        init_model.to(device)
        print(f'init model has been laoded, init_model = {init_model}')

        print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)}')

        user_id_list.append(user)

        train_data = data_util.get_user_train_data(raw_data, user)
        test_data = data_util.get_user_test_data(raw_data, user)

        test_loader = train_util.get_data_loader(test_data, dataset_name, batch_size, False)
        print(f'train_loader and test_loader has been loaded')

        num_train_samples_list.append(len(train_data))
        num_test_samples_list.append(len(test_data))

        # Cloud: 直接用初始模型在本地测试集上测试
        print(f'[{datetime.now()}] user = {user} Cloud {index}/{len(test_users)}')
        cloud_auc = train_util.test_model_with_dataset(init_model, test_loader, device)
        cloud_list.append(cloud_auc)
        print(f'Cloud user{user} = {cloud_auc}')

    # 保存数据
    run_name = dataset_name +'_'+ model_name + '_Cloud'
    log_fp = os.path.join(path_util.get_log_fp(), run_name)

    if not os.path.exists(log_fp):
        os.makedirs(log_fp)
        print(f"log_fp = {log_fp} created")

    names = ['user_id', 'num_train_samples', 'num_test_samples', 'Cloud']
    # 创建一个工作簿和工作表
    wb = Workbook()
    ws = wb.active
    
    # 设置列标题
    ws.append(names)
    
    # 将数据写入表格
    for user_id, train_samples, test_samples, cloud in zip(user_id_list, num_train_samples_list, num_test_samples_list, cloud_list):
        ws.append([user_id, train_samples, test_samples, cloud])
    
    # 保存 Excel 文件
    file_path = os.path.join(log_fp, str(task_index) + '.xlsx')
    wb.save(file_path)
    print(f"文件已保存为 {file_path}")

if __name__ == '__main__':
    main()