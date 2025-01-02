from torch.utils.data import DataLoader
import torch
import random
import torch.nn as nn
import argparse
import sys
from pathlib import Path
import os

# 引入项目根目录
home_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(home_dir))


from model import *
import utils.movielens_data_util as movielens_data_util
import utils.amazon_data_util as amazon_data_util
import utils.train_util as train_util
import utils.path_util as path_util
from scripts.movielens_dataset import MovielensDataset
from scripts.amazon_dataset import AmazonDataset
from scripts.metric import *

def config_args():
    parser = argparse.ArgumentParser('train global model')
    parser.add_argument('-model', type=str, default='NCF', help='模型名称')
    parser.add_argument('-epochs', type=int, default=10, help='模型在每个训练集上训练的epoch')
    parser.add_argument('-device', type=str, default='cuda:2', help='训练模型的设备')
    parser.add_argument('-batch_size', type=int, default=64, help='batch大小')
    parser.add_argument('-dataset', type=str, default='movielens', choices=['movielens', 'amazon'], help='训练集名称')
    parser.add_argument('-lr', type=float, default=0.0001, help='学习率')

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
    device = torch.device(args.device)
    learning_rate = args.lr
    model_name = args.model
    dataset_name = args.dataset

    # 加载模型
    model = train_util.get_model_class_by_name(model_name, device).to(device)
    print(f'model has been loaded')

    # 加载训练集和测试集
    if dataset_name == 'movielens':
        train_data, test_data = movielens_data_util.get_train_test_data()
        trainset = MovielensDataset(train_data)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = MovielensDataset(test_data)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    elif dataset_name == 'amazon':
        train_data, test_data = amazon_data_util.get_train_test_data()
        trainset = AmazonDataset(train_data)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = AmazonDataset(test_data)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    print(f'train and test data has been loaded')

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f'criterion and optimizer had been loaded')

    # 训练和测试
    for epoch in range(epochs):
        train_util.train_model_with_dataset(model, criterion, optimizer, train_loader, device=device)
        auc = train_util.test_model_with_dataset(model, test_loader, device=device)
        print(f'epoch{epoch}/{epochs} auc = {auc}')
        # 保存模型，每个epoch保存一次
        model_save_path = os.path.join(path_util.get_model_fp_by_name(str(model_name)), f'model_epoch{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'model for epoch {epoch} has been saved to {model_save_path}')

    print('train finish')

if __name__ == '__main__':
    main()