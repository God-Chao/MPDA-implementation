import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
import os
import random
import warnings


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

    parser.add_argument('-recall_num', type=int, default=100, help='云端召回用户数量')
    parser.add_argument('-dataset', type=str, default='movielens', choices=['movielens', 'amazon'], help='训练集名称')
    parser.add_argument('-recall_alg', type=str, default='random', choices=('random', 'item_interaction', 'item_interaction_with_random_mask', 
    'item_interaction_with_hypernet', 'recall_item_pair_similarity', 'item_interaction_with_single_mask', 'item_interaction_with_double_mask', 'item_interaction_with_triple_mask'), help='云端召回算法')
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
    recall_num = args.recall_num
    recall_alg = args.recall_alg
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
    num_selected_users_list =[]
    num_train_samples_list = []
    num_test_samples_list =[]
    local_plus_list = []
    mpda_minus_list = []
    mpda_list = []

    # 加载全局数据
    raw_data = data_util.get_raw_data()
    print(f'raw data has been loaded')
    
    # 四种算法的预处理
    if recall_alg == 'item_interaction':
        init_model = train_util.get_init_model_by_name(model_name, dataset_name, device)
        init_model.to(device)
        user_vectors = {}
        for index, user in enumerate(test_users):
            history_items = data_util.get_trainset_item_interaction(raw_data, user)
            print(f'history_items = {history_items}')
            item_features = init_model.get_item_embedding(torch.tensor(history_items))
            user_feature = torch.mean(item_features, dim=0)
            user_vectors[user] = user_feature.detach().cpu().numpy()
            print(f'[{datetime.now()}] user{user} feature vector has been loaded {index}/{len(test_users)}')
        print(f'[{datetime.now()}] user_vectors has been loaded, user_vectors = {user_vectors}')
    if recall_alg == 'item_interaction_with_hypernet':
        init_model = train_util.get_init_model_by_name(model_name, dataset_name, device)
        init_model.to(device)
        # 加载掩码网络
        maks_model = MaskNet(device)
        mask_model_saved_path = os.path.join(path_util.get_model_fp_by_name('MaskNet'), 'movielens_model_0.95.pth')
        print(f'mask_model_saved_path = {mask_model_saved_path}')
        maks_model.load_state_dict(torch.load(mask_model_saved_path))
        print(f'mask_model has been loaded, maks_model = {maks_model}')
        user_vectors = {}
        for index, user in enumerate(test_users):
            history_items = data_util.get_trainset_item_interaction(raw_data, user)

            # 经过超网过滤
            item_indices = torch.tensor(history_items)
            user_indices = torch.ones_like(item_indices) * user
            item_indices.to(device)
            user_indices.to(device)
            mask, preds = maks_model(user_indices, item_indices)
            mask = mask.squeeze(1).bool().cpu()
            filtered_items = (torch.tensor(history_items))[mask]

            item_features = init_model.get_item_embedding(torch.tensor(filtered_items))
            user_feature = torch.mean(item_features, dim=0)
            user_vectors[user] = user_feature.detach().cpu().numpy()
            print(f'[{datetime.now()}] user{user} feature vector has been loaded {index}/{len(test_users)}')
        print(f'[{datetime.now()}] user_vectors has been loaded, user_vectors = {user_vectors}')
    elif recall_alg == 'item_interaction_with_random_mask':
        print(f'recall_alg = item_interaction_with_random_mask')
        init_model = train_util.get_init_model_by_name(model_name, dataset_name, device)
        init_model.to(device)
        user_vectors = {}
        for index, user in enumerate(test_users):
            history_items = data_util.get_trainset_item_interaction(raw_data, user)
            # 对items做随机掩码
            ratio = config_util.get_random_mask_ratio()
            num_to_remove = int(len(history_items) * ratio)
            indices_to_remove = set(random.sample(range(len(history_items)), num_to_remove))
            history_items = [item for idx, item in enumerate(history_items) if idx not in indices_to_remove]

            item_features = init_model.get_item_embedding(torch.tensor(history_items))
            user_feature = torch.mean(item_features, dim=0)
            user_vectors[user] = user_feature.detach().cpu().numpy()
            print(f'[{datetime.now()}] user{user} feature vector has been loaded {index}/{len(test_users)}')
        print(f'[{datetime.now()}] user_vectors has been loaded, user_vectors = {user_vectors}')
    elif recall_alg == 'item_interaction_with_single_mask':
        init_model = train_util.get_init_model_by_name(model_name, dataset_name, device)
        init_model.to(device)
        user_vectors = {}
        for index, user in enumerate(test_users):
            history_items = data_util.get_trainset_item_interaction(raw_data, user)
            # 随机删除一个元素
            if len(history_items) > 1:
                index_to_remove = random.randint(0, len(history_items) - 1)
                history_items = np.delete(history_items, index_to_remove)
            item_features = init_model.get_item_embedding(torch.tensor(history_items))
            user_feature = torch.mean(item_features, dim=0)
            user_vectors[user] = user_feature.detach().cpu().numpy()
            print(f'[{datetime.now()}] user{user} feature vector has been loaded {index}/{len(test_users)}')
        print(f'[{datetime.now()}] user_vectors has been loaded, user_vectors = {user_vectors}')
    elif recall_alg == 'item_interaction_with_double_mask':
        init_model = train_util.get_init_model_by_name(model_name, dataset_name, device)
        init_model.to(device)
        user_vectors = {}
        for index, user in enumerate(test_users):
            history_items = data_util.get_trainset_item_interaction(raw_data, user)
            print(f'history_items = {history_items}, len={len(history_items)}')
            # 随机删两个元素
            if len(history_items) > 2:
                indices_to_remove = sorted(random.sample(range(len(history_items)), 2), reverse=True)
                for idx in indices_to_remove:
                    history_items = np.delete(history_items, idx)
            item_features = init_model.get_item_embedding(torch.tensor(history_items))
            user_feature = torch.mean(item_features, dim=0)
            user_vectors[user] = user_feature.detach().cpu().numpy()
            print(f'[{datetime.now()}] user{user} feature vector has been loaded {index}/{len(test_users)}')
        print(f'[{datetime.now()}] user_vectors has been loaded, user_vectors = {user_vectors}')
    elif recall_alg == 'item_interaction_with_triple_mask':
        init_model = train_util.get_init_model_by_name(model_name, dataset_name, device)
        init_model.to(device)
        user_vectors = {}
        for index, user in enumerate(test_users):
            history_items = data_util.get_trainset_item_interaction(raw_data, user)
            # 随机删三个元素
            if len(history_items) > 3:
                indices_to_remove = sorted(random.sample(range(len(history_items)), 3), reverse=True)
                for idx in indices_to_remove:
                    history_items = np.delete(history_items, idx)
            item_features = init_model.get_item_embedding(torch.tensor(history_items))
            user_feature = torch.mean(item_features, dim=0)
            user_vectors[user] = user_feature.detach().cpu().numpy()
            print(f'[{datetime.now()}] user{user} feature vector has been loaded {index}/{len(test_users)}')
        print(f'[{datetime.now()}] user_vectors has been loaded, user_vectors = {user_vectors}')
    elif recall_alg == 'recall_item_pair_similarity':
        recall_item_pairs = data_util.get_recall_item_pairs()
        print(f'recall item pairs = {recall_item_pairs}')
        user_vectors = {}
        criterion = nn.BCELoss() # 二元交叉熵损失
        for index, user in enumerate(test_users):
            # 现在本地数据上训练一次
            init_model = train_util.get_init_model_by_name(model_name, dataset_name, device)
            init_model.to(device)
            optimizer = torch.optim.Adam(init_model.parameters(), lr=lr)
            print(f'init model and optimizer has been loaded')

            train_data = data_util.get_user_train_data(raw_data, user)
            train_loader = train_util.get_data_loader(train_data, dataset_name, batch_size, True)
            print(f'train_loader has been loaded')

            train_util.train_model_with_dataset(init_model, criterion, optimizer, train_loader, device)
            user_feature = train_util.get_user_feature_by_recall_item_pairs(init_model, recall_item_pairs)
            user_vectors[user] = user_feature
            print(f'[{datetime.now()}] user{user} feature vector has been loaded {index}/{len(test_users)}')
        print(f'[{datetime.now()}] user_vectors has been loaded, user_vectors = {user_vectors}')

    start_time = datetime.now()
    print(f'[{start_time}] start test on test users')

    for index, user in enumerate(test_users):
        # 加载全局模型
        init_model = train_util.get_init_model_by_name(model_name, dataset_name, device)
        init_model.to(device)
        print(f'init model has been laoded, init_model = {init_model}')

        print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)}')
        criterion = nn.BCELoss() # 二元交叉熵损失

        user_id_list.append(user)
        augumented_users = []

        # 不同召回算法的增强用户的检索
        if recall_alg == 'random':
            augumented_users = random_match(recall_num, dataset_name)
            print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)} recall augumented users by random, augumented_users = {augumented_users}')
        elif recall_alg == 'item_interaction':
            augumented_users = knn(user, user_vectors, recall_num)
            print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)} recall augumented users by item_interaction, augumented_users = {augumented_users}')
        elif recall_alg == 'item_interaction_with_random_mask':
            augumented_users = knn(user, user_vectors, recall_num)
            print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)} recall augumented users by item_interaction_with_random_mask, augumented_users = {augumented_users}')
        elif recall_alg == 'item_interaction_with_hypernet':
            augumented_users = knn(user, user_vectors, recall_num)
            print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)} recall augumented users by item_interaction_with_hypernet, augumented_users = {augumented_users}')
        elif recall_alg == 'item_interaction_with_single_mask':
            augumented_users = knn(user, user_vectors, recall_num)
            print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)} recall augumented users by item_interaction_with_single_mask, augumented_users = {augumented_users}')
        elif recall_alg == 'item_interaction_with_double_mask':
            augumented_users = knn(user, user_vectors, recall_num)
            print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)} recall augumented users by item_interaction_with_single_mask, augumented_users = {augumented_users}')
        elif recall_alg == 'item_interaction_with_triple_mask':
            augumented_users = knn(user, user_vectors, recall_num)
            print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)} recall augumented users by item_interaction_with_single_mask, augumented_users = {augumented_users}')
        elif recall_alg == 'recall_item_pair_similarity':
            augumented_users = knn(user, user_vectors, recall_num)
            print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)} recall augumented users by recall_item_pair_similarity, augumented_users = {augumented_users}')

        train_data = data_util.get_user_train_data(raw_data, user)
        test_data = data_util.get_user_test_data(raw_data, user)

        train_loader = train_util.get_data_loader(train_data, dataset_name, batch_size, True)
        test_loader = train_util.get_data_loader(test_data, dataset_name, batch_size, False)
        print(f'train_loader and test_loader has been loaded')

        num_train_samples_list.append(len(train_data))
        num_test_samples_list.append(len(test_data))

        # Local+: 使用Cloud模型作为初始模型在本地训练集+增强数据上训练一个epoch
        print(f'[{datetime.now()}] user = {user} Local+ {index}/{len(test_users)}')
        init_model = train_util.get_init_model_by_name(model_name, dataset_name, device)
        init_model.to(device)
        optimizer = torch.optim.Adam(init_model.parameters(), lr=lr)
        train_util.train_model_with_dataset(init_model, criterion, optimizer, train_loader, device)
        for augumented_user in augumented_users:
            augumented_data = data_util.get_user_train_data(raw_data, augumented_user)
            augumented_loader = train_util.get_data_loader(augumented_data, dataset_name, batch_size, True)
            train_util.train_model_with_dataset(init_model, criterion, optimizer, augumented_loader, device)
        local_plus_auc = train_util.test_model_with_dataset(init_model, test_loader, device)
        local_plus_list.append(local_plus_auc)
        print(f'Local+  user{user} = {local_plus_auc}')

        # MPDA-: 使用Cloud模型作为初始模型在增强数据集上增量训练
        # MPDA: 使用Cloud模型作为初始模型在本地数据+增强数据进行增量训练
        print(f'[{datetime.now()}] user = {user} MPDA- {index}/{len(test_users)}')
        init_model = train_util.get_init_model_by_name(model_name, dataset_name, device)
        init_model.to(device)
        optimizer = torch.optim.Adam(init_model.parameters(), lr=lr)

        best_auc = train_util.test_model_with_dataset(init_model, test_loader, device)
        current_model = train_util.get_init_model_by_name(model_name, dataset_name, device)
        current_model.to(device)
        # 在增强用户上做增强训练，只选择提升了模型性能的增强用户
        current_model, best_auc, num_selected_users = train_util.incremental_training(init_model, criterion, dataset_name, raw_data, lr, test_loader, augumented_users, device, batch_size, best_auc)
        optimizer = torch.optim.Adam(current_model.parameters(), lr=lr)
        mpda_minus_list.append(best_auc)
        print(f'MPDA- user{user} = {best_auc}')

        # 在本地训练集上训练一个epoch
        train_util.train_model_with_dataset(current_model, criterion, optimizer, train_loader, device)
        mpda_auc = train_util.test_model_with_dataset(current_model, test_loader, device)
        mpda_list.append(mpda_auc)
        print(f'MPDA user{user}= {mpda_auc}')
        num_selected_users_list.append(num_selected_users)

    run_name = 'transfer_' + dataset_name +'_'+ model_name +'_' + str(recall_num) +'_' + recall_alg
    log_fp = os.path.join(path_util.get_log_fp(), run_name)
    train_util.result_to_xlsx(user_id_list, num_selected_users_list, num_train_samples_list, num_test_samples_list, local_plus_list, mpda_minus_list, mpda_list, log_fp, task_index)


if __name__ == '__main__':
    main()