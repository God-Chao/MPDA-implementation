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
import utils.path_util as path_util
import utils.train_util as train_util
from scripts.movielens_dataset import MovielensDataset
from scripts.amazon_dataset import AmazonDataset
from scripts.metric import *
from model.MaskNet.model import Model

def config_args():
    parser = argparse.ArgumentParser('train global model')
    parser.add_argument('-epochs', type=int, default=20, help='模型在每个训练集上训练的epoch')
    parser.add_argument('-device', type=str, default='cuda:2', help='训练模型的设备')
    parser.add_argument('-batch_size', type=int, default=64, help='batch大小')
    parser.add_argument('-dataset', type=str, default='movielens', choices=['movielens', 'amazon'], help='训练集名称')
    parser.add_argument('-lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('-alpha', type=float, default=0.5, help='KL散度损失权重')
    parser.add_argument('-beta', type=float, default=5, help='掩码稀疏度损失权重')
    parser.add_argument('-sparsity_ratio', type=float, default=0.9, help='掩码稀疏度期望值')
    parser.add_argument('-init_model_path', type=str, help='初始模型路径')


    args = parser.parse_args()
    return args

def getOptim(network, optim, lr, l2):
    weight_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask' not in p[0], network.named_parameters()))
    mask_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask' in p[0], network.named_parameters()))

    optim = optim.lower()
    if optim == "sgd":
        return [torch.optim.SGD(weight_params, lr=lr, weight_decay=l2), torch.optim.SGD(mask_params, lr=0.01 * lr)]
    elif optim == "adam":
        return [torch.optim.Adam(weight_params, lr=lr, weight_decay=l2), torch.optim.Adam(mask_params, lr=0.01 * lr)]
    else:
        raise ValueError("Invalid optimizer type: {}".format(optim))

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
    dataset_name = args.dataset
    alpha = args.alpha
    beta = args.beta
    sparsity_ratio = args.sparsity_ratio

    # 加载模型
    if args.init_model_path is not None:
        print(f'model path = {args.init_model_path}')
        model = Model(device)
        model.load_state_dict(torch.load(args.init_model_path))
    else:
        model = Model(device)
    print(f'model has been loaded')

    # 导入对应的datautil
    if dataset_name == 'movielens':
        import utils.movielens_data_util as data_util
    elif dataset_name == 'amazon':
        import utils.amazon_data_util as data_util

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = getOptim(model, optim='Adam', lr=learning_rate, l2=0)
    print(f'criterion and optimizer had been loaded')

    raw_data = data_util.get_raw_data()
    user_with_train = data_util.get_user_with_train()
    test_users = data_util.get_user_with_train_and_test()
    print(f'user_with_train has been loaded {user_with_train}')

    # 训练和测试
    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0.
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

        average_loss = 0.
        for index, user in enumerate(user_with_train):
            user_loss = 0.
            print(f'user{user} {index}/{len(user_with_train)}')
            train_data = data_util.get_user_train_data(raw_data, user)
            train_loader = train_util.get_data_loader(train_data, dataset_name, len(train_data), True)

            for batch_user, batch_item, batch_label in train_loader:
                batch_user = batch_user.to(device)
                batch_item = batch_item.to(device)
                batch_label = batch_label.unsqueeze(1).to(device)  # 形状: (batch_size, 1)
                # 前向传播
                mask, preds = model(batch_user, batch_item)

                # 计算损失
                loss1 = torch.mean(criterion(preds, batch_label) * mask)
                loss2 = alpha * kl_loss(mask, torch.ones_like((mask)))
                loss3 = beta * torch.abs((mask.mean() - sparsity_ratio))
                loss = loss1 + loss2 + loss3
                user_loss += loss
                print(f'mask = {mask.flatten()}, sparsity_ratio = {mask.mean()}')

                average_loss += total_loss

                # 反向传播和优化
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()

                loss.backward()
                optimizer[0].step()
                optimizer[1].step()
            user_loss /= len(train_loader)
            average_loss += user_loss
        
        # 测试
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            global_mask_ratio = 0.
            for index, user in enumerate(test_users):
                user_mask_ratio = 0.
                test_data = data_util.get_user_test_data(raw_data, user)
                test_loader = train_util.get_data_loader(test_data, dataset_name, batch_size, False)
                for index, (batch_user, batch_item, batch_label) in enumerate(test_loader):
                    batch_user = batch_user.to(device)
                    batch_item = batch_item.to(device)
                    batch_label = batch_label.to(device).unsqueeze(1)
                    
                    mask, preds = model(batch_user, batch_item)
                    all_labels.extend(batch_label.cpu().detach().numpy())
                    all_preds.extend(preds.cpu().detach().numpy())
                    user_mask_ratio += mask.mean()
                user_mask_ratio /= len(test_loader)
                global_mask_ratio += user_mask_ratio
            global_mask_ratio += user_mask_ratio

        auc = cal_auc(all_labels, all_preds)

        print(f'epoch{epoch}/{epochs} train loss = {average_loss/len(user_with_train)} auc = {auc}, mask_ratio = {global_mask_ratio / len(test_users)}')
        # 保存模型，每个epoch保存一次
        model_save_path = os.path.join(path_util.get_model_fp_by_name('MaskNet'), f'user_level_model_{dataset_name}_{sparsity_ratio}_epoch{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'model for epoch {epoch} has been saved to {model_save_path}')

    print('train finish')

if __name__ == '__main__':
    main()
