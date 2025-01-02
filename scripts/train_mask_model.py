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
from scripts.movielens_dataset import MovielensDataset
from scripts.amazon_dataset import AmazonDataset
from scripts.metric import *
from model.MaskNet.model import Model

def config_args():
    parser = argparse.ArgumentParser('train global model')
    parser.add_argument('-epochs', type=int, default=10, help='模型在每个训练集上训练的epoch')
    parser.add_argument('-device', type=str, default='cuda:2', help='训练模型的设备')
    parser.add_argument('-batch_size', type=int, default=64, help='batch大小')
    parser.add_argument('-dataset', type=str, default='movielens', choices=['movielens', 'amazon'], help='训练集名称')
    parser.add_argument('-lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('-alpha', type=float, default=0.5, help='KL散度损失权重')
    parser.add_argument('-beta', type=float, default=5, help='掩码稀疏度损失权重')
    parser.add_argument('-sparsity_ratio', type=float, default=0.9, help='掩码稀疏度期望值')


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
    model = Model(device)
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
    optimizer = getOptim(model, optim='Adam', lr=learning_rate, l2=0)
    print(f'criterion and optimizer had been loaded')

    # 训练和测试
    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0.
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
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
            total_loss += loss

            # 反向传播和优化
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()

            loss.backward()
            optimizer[0].step()
            optimizer[1].step()
        
        # 测试
        model.eval()
        all_labels = []
        all_preds = []
        mask_ratio = 0.
        with torch.no_grad():
            for index, (batch_user, batch_item, batch_label) in enumerate(test_loader):
                batch_user = batch_user.to(device)
                batch_item = batch_item.to(device)
                batch_label = batch_label.to(device).unsqueeze(1)
                
                mask, preds = model(batch_user, batch_item)
                all_labels.extend(batch_label.cpu().detach().numpy())
                all_preds.extend(preds.cpu().detach().numpy())
                mask_ratio += mask.mean()

        auc = cal_auc(all_labels, all_preds)

        print(f'epoch{epoch}/{epochs} auc = {auc}, mask_ratio = {mask_ratio / len(test_loader)}')
        # 保存模型，每个epoch保存一次
        model_save_path = os.path.join(path_util.get_model_fp_by_name('MaskNet'), f'model_epoch{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'model for epoch {epoch} has been saved to {model_save_path}')

    print('train finish')

if __name__ == '__main__':
    main()
    # sparsity_ratio = 0.9
    # epoch0/10 auc = 0.6643327061019099, mask_ratio = 0.902767539024353
    # epoch1/10 auc = 0.6719419868584395, mask_ratio = 0.912447452545166
    # epoch2/10 auc = 0.6770599806794032, mask_ratio = 0.9080781936645508
    # epoch3/10 auc = 0.6789260279548517, mask_ratio = 0.9103572964668274 selected
    # epoch4/10 auc = 0.6777593047147239, mask_ratio = 0.907768189907074
    # epoch5/10 auc = 0.6765854133326147, mask_ratio = 0.9129320383071899
    # epoch6/10 auc = 0.6741654454952448, mask_ratio = 0.9097239375114441
    # epoch7/10 auc = 0.6735290292114429, mask_ratio = 0.9186652898788452
    # epoch8/10 auc = 0.6710736607713864, mask_ratio = 0.9053058624267578
    # epoch9/10 auc = 0.6685115682221288, mask_ratio = 0.9077956080436707

    # sparsity_ratio = 0.95
    # epoch0/10 auc = 0.6642176531916775, mask_ratio = 0.9493509531021118
    # epoch1/10 auc = 0.671986056989919, mask_ratio = 0.954486608505249
    # epoch2/10 auc = 0.6766938617085947, mask_ratio = 0.9438290596008301
    # epoch3/10 auc = 0.6787642742542913, mask_ratio = 0.9518014788627625 selected
    # epoch4/10 auc = 0.6775675354901178, mask_ratio = 0.9533911943435669
    # epoch5/10 auc = 0.6763543420021186, mask_ratio = 0.955257773399353

    # sparsity_ratio = 0.85
    