import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Bird_Model import *

def train_model(data, label, model, lr, batch_size, epoch, save_model_path):
    net = model
    print(net)
    # 使用显卡，CPU则net.cpu()
    net = net.cuda()
    data = torch.Tensor(data)
    label = torch.Tensor(label)

    # 训练集和测试集7：3
    train_data, test_data, train_label, test_label = train_test_split(data,
                                                                      label,
                                                                      test_size=0.3,
                                                                      random_state=1)

    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)

    train_dataset = Data.TensorDataset(train_data, train_label)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_dataset = Data.TensorDataset(test_data, test_label)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # 优化器，Adam函数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # 动态学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.3,
        patience=2,
        cooldown=1,
    )

    best_loss = 50
    for cur_epoch in tqdm(range(epoch), position=0):
        print('\nEpoch:', cur_epoch + 1, "\n")
        net.train()
        train_loss = 0

        for step, (batch_data, batch_label) in tqdm(enumerate(train_loader), position=0):
            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()
            batch_output_dict = net(batch_data)
            step_loss = F.binary_cross_entropy(batch_output_dict['clipwise_output'], batch_label)
            train_loss += step_loss.item()
            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()

            batch_data = batch_data.cpu()
            batch_label = batch_label.cpu()

        test_loss = 0
        accuracy = []
        net.eval()
        with torch.no_grad():
            for step, (batch_data, batch_label) in tqdm(enumerate(test_loader), position=0):
                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()
                batch_output_dict = net(batch_data)
                step_loss = F.binary_cross_entropy(batch_output_dict['clipwise_output'], batch_label)
                test_loss += step_loss.item()
                _, pred_y = torch.max(batch_output_dict['clipwise_output'], 1)
                _, test_y = torch.max(batch_label, 1)
                # print(pred_y, test_y)
                accuracy.append(torch.sum(pred_y == test_y).item() / len(test_y))

                batch_data = batch_data.cpu()
                batch_label = batch_label.cpu()

        scheduler.step(test_loss)
        print('Epoch:', cur_epoch + 1,
              '| train loss:%.4f' % train_loss,
              '| test loss:%.4f' % test_loss,
              '| accuracy:%.4f' % np.mean(accuracy))

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(net.state_dict(), save_model_path + "\\" + "best_loss.pkl")

    torch.save(net.state_dict(), save_model_path + "\\" + "last_epoch.pkl")


def train(dataset_path, save_model_path, lr=0.001, batch_size=128, epoch=30):
    data_path = dataset_path + "/total_data.npy"
    label_path = dataset_path + "/total_label.npy"

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    total_data = np.load(data_path)
    total_label = np.load(label_path)

    lr = 0.001
    batch_size = 128
    epoch = 30

    df = pd.read_excel(dataset_path + "/class_label.xlsx")
    class_labels = df.iloc[:, 1].values
    class_num = len(class_labels)

    Model = eval("Cnn14_sed")
    model = Model(sample_rate=32000, window_size=1024,
                  hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                  classes_num=class_num)

    train_model(data=total_data, label=total_label, model=model, lr=lr,
                batch_size=batch_size, epoch=epoch,
                save_model_path=save_model_path)


def caesar():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_model', action='store_true', help="enable train_model")
    parser.add_argument('--dataset_path', default="")
    parser.add_argument('--save_model_path', default="")
    parser.add_argument('--lr', default=0.001, help="learning rate")
    parser.add_argument('--batch_size', default=128, help="batch samples size")
    parser.add_argument('--epoch', default=30, help="train epochs")
    args = parser.parse_args()

    if args.train_model:
        print("train model")
        train(args.dataset_path, args.save_model_path, args.lr, args.batch_size, args.epoch)
    else:
        print("Nothing have done")


if __name__ == '__main__':
    caesar()
