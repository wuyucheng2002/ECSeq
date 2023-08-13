import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, pack_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from time import time
from utils import *
from transformer import vanilla_transformer_encoder


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, x_length, end):
        x, _ = self.lstm(x)
        output, _ = pad_packed_sequence(x)
        target_x = []
        for j, length in enumerate(x_length):
            target_x.append(output[length - 1, j, :])
        target_x = torch.stack(target_x, dim=0)
        return target_x, self.fc(target_x).softmax(dim=1)
    

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num)
        self.fc1 = nn.Linear(hidden_size + input_size, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x, x_length, end):
        x, _ = self.lstm(x)
        output, _ = pad_packed_sequence(x)
        target_x = []
        for j, length in enumerate(x_length):
            target_x.append(output[length - 1, j, :])
        target_x = torch.stack(target_x, dim=0)
        x = torch.cat([target_x, end], dim=1)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return target_x, x.softmax(dim=1)


def train(model, train_loader, optimizer, epoch, loss_func, desc):
    model.train()
    with tqdm(train_loader, desc=desc) as loop:
        for batch in loop:
            inputs, labels, lengths, ends, _ = batch
            model.zero_grad()
            _, prob = model(inputs, lengths, ends)
            loss = loss_func(prob.log(), labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(T=t, epoch=epoch, loss=loss.item())


@torch.no_grad()
def eval(model, eval_loader, desc):
    model.eval()
    label_list = []
    prob_list = []
    with tqdm(eval_loader, desc=desc) as loop:
        for batch in loop:
            inputs, labels, lengths, ends, _ = batch
            _, prob = model(inputs, lengths, ends)
            label_list.append(labels.cpu().numpy())
            prob_list.append(prob[:, 1].cpu().numpy())

    label_array = np.concatenate(label_list, axis=0)
    prob_array = np.concatenate(prob_list, axis=0)
    return get_auc_ap_rp(label_array, prob_array)


def run():
    if args.backbone == 'lstm':
        model = LSTM(input_size, hidden_size, layer_num).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
    elif args.backbone == 'xlstm':
        model = xLSTM(input_size, hidden_size, layer_num).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
    elif args.backbone == 'transformer':
        model = vanilla_transformer_encoder(input_dim=input_size, hidden_dim=32, d_model=32,  MHD_num_head=4, d_ff=2*32, output_dim=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    else:
        raise NotImplementedError
    
    loss_func = nn.NLLLoss()
    model_path = f'model/{args.backbone}_{args.dataset}_{t}.pkl'

    best_auc_val = 0
    auc_vals, auc_tests = [], []
    step = 0
    # for epoch in range(1, 31):  # 50 
    #     train(model, train_loader, optimizer, epoch, loss_func, 'Train')
    #     auc_val, _, _ = eval(model, val_loader, 'Val')
    #     auc_test, _, _ = eval(model, test_loader, 'Test')
    #     auc_vals.append(auc_val)
    #     auc_tests.append(auc_test)

    #     if auc_val > best_auc_val:
    #         best_auc_val = auc_val
    #         step = 0
    #         torch.save(model.state_dict(), model_path)
    #         print('Save succsessfully.')
    #     else:
    #         step += 1
        
    #     if step == 5:  # 10 
    #         break

    #     print(f'T: {t}, time: {time() - t0}, Epoch: {epoch}, val AUC: {auc_val}, test AUC: {auc_test}')

    #     plt.figure()
    #     plt.plot(range(len(auc_vals)), auc_vals, label='auc_val')
    #     plt.plot(range(len(auc_vals)), auc_tests, label='auc_test')
    #     plt.legend()
    #     plt.savefig(f'fig/{args.backbone}_{args.dataset}_{t}.jpg')
    #     plt.close()

    model.load_state_dict(torch.load(model_path))
    # t0 = time()
    auc, f1, rp = eval(model, test_loader, 'Test')
    # t1 = time()
    # print(t1-t0)
    return len(auc_vals) - step, auc, f1, rp


if __name__ == '__main__':
    hidden_size = 256
    layer_num = 1
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='HK')
    parser.add_argument("--backbone", type=str, default='transformer')
    args = parser.parse_args()

    if args.backbone == 'transformer':
        def collate_fn(batch):
            inputs, labels, lengths, ends, idxs = zip(*batch)
            inputs_pad = pad_sequence(inputs, batch_first=True)
            return inputs_pad.float().to(device), torch.LongTensor(labels).to(device), torch.LongTensor(lengths), \
                torch.stack(ends, dim=0).float().to(device), torch.LongTensor(idxs).to(device)
        train_loader, train_loader2, val_loader, test_loader, input_size = get_dataset(args, batch_size, device, collate_fn=collate_fn)
    else:
        train_loader, train_loader2, val_loader, test_loader, input_size = get_dataset(args, batch_size, device)

    results = []
    t0 = time()
    for t in range(1, 6):
        print('times', t)
        res = run()
        results.append(res)
        with open('results.txt', 'a') as f:
            f.write(f'{args.backbone} final, time: {time() - t0}, {args.dataset}, Epoch: {res[0]}, test AUC: {res[1]}, test F1: {res[2]}, test RP: {res[3]}\n')
            f.flush()
    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    print(f'time: {time() - t0}, Epoch: {mean[0]}, {mean[1]:.4f}±{std[1]:.4f}, {mean[2]:.4f}±{std[2]:.4f}, {mean[3]:.4f}±{std[3]:.4f}')
    with open('results.txt', 'a') as f:
        f.write(f'{args.backbone} final avg, time: {time() - t0}, {args.dataset}, Epoch: {mean[0]}, {mean[1]:.4f}±{std[1]:.4f}, {mean[2]:.4f}±{std[2]:.4f}, {mean[3]:.4f}±{std[3]:.4f}\n')
        f.flush()