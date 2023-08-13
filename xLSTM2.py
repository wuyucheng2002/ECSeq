import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, pack_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.metrics import precision_recall_curve
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from time import time
from utils import *
from torch_geometric import seed_everything
import json
from transformer import *


class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num, batch_first=True)
        self.fc1 = nn.Linear(hidden_size + input_size, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        batch, num_node, seq_len = x.shape
        x = x.contiguous().view(batch * num_node, seq_len, 1)
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        y = torch.cat([output, x[:, -1, :]], dim=1)
        y = self.fc1(y).relu()
        y = self.fc2(y)
        return output.contiguous().view(batch, num_node, -1), y.view(batch, num_node)
    
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch, num_node, seq_len = x.shape
        x = x.contiguous().view(batch * num_node, seq_len, 1)
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        y = self.fc(output)
        return output.contiguous().view(batch, num_node, -1), y.view(batch, num_node)


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num):
        super().__init__()
        self.transformer = vanilla_transformer_encoder(input_dim=input_size, hidden_dim=32, d_model=32,  MHD_num_head=4, d_ff=2*32, output_dim=1).to(device)
    
    def forward(self, x):
        batch, num_node, seq_len = x.shape
        x = x.contiguous().view(batch * num_node, seq_len, 1)
        lens = torch.ones(batch * num_node) * seq_len
        output, y = self.transformer(x, lens, 1)
        # output = output[:, -1, :]
        # y = self.fc(output)
        return output.contiguous().view(batch, num_node, -1), y.view(batch, num_node)
    

def train(model, optimizer, epoch, loss_func, desc):
    model.train()
    idx = np.random.permutation(range(train_x.shape[0]))
    batch_num = int(np.ceil(train_x.shape[0] / batch_size))
    with tqdm(range(1, batch_num + 1), desc=desc) as loop:
        for i in loop:
            inputs = train_x[idx[(i-1)*batch_size: i*batch_size]]
            labels = train_y[idx[(i-1)*batch_size: i*batch_size]]
            model.zero_grad()
            _, prob = model(inputs)
            loss = loss_func(prob, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(T=t, epoch=epoch, loss=loss.item())


@torch.no_grad()
def eval(model, eval_x, eval_y , desc):
    model.eval()
    prob_list = []
    batch_num = int(np.ceil(eval_x.shape[0] / batch_size))
    with tqdm(range(1, batch_num + 1), desc=desc) as loop:
        for i in loop:
            inputs = eval_x[(i-1)*batch_size: i*batch_size]
            _, prob = model(inputs)
            prob_list.append(prob.cpu().numpy())

    prob_array = np.concatenate(prob_list, axis=0)
    return sttpmetric.get_rmse_smape(eval_y.cpu().numpy(), prob_array)


def run():
    if args.backbone == 'lstm':
        model = LSTM(1, hidden_size, layer_num).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
    elif args.backbone == 'xlstm':
        model = xLSTM(1, hidden_size, layer_num).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
    elif args.backbone == 'transformer':
        model = Transformer(1, hidden_size, layer_num).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    else:
        raise NotImplementedError
    
    loss_func = nn.MSELoss()

    # rmse_vals, rmse_tests = [], []
    # for epoch in range(1, 201):
    #     train(model, optimizer, epoch, loss_func, 'Train')
    #     if epoch % 1 == 0:
    #         rmse_val, smape_val = eval(model, val_x, val_y, 'Val')
    #         rmse_test, smape_test = eval(model, test_x, test_y, 'Test')
    #         rmse_vals.append(rmse_val)
    #         rmse_tests.append(rmse_test)

    #         print(f'T: {t}, time: {time() - t0}, Epoch: {epoch}, val rmse: {rmse_val}, test rmse: {rmse_test}')

    #         plt.figure()
    #         plt.plot(range(len(rmse_vals)), rmse_vals, label='rmse_val')
    #         plt.plot(range(len(rmse_vals)), rmse_tests, label='rmse_test')
    #         plt.legend()
    #         plt.savefig(f'fig/{args.backbone}_{args.dataset}_{t}.jpg')
    #         plt.close()

    # model_path = f'model/{args.backbone}_{args.dataset}_{t}.pkl'
    # torch.save(model.state_dict(), model_path)
    # print('Save succsessfully.')

    # Transformer
    rmse_vals, rmse_tests = [], []
    best_rmse_test = 100
    step = 0
    for epoch in range(1, 201):
        train(model, optimizer, epoch, loss_func, 'Train')
        rmse_val, smape_val = eval(model, val_x, val_y, 'Val')
        rmse_test, smape_test = eval(model, test_x, test_y, 'Test')
        rmse_vals.append(rmse_val)
        rmse_tests.append(rmse_test)

        if rmse_val < best_rmse_test:
            best_rmse_test = rmse_val
            model_path = f'model/{args.backbone}_{args.dataset}_{t}.pkl'
            torch.save(model.state_dict(), model_path)
            print('Save succsessfully.')
            step = 0
        else:
            step += 1

        if step == 10:
            break

        print(f'T: {t}, time: {time() - t0}, Epoch: {epoch}, val rmse: {rmse_val}, test rmse: {rmse_test}')

        plt.figure()
        plt.plot(range(len(rmse_vals)), rmse_vals, label='rmse_val')
        plt.plot(range(len(rmse_vals)), rmse_tests, label='rmse_test')
        plt.legend()
        plt.savefig(f'fig/{args.backbone}_{args.dataset}_{t}.jpg')
        plt.close()


    return epoch, rmse_test, smape_test


if __name__ == '__main__':
    seed_everything(42)
    hidden_size = 64
    layer_num = 1
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='pems_bay')
    parser.add_argument("--backbone", type=str, default='transformer')
    args = parser.parse_args()

    with open('params.json', 'r', encoding='utf-8') as f:
        params = json.load(f)
        scalar_min = params[args.dataset]["scalar_min"]
        scalar_max = params[args.dataset]["scalar_max"]
    
    sttpmetric = STTPMetric(scalar_min, scalar_max)

    dict_data = np.load(f'data/{args.dataset}.npy', allow_pickle=True).item()
    train_x, train_y = torch.tensor(dict_data['train']['X'], dtype=torch.float).to(device), torch.tensor(dict_data['train']['y'], dtype=torch.float).to(device)
    val_x, val_y = torch.tensor(dict_data['val']['X'], dtype=torch.float).to(device), torch.tensor(dict_data['val']['y'], dtype=torch.float).to(device)
    test_x, test_y = torch.tensor(dict_data['test']['X'], dtype=torch.float).to(device), torch.tensor(dict_data['test']['y'], dtype=torch.float).to(device)

    results = []
    t0 = time()
    for t in range(1, 6):
        print('times', t)
        res = run()
        results.append(res)
        with open('results.txt', 'a') as f:
            f.write(f'{args.backbone} final, time: {time() - t0}, {args.dataset}, Epoch: {res[0]}, test rmse: {res[1]}, test smape: {res[2]}\n')
            f.flush()
    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    print(f'time: {time() - t0}, Epoch: {mean[0]}, test rmse: {mean[1]}±{std[1]}, test smape: {mean[2]}±{std[2]}')
    with open('results.txt', 'a') as f:
        f.write(f'{args.backbone} final avg, time: {time() - t0}, {args.dataset}, Epoch: {mean[0]}, test rmse: {mean[1]:.4f}±{std[1]:.4f}, test smape: {mean[2]:.4f}±{std[2]:.4f}\n')
        f.flush()