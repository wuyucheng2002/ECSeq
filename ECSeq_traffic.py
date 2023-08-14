import torch
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from time import time
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv
from torch_geometric import seed_everything
import json
import os
from utils import *
from utils_compress import *
from utils_transformer import *


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
        return output.contiguous().view(batch, num_node, -1), y.view(batch, num_node)


class GNN(torch.nn.Module):
    def __init__(self, hidden_lstm, hidden_gnn, out_channels=1, backbone='GraphSAGE', heads=8):
        super().__init__()
        self.backbone = backbone
        self.heads = heads if self.backbone == 'GAT' else 1
        if self.backbone == 'GCN':
            self.conv = GCNConv(hidden_lstm, hidden_gnn)
            self.fc = torch.nn.Linear(hidden_gnn, out_channels)
        elif self.backbone == 'GraphSAGE':
            self.conv = GraphSAGE(hidden_lstm, hidden_gnn, num_layers=1)
            self.fc = torch.nn.Linear(hidden_gnn, out_channels)
        elif self.backbone == 'GraphSAGE_max':
            self.conv = GraphSAGE(hidden_lstm, hidden_gnn, num_layers=1, aggr='max')
            self.fc = torch.nn.Linear(hidden_gnn, out_channels)
        elif self.backbone == 'GAT':
            self.conv = GATConv(hidden_lstm, hidden_gnn, heads, dropout=0.6)
            self.fc = torch.nn.Linear(hidden_gnn * heads, out_channels)

    def forward(self, input, edge_index, edge_weight=None):
        seq_len, node_num, hidden_size = input.shape
        xs = torch.empty((0, node_num, hidden_size_gnn * self.heads)).to(device)
        ys = torch.empty((0, node_num)).to(device)
        for i in range(seq_len):
            x = input[i]
            if self.backbone == 'GCN':
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv(x, edge_index, edge_weight)
            elif self.backbone in ['GraphSAGE', 'GraphSAGE_max']:
                x = self.conv(x, edge_index)
            elif self.backbone == 'GAT':
                x = F.dropout(x, p=0.6, training=self.training)
                x = self.conv(x, edge_index)
            y = self.fc(x).squeeze()
            xs = torch.cat([xs, x.unsqueeze(0)], dim=0)
            ys = torch.cat([ys, y.unsqueeze(0)], dim=0)
        return xs, ys


def train_lstm(lstm_model):
    if args.seq_backbone == 'lstm':
        lstm_opt = optim.Adam(lstm_model.parameters(), lr=0.00001)
    elif args.seq_backbone == 'transformer':
        lstm_opt = optim.Adam(lstm_model.parameters(), lr=0.0001)
    else:
        raise NotImplementedError

    rmse_vals, rmse_tests = [], []

    for epoch in range(1, 100 + 1):
        lstm_model.train()
        idx = np.random.permutation(range(train_x.shape[0]))
        batch_num = int(np.ceil(train_x.shape[0] / batch_size))

        for i in range(1, batch_num + 1):
            input = train_x[idx[(i-1)*batch_size: i*batch_size]]
            label = train_y[idx[(i-1)*batch_size: i*batch_size]]
            lstm_model.zero_grad()
            _, out = lstm_model(input)
            loss = loss_mse(out, label)
            loss.backward()
            lstm_opt.step()

        rmse_val, _ = eval_lstm(lstm_model, val_x, val_y , 'Val')
        rmse_test, _ = eval_lstm(lstm_model, test_x, test_y , 'Test')
        rmse_vals.append(rmse_val)
        rmse_tests.append(rmse_test)

        print(f'T: {t}, time: {time() - t0}, Epoch: {epoch}, val RMSE: {rmse_val}, test RMSE: {rmse_test}')

    plt.figure()
    plt.plot(range(len(rmse_vals)), rmse_vals, label='rmse_val')
    plt.plot(range(len(rmse_tests)), rmse_tests, label='rmse_test')
    plt.legend()
    plt.savefig(f'fig/{args.seq_backbone}_{args.dataset}_{t}.jpg')
    plt.close()

    torch.save(lstm_model.state_dict(), f'model/{args.seq_backbone}_{args.dataset}_{t}.pkl')
    print('Save succsessfully.')


def train_gnn(lstm_model, gnn_model, edge_index, embeds, embeds_val, embeds_test, phi=None):
    lstm_model.eval()
    gnn_model.train()
    gnn_opt = torch.optim.Adam(gnn_model.parameters(), lr=0.0001)

    if phi is not None:
        X_coms = torch.matmul(phi.T, embeds)
    
    best_rmse_val1 = 100
    rmse_vals1, rmse_tests1 = [], []
    step = 0
    for epoch in range(1, 100 + 1):
        idx = np.random.permutation(range(train_x.shape[0]))
        batch_num = int(np.ceil(embeds.shape[0] / batch_size))
        with tqdm(range(1, batch_num + 1), desc=f'T{t} Train GNN') as loop:
            for i in loop:
                gnn_model.zero_grad()
                input = embeds[idx[(i-1)*batch_size: i*batch_size]]
                label = train_y[idx[(i-1)*batch_size: i*batch_size]]

                if phi is None:
                    _, prob = gnn_model(input, edge_index)
                else:
                    X_com = X_coms[idx[(i-1)*batch_size: i*batch_size]]
                    _, prob = gnn_model(torch.cat([X_com, input], dim=1), edge_index)
                    prob = prob[:, n_cluster:]

                loss = loss_mse(prob, label)
                loss.backward()
                gnn_opt.step()
                loop.set_postfix(epoch=epoch, loss=loss.item())

        rmse_val1, _ = eval_gnn(lstm_model, gnn_model, embeds_val, val_y, edge_index, phi)
        rmse_test1, _ = eval_gnn(lstm_model, gnn_model, embeds_test, test_y, edge_index, phi)
        print(f'rmse_val1: {rmse_val1}, rmse_test1: {rmse_test1}')

        rmse_vals1.append(rmse_val1)
        rmse_tests1.append(rmse_test1)

        if rmse_val1 < best_rmse_val1:
            best_rmse_val1 = rmse_val1
            step = 0
            torch.save(gnn_model.state_dict(), f'model/{args.method}_{args.dataset}_{t}_gnn.pkl')
            print('GNN save succsessfully.')
        else:
            step += 1

        if step == 10:
            break
        
    plt.figure()
    plt.plot(range(len(rmse_vals1)), rmse_vals1, label='gnn_rmse_val')
    plt.plot(range(len(rmse_vals1)), rmse_tests1, label='gnn_rmse_test')
    plt.legend()
    plt.savefig(f'fig/{args.method}_{args.dataset}_{t}_gnn.jpg')


@torch.no_grad()
def eval_lstm(model, eval_x, eval_y , desc):
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


@torch.no_grad()
def eval_gnn(lstm_model, gnn_model, embeds_eval, eval_y, edge_index, phi=None):
    lstm_model.eval()
    gnn_model.eval()

    if phi is None:
        _, out2 = gnn_model(embeds_eval, edge_index)
    else:
        _, out2 = gnn_model(torch.cat([torch.matmul(phi.T, embeds_eval), embeds_eval], dim=1), edge_index)
        out2 = out2[:, n_cluster:]

    return sttpmetric.get_rmse_smape(eval_y.cpu().numpy(), out2.cpu().numpy())


def run():
    # sequence embedding extractor
    if args.seq_backbone == 'lstm':
        lstm_model = LSTM(1, hidden_size, layer_num).to(device)
    elif args.seq_backbone == 'transformer':
        lstm_model = Transformer(1, hidden_size, layer_num).to(device)
    else:
        raise NotImplementedError
    
    if not os.path.exists(f'model/{args.seq_backbone}_{args.dataset}_{t}.pkl'):
        train_lstm(lstm_model)

    lstm_model.load_state_dict(torch.load(f'model/{args.seq_backbone}_{args.dataset}_{t}.pkl'))
    print(f'{args.seq_backbone} load succsessfully.')
    rmse0, smape0 = eval_lstm(lstm_model, test_x, test_y, 'Test')

    with torch.no_grad():
        lstm_model.eval()
        embeds = torch.empty((0, num_node, hidden_size)).to(device)
        batch_num = int(np.ceil(train_x.shape[0] / batch_size))
        for i in range(1, batch_num + 1):
            input = train_x[(i-1)*batch_size: i*batch_size]
            embed, _ = lstm_model(input)
            embeds = torch.cat([embeds, embed], dim=0)

        embeds_val = torch.empty((0, num_node, hidden_size)).to(device)
        batch_num = int(np.ceil(val_x.shape[0] / batch_size))
        for i in range(1, batch_num + 1):
            input = val_x[(i-1)*batch_size: i*batch_size]
            embed, _ = lstm_model(input)
            embeds_val = torch.cat([embeds_val, embed], dim=0)

        embeds_test = torch.empty((0, num_node, hidden_size)).to(device)
        batch_num = int(np.ceil(test_x.shape[0] / batch_size))
        for i in range(1, batch_num + 1):
            input = test_x[(i-1)*batch_size: i*batch_size]
            embed, _ = lstm_model(input)
            embeds_test = torch.cat([embeds_test, embed], dim=0)

    # graph mining
    gnn_model = GNN(hidden_size, hidden_size_gnn, backbone=args.gnn_backbone).to(device)
    if not os.path.exists(f'model/{args.method}_{args.dataset}_{t}_gnn.pkl'):
        train_gnn(lstm_model, gnn_model, edge_index, embeds, embeds_val, embeds_test, phi)
    gnn_model.load_state_dict(torch.load(f'model/{args.method}_{args.dataset}_{t}_gnn.pkl'))
    print('GNN load succsessfully.')
    rmse1, smape1 = eval_gnn(lstm_model, gnn_model, embeds_test, test_y, edge_index, phi)

    return rmse0, smape0, rmse1, smape1


if __name__ == '__main__':
    seed_everything(42)
    hidden_size = 64
    hidden_size_gnn = 16
    layer_num = 1
    batch_size = 16
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='bike_nyc')  # bike_nyc, pems_bay
    parser.add_argument("--seq_backbone", type=str, default='lstm')  # lstm, transformer
    parser.add_argument("--gnn_backbone", type=str, default='GraphSAGE')  # GraphSAGE, GraphSAGE_max, GCN, GAT
    parser.add_argument("--method", type=str, default='ECSeq')  # ECSeq, batchGNN
    parser.add_argument("--compress", type=str, default='kmeans')  # kmeans, AGC, Grain, Loukas
    parser.add_argument("--n_cluster", type=int, default=100)  # number of clusters
    args = parser.parse_args()

    with open('params.json', 'r', encoding='utf-8') as f:
        params = json.load(f)
        cor_thres = params[args.dataset]["cor_thres"]
        scalar_min = params[args.dataset]["scalar_min"]
        scalar_max = params[args.dataset]["scalar_max"]
    n_cluster = args.n_cluster
       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(args)

    sttpmetric = STTPMetric(scalar_min, scalar_max)

    dict_data = np.load(f'data/{args.dataset}.npy', allow_pickle=True).item()
    train_x, train_y = torch.tensor(dict_data['train']['X'], dtype=torch.float).to(device), torch.tensor(dict_data['train']['y'], dtype=torch.float).to(device)
    val_x, val_y = torch.tensor(dict_data['val']['X'], dtype=torch.float).to(device), torch.tensor(dict_data['val']['y'], dtype=torch.float).to(device)
    test_x, test_y = torch.tensor(dict_data['test']['X'], dtype=torch.float).to(device), torch.tensor(dict_data['test']['y'], dtype=torch.float).to(device)
    
    num_node = train_x.shape[1]
    edge_index = torch.tensor(dict_data['edge_index_corr']).long().to(device)
    print(f'ori, nodes={num_node}, deg={edge_index.shape[1]/num_node}')

    loss_mse = torch.nn.MSELoss()

    # compress
    if args.method == 'ECSeq':
        X_cor = train_x[-24*30:, :, -1].T
        
        if args.compress == 'kmeans':
            phi = kmeans_no(X_cor.cpu().numpy(), n_cluster)
            phi = phi / phi.sum(axis=0)
        elif args.compress == 'AGC':
            agc = AGC(X_cor.cpu().numpy(), edge_index, n_cluster=n_cluster) #, bestpower=5
            phi = agc.get_phi()
            phi = phi / phi.sum(axis=0)
        elif args.compress == 'Grain':
            grain = Grain(X_cor, edge_index, num_coreset=n_cluster)
            phi, n_cluster = grain.get_phi()
        elif args.compress == 'Loukas':
            loukas = Loukas(torch.tensor(X_cor, dtype=torch.float), edge_index, n_cluster)
            phi = loukas.get_phi()
            n_cluster = phi.shape[1]
            phi = phi / phi.sum(axis=0)
        else:
            raise NotImplementedError

        phi = torch.tensor(phi, device=device, dtype=torch.float)
        com_cor = torch.mm(phi.T, X_cor)
        edge_index = correlation_edge(com_cor, X_cor, cor_thres).to(device)
        edge_index[1] = edge_index[1] + n_cluster
    
        print(f'nodes={n_cluster+num_node}, deg={edge_index.shape[1]/num_node}')
        print(phi.shape)
    
    else:
        phi = None
        print(f'nodes={train_x.shape[1]}, deg={edge_index.shape[1]/train_x.shape[1]}')

    results = []
    t0 = time()
    for t in range(1, 6):
        print('times', t)
        res = run()
        results.append(res)
        with open('results.txt', 'a') as f:
            f.write(f'{args.method} final, time: {time() - t0}, {args.dataset}, '
                    f'[lstm] rmse={res[0]}, smape={res[1]}, '
                    f'[gnn] rmse={res[2]}, smape={res[3]}\n')
            f.flush()
    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    with open('results.txt', 'a') as f:
        f.write(f'{args.method} final avg, time: {time() - t0}, {args.dataset}, '
                f'[lstm] {mean[0]:.4f}±{std[0]:.4f}, {mean[1]:.4f}±{std[1]:.4f}, '
                f'[gnn] {mean[2]:.4f}±{std[2]:.4f}, {mean[3]:.4f}±{std[3]:.4f}\n')
        f.flush()
    
    print(f'{args.method} final avg, time: {time() - t0}, {args.dataset}, '
          f'[lstm] {mean[0]:.4f}±{std[0]:.4f}, {mean[1]:.4f}±{std[1]:.4f}, '
          f'[gnn] {mean[2]:.4f}±{std[2]:.4f}, {mean[3]:.4f}±{std[3]:.4f}')





