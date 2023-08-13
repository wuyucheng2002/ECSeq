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


# class GNN(torch.nn.Module):
#     def __init__(self, hidden_lstm, hidden_gnn, out_channels=1):
#         super().__init__()
#         self.conv = GraphSAGE(hidden_lstm, hidden_gnn, num_layers=1)
#         self.fc = torch.nn.Linear(hidden_gnn, out_channels)

#     def forward(self, input, edge_index, edge_weight=None):
#         seq_len, node_num, hidden_size = input.shape
#         xs = torch.empty((0, node_num, hidden_size_gnn)).to(device)
#         ys = torch.empty((0, node_num)).to(device)
#         for i in range(seq_len):
#             x = self.conv(input[i], edge_index)
#             y = self.fc(x).squeeze()
#             xs = torch.cat([xs, x.unsqueeze(0)], dim=0)
#             ys = torch.cat([ys, y.unsqueeze(0)], dim=0)
#         return xs, ys


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
            elif self.backbone == 'GraphSAGE':
                x = self.conv(x, edge_index)
            elif self.backbone == 'GAT':
                x = F.dropout(x, p=0.6, training=self.training)
                x = self.conv(x, edge_index)
            y = self.fc(x).squeeze()
            xs = torch.cat([xs, x.unsqueeze(0)], dim=0)
            ys = torch.cat([ys, y.unsqueeze(0)], dim=0)
        return xs, ys


class Ensemble(torch.nn.Module):
    def __init__(self, input_size, out_size=1, hidden_size=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, out_size)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=2)
        seq_len, node_num, hidden_size = x.shape
        x = x.contiguous().view(node_num * seq_len, hidden_size)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        x = x.view(seq_len, node_num, 1).squeeze(dim=2)
        return x.squeeze()


def train_lstm(lstm_model, t):
    lstm_opt = optim.Adam(lstm_model.parameters(), lr=0.00001)
    for epoch in range(1, lstm_epo + 1):
        lstm_model.train()
        probs = np.empty((0, num_node))
        labels = np.empty((0, num_node))
        idx = np.random.permutation(range(train_x.shape[0]))
        batch_num = int(np.ceil(train_x.shape[0] / batch_size))
        with tqdm(range(1, batch_num + 1), desc=f'T{t} Train LSTM {epoch}') as loop:
            for i in loop:
                input = train_x[idx[(i-1)*batch_size: i*batch_size]]
                label = train_y[idx[(i-1)*batch_size: i*batch_size]]
                lstm_model.zero_grad()
                _, out = lstm_model(input)
                loss = loss_mse(out, label)
                loss.backward()
                lstm_opt.step()

                prob = out.detach()
                probs = np.concatenate([probs, prob.cpu().numpy()], axis=0)
                labels = np.concatenate([labels, label.cpu().numpy()], axis=0)

            mse, smape = sttpmetric.get_rmse_smape(labels, probs)
            print(f'[lstm] mse={mse}, smape={smape}')

    torch.save(lstm_model.state_dict(), f'model/lstm_{args.dataset}_{t}.pkl')
    print('LSTM save succsessfully.')


def train_gnn(lstm_model, gnn_model, X, Y, edge_index, phi, full):
    lstm_model.eval()
    gnn_model.train()
    gnn_opt = torch.optim.Adam(gnn_model.parameters(), lr=0.0001)
    
    best_rmse_val1 = 100
    rmse_vals1, rmse_tests1 = [], []
    step = 0
    for epoch in range(1, gnn_epo + 1):
        idx = np.random.permutation(range(train_x.shape[0]))
        batch_num = int(np.ceil(X.shape[0] / batch_size))
        with tqdm(range(1, batch_num + 1), desc=f'T{t} Train GNN') as loop:
            for i in loop:
                gnn_model.zero_grad()
                input = X[idx[(i-1)*batch_size: i*batch_size]]
                label = Y[idx[(i-1)*batch_size: i*batch_size]]
                _, prob = gnn_model(input, edge_index)
                loss = loss_mse(prob, label)
                loss.backward()
                gnn_opt.step()
                loop.set_postfix(epoch=epoch, loss=loss.item())

        rmse_val1, _ = eval_gnn(lstm_model, gnn_model, val_x, val_y, edge_index, phi, full)
        rmse_test1, _ = eval_gnn(lstm_model, gnn_model, test_x, test_y, edge_index, phi, full)
        print(f'rmse_val1: {rmse_val1}, rmse_test1: {rmse_test1}')

        rmse_vals1.append(rmse_val1)
        rmse_tests1.append(rmse_test1)

        if rmse_val1 < best_rmse_val1:
            best_rmse_val1 = rmse_val1
            step = 0
            torch.save(gnn_model.state_dict(), f'model/{args.method}_{args.graph}_{args.dataset}_{t}_gnn.pkl')
            print('GNN save succsessfully.')
        else:
            step += 1

        if step == 10:
            break
        
    plt.figure()
    plt.plot(range(len(rmse_vals1)), rmse_vals1, label='gnn_rmse_val')
    plt.plot(range(len(rmse_vals1)), rmse_tests1, label='gnn_rmse_test')
    plt.legend()
    plt.savefig(f'fig/{args.method}_{args.graph}_{args.dataset}_{t}_gnn.jpg')

    return len(rmse_vals1) - step


def train_gnn2(lstm_model, gnn_model, X, Y, edge_index, phi, full, X_coms):
    lstm_model.eval()
    gnn_model.train()
    gnn_opt = torch.optim.Adam(gnn_model.parameters(), lr=0.0001)
    
    best_rmse_val1 = 100
    rmse_vals1, rmse_tests1 = [], []
    step = 0
    for epoch in range(1, gnn_epo + 1):
        idx = np.random.permutation(range(train_x.shape[0]))
        batch_num = int(np.ceil(X.shape[0] / batch_size))
        with tqdm(range(1, batch_num + 1), desc=f'T{t} Train GNN') as loop:
            for i in loop:
                gnn_model.zero_grad()
                input = X[idx[(i-1)*batch_size: i*batch_size]]
                label = Y[idx[(i-1)*batch_size: i*batch_size]]
                X_com = X_coms[idx[(i-1)*batch_size: i*batch_size]]

                # _, prob = gnn_model(input, edge_index)
                _, prob = gnn_model(torch.cat([X_com, input], dim=1), edge_index)
                prob = prob[:, n_cluster:]
                loss = loss_mse(prob, label)
                loss.backward()
                gnn_opt.step()
                loop.set_postfix(epoch=epoch, loss=loss.item())

        rmse_val1, _ = eval_gnn(lstm_model, gnn_model, val_x, val_y, edge_index, phi, full)
        rmse_test1, _ = eval_gnn(lstm_model, gnn_model, test_x, test_y, edge_index, phi, full)
        print(f'rmse_val1: {rmse_val1}, rmse_test1: {rmse_test1}')

        rmse_vals1.append(rmse_val1)
        rmse_tests1.append(rmse_test1)

        if rmse_val1 < best_rmse_val1:
            best_rmse_val1 = rmse_val1
            step = 0
            torch.save(gnn_model.state_dict(), f'model/{args.method}_{args.graph}_{args.dataset}_{t}_gnn.pkl')
            print('GNN save succsessfully.')
        else:
            step += 1

        if step == 10:
            break
        
    plt.figure()
    plt.plot(range(len(rmse_vals1)), rmse_vals1, label='gnn_rmse_val')
    plt.plot(range(len(rmse_vals1)), rmse_tests1, label='gnn_rmse_test')
    plt.legend()
    plt.savefig(f'fig/{args.method}_{args.graph}_{args.dataset}_{t}_gnn.jpg')

    return len(rmse_vals1) - step
   

def train_ensemble(lstm_model, gnn_model, ens_model, edge_index, phi, full):
    ens_opt = optim.Adam(ens_model.parameters(), lr=0.00005)
    ens_model.train()
    lstm_model.eval()
    gnn_model.eval()

    best_rmse_val2 = 100
    rmse_vals2, rmse_tests2 = [], []
    step = 0

    for epoch in range(1, ens_epo + 1):
        idx = np.random.permutation(range(train_x.shape[0]))
        batch_num = int(np.ceil(train_x.shape[0] / batch_size))
        with tqdm(range(1, batch_num + 1), desc=f'T{t} Train Ensemble') as loop:
            for i in loop:
                with torch.no_grad():
                    input = train_x[idx[(i-1)*batch_size: i*batch_size]]
                    label = train_y[idx[(i-1)*batch_size: i*batch_size]]
                    x_lstm, _ = lstm_model(input)

                    if full:
                        x_gnn, _ = gnn_model(x_lstm, edge_index)
                    else:
                        x_gnn, _ = gnn_model(torch.cat([torch.matmul(phi.T, x_lstm), x_lstm], dim=1), edge_index)
                        x_gnn = x_gnn[:, n_cluster:, :]
                
                out3 = ens_model(x_lstm, x_gnn)
                loss = loss_mse(out3, label)
                loss.backward()
                ens_opt.step()
                loop.set_postfix(epoch=epoch, loss=loss.item())

        _, _, (rmse_val2, _) = eval_all(lstm_model, gnn_model, ens_model, val_x, val_y, edge_index, phi, full)
        _, _, (rmse_test2, _) = eval_all(lstm_model, gnn_model, ens_model, test_x, test_y, edge_index, phi, full)
        print(f'rmse_val2: {rmse_val2}, rmse_test2: {rmse_test2}')

        rmse_vals2.append(rmse_val2)
        rmse_tests2.append(rmse_test2)

        if rmse_val2 < best_rmse_val2:
            best_rmse_val2 = rmse_val2
            step = 0
            torch.save(ens_model.state_dict(), f'model/{args.method}_{args.graph}_{args.dataset}_{t}_ens.pkl')
            print('ENS save succsessfully.')
        else:
            step += 1

        if step == 10:
            break
        
    plt.figure()
    plt.plot(range(len(rmse_vals2)), rmse_vals2, label='ens_rmse_val')
    plt.plot(range(len(rmse_vals2)), rmse_tests2, label='ens_rmse_test')
    plt.legend()
    plt.savefig(f'fig/{args.method}_{args.graph}_{args.dataset}_{t}_ens.jpg')

    return len(rmse_vals2) - step
            

@torch.no_grad()
def eval_gnn(lstm_model, gnn_model, eval_x, eval_y, edge_index, phi, full):
    lstm_model.eval()
    gnn_model.eval()

    prob2s = np.empty((0, num_node))
    batch_num = int(np.ceil(eval_x.shape[0] / batch_size))

    for i in range(1, batch_num + 1):
        input = eval_x[(i-1)*batch_size: i*batch_size]
        x_lstm, _ = lstm_model(input)

        if full:
            _, out2 = gnn_model(x_lstm, edge_index)
        else:
            _, out2 = gnn_model(torch.cat([torch.matmul(phi.T, x_lstm), x_lstm], dim=1), edge_index)
            out2 = out2[:, n_cluster:]

        prob2s = np.concatenate([prob2s, out2.cpu().numpy()], axis=0)

    labels = eval_y.cpu().numpy()
    return sttpmetric.get_rmse_smape(labels, prob2s)


@torch.no_grad()
def eval_all(lstm_model, gnn_model, ens_model, eval_x, eval_y, edge_index, phi, full):
    lstm_model.eval()
    gnn_model.eval()
    ens_model.eval()

    prob1s = np.empty((0, num_node))
    prob2s = np.empty((0, num_node))
    prob3s = np.empty((0, num_node))
    batch_num = int(np.ceil(eval_x.shape[0] / batch_size))

    for i in range(1, batch_num + 1):
        input = eval_x[(i-1)*batch_size: i*batch_size]
        x_lstm, out1 = lstm_model(input)

        if full:
            x_gnn, out2 = gnn_model(x_lstm, edge_index)
        else:
            x_gnn, out2 = gnn_model(torch.cat([torch.matmul(phi.T, x_lstm), x_lstm], dim=1), edge_index)
            x_gnn = x_gnn[:, n_cluster:, :]
            out2 = out2[:, n_cluster:]

        out3 = ens_model(x_lstm, x_gnn)

        prob1s = np.concatenate([prob1s, out1.cpu().numpy()], axis=0)
        prob2s = np.concatenate([prob2s, out2.cpu().numpy()], axis=0)
        prob3s = np.concatenate([prob3s, out3.cpu().numpy()], axis=0)

    labels = eval_y.cpu().numpy()
    return sttpmetric.get_rmse_smape(labels, prob1s), sttpmetric.get_rmse_smape(labels, prob2s), sttpmetric.get_rmse_smape(labels, prob3s)


def run():
    best_epoch1, best_epoch2 = 0, 0
    # LSTM
    if args.backbone == 'lstm':
        lstm_model = LSTM(1, hidden_size, layer_num).to(device)
    elif args.backbone == 'xlstm':
        lstm_model = xLSTM(1, hidden_size, layer_num).to(device)
    elif args.backbone == 'transformer':
        lstm_model = Transformer(1, hidden_size, layer_num).to(device)
    else:
        raise NotImplementedError
    
    lstm_model.load_state_dict(torch.load(f'model/{args.backbone}_{args.dataset}_{t}.pkl'))
    print(f'{args.backbone} load succsessfully.')

    # LSTM
    # lstm_model = LSTM(1, hidden_size, layer_num).to(device)
    # if not os.path.exists(f'model/lstm_{args.dataset}_{t}.pkl'):
    #     train_lstm(lstm_model, t)
    # lstm_model.load_state_dict(torch.load(f'model/lstm_{args.dataset}_{t}.pkl'))
    # print('LSTM load succsessfully.')

    with torch.no_grad():
        lstm_model.eval()
        embeds = torch.empty((0, num_node, hidden_size)).to(device)
        batch_num = int(np.ceil(train_x.shape[0] / batch_size))
        for i in range(1, batch_num + 1):
            input = train_x[(i-1)*batch_size: i*batch_size]
            embed, _ = lstm_model(input)
            embeds = torch.cat([embeds, embed], dim=0)
    
    # # compress
    # phi = kmeans_no(embeds[-1], n_cluster)
    # phi = torch.tensor(phi, device=device, dtype=torch.float)
    # phi = phi / phi.sum(dim=0)

    # X_com = torch.matmul(phi.T, embeds)
    # Y_com = torch.matmul(phi.T, train_y.unsqueeze(2)).squeeze(2)

    # com_edge_index = epsilon_graph(X_com[-1], epsilon).to(device)
    # new_edge_index = epsilon_graph_add(X_com[-1], embeds[-1], n_cluster, epsilon)
    # print(f'nodes={n_cluster}, deg={com_edge_index.shape[1]/n_cluster}')
    # print(f'nodes={n_cluster+num_node}, deg={new_edge_index.shape[1]/(n_cluster+num_node)}')

    # GNN
    gnn_model = GNN(hidden_size, hidden_size_gnn).to(device)
    if not os.path.exists(f'model/{args.method}_{args.graph}_{args.dataset}_5_gnn.pkl') or update:
        if args.method == 'fullGNN':
            best_epoch1 = train_gnn(lstm_model, gnn_model, embeds, train_y, edge_index, phi, full)
        else:
            X_com = torch.matmul(phi.T, embeds)
            best_epoch1 = train_gnn2(lstm_model, gnn_model, embeds, train_y, new_edge_index, phi, full, X_com)
    gnn_model.load_state_dict(torch.load(f'model/{args.method}_{args.graph}_{args.dataset}_{t}_gnn.pkl'))
    print('GNN load succsessfully.')

    # Ensemble
    ens_model = Ensemble(hidden_size + hidden_size_gnn).to(device)
    # if not os.path.exists(f'model/{args.method}_{args.graph}_{args.dataset}_5_ens.pkl') or update:
    #     best_epoch2 = train_ensemble(lstm_model, gnn_model, ens_model, new_edge_index, phi, full)
    # ens_model.load_state_dict(torch.load(f'model/{args.method}_{args.graph}_{args.dataset}_{t}_ens.pkl'))
    # print('ENS load succsessfully.')

    # evaluate
    (rmse_test0, smape_test0), (rmse_test1, smape_test1), (rmse_test2, smape_test2) = eval_all(lstm_model, gnn_model, ens_model, test_x, test_y, new_edge_index, phi, full)
    
    print(f'mse_test0: {rmse_test0}, smape_test0: {smape_test0}, rmse_test1: {rmse_test1}, smape_test1: {smape_test1}, rmse_test2: {rmse_test2}, smape_test2: {smape_test2}')

    return rmse_test0, smape_test0, best_epoch1, rmse_test1, smape_test1, best_epoch2, rmse_test2, smape_test2


if __name__ == '__main__':
    seed_everything(42)
    hidden_size = 64
    hidden_size_gnn = 16
    # hidden_size_gnn = 64
    layer_num = 1
    batch_size = 16
    # epsilon = 0.95
    update = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='bike_nyc')  # bike_nyc, pems_bay
    parser.add_argument("--graph", type=str, default='corr') # dist, corr
    parser.add_argument("--method", type=str, default='ECSeq')  # fullGNN, ECSeq
    parser.add_argument("--backbone", type=str, default='lstm')  # fullGNN, ECSeq
    args = parser.parse_args()

    with open('params.json', 'r', encoding='utf-8') as f:
        params = json.load(f)
        lstm_epo = params[args.dataset]["lstm_epo"]
        gnn_epo = params[args.dataset]["gnn_epo"]
        ens_epo = params[args.dataset]["ens_epo"]
        eva = params[args.dataset]["eva"]
        cor_thres = params[args.dataset]["cor_thres"]
        # cor_thres_com = params[args.dataset]["cor_thres_com"]
        scalar_min = params[args.dataset]["scalar_min"]
        scalar_max = params[args.dataset]["scalar_max"]
        # n_cluster = params[args.dataset]["n_cluster"]
    n_cluster = 100
       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    sttpmetric = STTPMetric(scalar_min, scalar_max)

    dict_data = np.load(f'data/{args.dataset}.npy', allow_pickle=True).item()
    train_x, train_y = torch.tensor(dict_data['train']['X'], dtype=torch.float).to(device), torch.tensor(dict_data['train']['y'], dtype=torch.float).to(device)
    val_x, val_y = torch.tensor(dict_data['val']['X'], dtype=torch.float).to(device), torch.tensor(dict_data['val']['y'], dtype=torch.float).to(device)
    test_x, test_y = torch.tensor(dict_data['test']['X'], dtype=torch.float).to(device), torch.tensor(dict_data['test']['y'], dtype=torch.float).to(device)
    
    if args.graph == 'dist':
        edge_index = torch.tensor(dict_data['edge_index_dist']).long().to(device)
    elif args.graph == 'corr':
        edge_index = torch.tensor(dict_data['edge_index_corr']).long().to(device)
    else:
        raise NotImplementedError

    num_node = train_x.shape[1]

    loss_mse = torch.nn.MSELoss()
    full = False if args.method == 'ECSeq' else True

    # compress
    if args.method == 'ECSeq':
        if args.graph == 'dist':
            position = torch.tensor(dict_data['pos']).float().to(device)
            phi = kmeans_no(position, n_cluster)
            phi = torch.tensor(phi, device=device, dtype=torch.float)
            phi = phi / phi.sum(dim=0)
            
            com_position = torch.mm(phi.T, position)
            # com_edge_index = distance_edge(com_position, com_position).to(device)
            new_edge_index = distance_edge(com_position, position).to(device)
            new_edge_index[1] = new_edge_index[1] + n_cluster

        elif args.graph == 'corr':
            X_cor = train_x[-24*30:, :, -1].T

            # phi = kmeans_no(X_cor.cpu().numpy(), n_cluster)
            # phi = phi / phi.sum(axis=0)

            # print(f'ori, nodes={num_node}, deg={edge_index.shape[1]/num_node}')
            # agc = AGC(X_cor.cpu().numpy(), edge_index, n_cluster=n_cluster) #, bestpower=5
            # phi = agc.get_phi()
            # phi = phi / phi.sum(axis=0)

            # grain = Grain(X_cor, edge_index, num_coreset=n_cluster) #, bestpower=5
            # phi, n_cluster = grain.get_phi()

            loukas = Loukas(torch.tensor(X_cor, dtype=torch.float), edge_index, n_cluster)
            phi = loukas.get_phi()
            n_cluster = phi.shape[1]
            phi = phi / phi.sum(axis=0)

            phi = torch.tensor(phi, device=device, dtype=torch.float)
            com_cor = torch.mm(phi.T, X_cor)
            # com_edge_index = correlation_edge(com_cor, com_cor, cor_thres_com).to(device)
            new_edge_index = correlation_edge(com_cor, X_cor, cor_thres).to(device)
            new_edge_index[1] = new_edge_index[1] + n_cluster

        else:
            raise NotImplementedError

        # print(f'nodes={n_cluster}, deg={com_edge_index.shape[1]/n_cluster}')
        print(f'nodes={n_cluster+num_node}, deg={new_edge_index.shape[1]/num_node}')
        print(f'nodes={num_node}, deg={edge_index.shape[1]/num_node}')
        print(phi.shape)
    
    else:
        new_edge_index = edge_index
        phi = None
        print(f'nodes={train_x.shape[1]}, deg={edge_index.shape[1]/train_x.shape[1]}')

    results = []
    t0 = time()
    for t in range(1, 6):
        print('times', t)
        res = run()
        results.append(res)
        with open('results.txt', 'a') as f:
            f.write(f'{args.method}_{args.graph} final, time: {time() - t0}, {args.dataset}, '
                    f'[lstm] rmse={res[0]}, smape={res[1]}, '
                    f'[gnn] epoch={res[2]}, rmse={res[3]}, smape={res[4]}, '
                    f'[ens] epoch={res[5]}, rmse={res[6]}, smape={res[7]}\n')
            f.flush()
    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    with open('results.txt', 'a') as f:
        f.write(f'{args.method}_{args.graph} final avg, time: {time() - t0}, {args.dataset}, '
                f'[lstm] rmse={mean[0]:.4f}±{std[0]:.4f}, smape={mean[1]:.4f}±{std[1]:.4f}, '
                f'[gnn] epoch={mean[2]}, rmse={mean[3]:.4f}±{std[3]:.4f}, smape={mean[4]:.4f}±{std[4]:.4f}, '
                f'[ens] epoch={mean[5]}, rmse={mean[6]:.4f}±{std[6]:.4f}, smape={mean[7]:.4f}±{std[7]:.4f}\n')
        f.flush()




