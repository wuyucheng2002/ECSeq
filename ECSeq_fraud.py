import torch
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from time import time
from utils import *
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv
from torch_geometric.data import Data
from torch_geometric import seed_everything
import json
import os
from utils import *
from utils_compress import *
from utils_transformer import *


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_size, layer_num):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, layer_num)
        self.fc = torch.nn.Linear(hidden_size, out_size)

    def forward(self, x, x_length, end):
        x = pack_padded_sequence(x, x_length.cpu(), enforce_sorted=False, batch_first=True)
        x, _ = self.lstm(x)
        output, _ = pad_packed_sequence(x)
        target_x = []
        for j, length in enumerate(x_length):
            target_x.append(output[length - 1, j, :])
        target_x = torch.stack(target_x, dim=0)
        return target_x, self.fc(target_x).softmax(dim=1)
    

class GNN(torch.nn.Module):
    def __init__(self, hidden_lstm, hidden_gnn, out_channels, backbone='GraphSAGE', heads=8):
        super().__init__()
        self.backbone = backbone
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

    def forward(self, x, edge_index, edge_weight=None):
        if self.backbone == 'GCN':
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv(x, edge_index, edge_weight)
        elif self.backbone in ['GraphSAGE', 'GraphSAGE_max']:
            x = self.conv(x, edge_index)
        elif self.backbone == 'GAT':
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv(x, edge_index)
        return x, self.fc(x).sigmoid().squeeze()
    

def train_lstm(lstm_model):
    if args.seq_backbone == 'lstm':
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.00001)
    elif args.seq_backbone == 'transformer':
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.0001)
    else:
        raise NotImplementedError

    best_auc_val = 0
    auc_vals, auc_tests = [], []
    step = 0
    for epoch in range(1, 51):
        lstm_model.train()
        with tqdm(train_loader, desc='Train') as loop:
            for batch in loop:
                inputs, labels, lengths, ends, _ = batch
                lstm_model.zero_grad()
                _, prob = lstm_model(inputs, lengths, ends)
                loss = loss_ce(prob.log(), labels)
                loss.backward()
                optimizer.step()
                loop.set_postfix(T=t, epoch=epoch, loss=loss.item())
        
        auc_val, _, _ = eval_lstm(lstm_model, val_loader, 'Val')
        auc_test, _, _ = eval_lstm(lstm_model, test_loader, 'Test')
        auc_vals.append(auc_val)
        auc_tests.append(auc_test)

        if auc_val > best_auc_val:
            best_auc_val = auc_val
            step = 0
            torch.save(lstm_model.state_dict(), f'model/{args.seq_backbone}_{args.dataset}_{t}.pkl')
            print('Save succsessfully.')
        else:
            step += 1
        
        if step == 10: 
            break

        print(f'T: {t}, time: {time() - t0}, Epoch: {epoch}, val AUC: {auc_val}, test AUC: {auc_test}')

        plt.figure()
        plt.plot(range(len(auc_vals)), auc_vals, label='auc_val')
        plt.plot(range(len(auc_vals)), auc_tests, label='auc_test')
        plt.legend()
        plt.savefig(f'fig/{args.seq_backbone}_{args.dataset}_{t}.jpg')
        plt.close()


def train_gnn_batch(lstm_model, gnn_model, embeds, labels, embeds_val, labels_val, embeds_test, labels_test):
    gnn_model.train()
    gnn_opt = torch.optim.Adam(gnn_model.parameters(), lr=0.005)
    
    best_auc_val1 = 0
    auc_vals1, auc_tests1 = [], []
    step = 0
    idxs = np.random.permutation(range(embeds.shape[0]))

    for epoch in range(1, 300 + 1):
        batch_num = int(np.ceil(embeds.shape[0] / gnn_batch_size))
        edge_num = 0

        with tqdm(range(batch_num), desc=f'T{t} Train GNN {epoch}') as loop:
            for i in loop:
                gnn_model.zero_grad()
                idx = idxs[i*gnn_batch_size: (i+1)*gnn_batch_size]
                X = embeds[idx, :]
                Y = labels[idx]
                edge_index = epsilon_graph(X, epsilon).to(device)
                _, prob = gnn_model(X, edge_index)
                loss = loss_mse(prob, Y)
                loss.backward()
                gnn_opt.step()
                edge_num += edge_index.shape[1]
                loop.set_postfix(epoch=epoch, loss=loss.item(), deg=edge_index.shape[1]/X.shape[0], sparsity=edge_index.shape[1]/X.shape[0]/X.shape[0])
        
        auc_val1, _, _ = eval_gnn(lstm_model, gnn_model, embeds, embeds_val, labels_val)
        auc_test1, _, _ = eval_gnn(lstm_model, gnn_model, embeds, embeds_test, labels_test)

        auc_vals1.append(auc_val1)
        auc_tests1.append(auc_test1)

        if auc_val1 > best_auc_val1:
            best_auc_val1 = auc_val1
            step = 0
            torch.save(gnn_model.state_dict(), f'model/{args.method}_{args.dataset}_{t}_gnn.pkl')
            print('GNN save succsessfully.')
        else:
            step += 1
        
        if step == 20:
            break

        print(f'{args.method}, {args.dataset}, [gnn] T={t}, Epoch={epoch}, val auc={auc_val1:.4f}, test auc={auc_test1:.4f}')

        plt.figure()
        plt.plot(range(len(auc_vals1)), auc_vals1, label='gnn_auc_val')
        plt.plot(range(len(auc_tests1)), auc_tests1, label='gnn_auc_test')
        plt.legend()
        plt.savefig(f'fig/{args.method}_{args.dataset}_{t}_gnn.jpg')
        plt.close()


def train_gnn2(lstm_model, gnn_model, graph, embeds, labels, embeds_val, labels_val, embeds_test, labels_test):
    gnn_opt = torch.optim.Adam(gnn_model.parameters(), lr=0.005)
    
    best_auc_val1 = 0
    auc_vals1, auc_tests1 = [], []
    # auc_trains1 = []
    step = 0

    idx = np.random.permutation(range(embeds.shape[0]))
    batch_num = int(np.ceil(embeds.shape[0] / gnn_batch_size))
    edge_nums = []

    for epoch in range(1, 300 + 1):
        train_loss = 0
        edge_num = 0
        for i in range(batch_num):
            embed = embeds[idx[i*gnn_batch_size: (i+1)*gnn_batch_size]]
            label = labels[idx[i*gnn_batch_size: (i+1)*gnn_batch_size]]

            gnn_model.train()
            gnn_model.zero_grad()
            num = graph.x.shape[0]

            new_edge_index = epsilon_graph_add(graph.x, embed, num, epsilon)
            edge_num += new_edge_index.shape[1]

            _, prob = gnn_model(torch.cat([graph.x, embed], dim=0), new_edge_index)
            prob = prob[num:]

            loss = loss_mse(prob, label)
            loss.backward()
            gnn_opt.step()
            train_loss += loss.item()/batch_num

        auc_val1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, embeds_val, labels_val)
        auc_test1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, embeds_test, labels_test)

        auc_vals1.append(auc_val1)
        auc_tests1.append(auc_test1)

        if auc_val1 > best_auc_val1:
            best_auc_val1 = auc_val1
            step = 0
            torch.save(gnn_model.state_dict(), f'model/{args.method}_{args.dataset}_{t}_gnn.pkl')
            print('GNN save succsessfully.')
        else:
            step += 1
        
        if step == 30:
            break

        edge_nums.append(edge_num)
        print(f'{args.method}, {args.dataset}, [gnn] T={t}, Epoch={epoch}, loss={train_loss}, edge_num={edge_num}'
              f', val auc={auc_val1:.4f}, test auc={auc_test1:.4f}')

        plt.figure()
        plt.plot(range(len(auc_vals1)), auc_vals1, label='gnn_auc_val')
        plt.plot(range(len(auc_tests1)), auc_tests1, label='gnn_auc_test')
        plt.legend()
        plt.savefig(f'fig/{args.method}_{args.dataset}_{t}_gnn.jpg')
        plt.close()


@torch.no_grad()
def eval_lstm(lstm_model, eval_loader, desc):
    lstm_model.eval()
    label_list = []
    prob_list = []
    with tqdm(eval_loader, desc=desc) as loop:
        for batch in loop:
            inputs, labels, lengths, ends, _ = batch
            _, prob = lstm_model(inputs, lengths, ends)
            label_list.append(labels.cpu().numpy())
            prob_list.append(prob[:, 1].cpu().numpy())

    label_array = np.concatenate(label_list, axis=0)
    prob_array = np.concatenate(prob_list, axis=0)
    return get_auc_ap_rp(label_array, prob_array)


@torch.no_grad()
def eval_gnn(lstm_model, gnn_model, X_com, embeds_eva, labels_eva):
    lstm_model.eval()
    gnn_model.eval()

    prob2s = np.empty(0)
    batch_num = int(np.ceil(embeds_eva.shape[0] / gnn_batch_size))
    for i in range(batch_num):
        x_lstm = embeds_eva[i*gnn_batch_size: (i+1)*gnn_batch_size]

        num = X_com.shape[0]
        new_edge_index = epsilon_graph_add(X_com, x_lstm, num, epsilon)

        _, prob2 = gnn_model(torch.cat([X_com, x_lstm], dim=0), new_edge_index)
        prob2 = prob2[num:]
        prob2s = np.concatenate([prob2s, prob2.cpu().numpy()], axis=0)

    return get_auc_ap_rp(labels_eva, prob2s)


def run():
    # sequence embedding extractor
    if args.seq_backbone == 'lstm':
        lstm_model = LSTM(input_size, hidden_size, 2, layer_num).to(device)
    elif args.seq_backbone == 'transformer':
        lstm_model = vanilla_transformer_encoder(input_dim=input_size, hidden_dim=32, d_model=32,  MHD_num_head=4, d_ff=2*32, output_dim=2).to(device)
    else:
        raise NotImplementedError
    
    if not os.path.exists(f'model/{args.seq_backbone}_{args.dataset}_{t}.pkl'):
        train_lstm(lstm_model)
    
    lstm_model.load_state_dict(torch.load(f'model/{args.seq_backbone}_{args.dataset}_{t}.pkl'))
    print(f'{args.seq_backbone} load succsessfully.')
    auc0, ap0, rp0 = eval_lstm(lstm_model, test_loader, 'Test')

    with torch.no_grad():
        lstm_model.eval()
        embeds = np.empty((0, hidden_size))
        labels = np.empty(0)
        for batch in train_loader2:
            input, label, length, end, _ = batch
            lstm_model.zero_grad()
            embed, _ = lstm_model(input, length, end)
            embeds = np.concatenate([embeds, embed.cpu().numpy()], axis=0)
            labels = np.concatenate([labels, label.cpu().numpy()], axis=0)

        embeds_val = np.empty((0, hidden_size))
        labels_val = np.empty(0)
        for batch in val_loader:
            input, label, length, end, _ = batch
            lstm_model.zero_grad()
            embed, _ = lstm_model(input, length, end)
            embeds_val = np.concatenate([embeds_val, embed.cpu().numpy()], axis=0)
            labels_val = np.concatenate([labels_val, label.cpu().numpy()], axis=0)
        embeds_val = torch.tensor(embeds_val).float().to(device)

        embeds_test = np.empty((0, hidden_size))
        labels_test = np.empty(0)
        for batch in test_loader:
            input, label, length, end, _ = batch
            lstm_model.zero_grad()
            embed, _ = lstm_model(input, length, end)
            embeds_test = np.concatenate([embeds_test, embed.cpu().numpy()], axis=0)
            labels_test = np.concatenate([labels_test, label.cpu().numpy()], axis=0)
        embeds_test = torch.tensor(embeds_test).float().to(device)

    # graph compression
    if args.method == 'ECSeq':
        if args.compress == 'kmeans_ba':
            phi = kmeans_ba(embeds, labels, n_pos=args.n_pos, n_neg=args.n_neg)
        elif args.compress == 'kmeans_no':
            phi = kmeans_no(embeds, n_cluster=1000)
        elif args.compress == 'AGC':
            edge_index_ori = epsilon_graph(torch.tensor(embeds, dtype=torch.float), epsilon).to(device)
            print(f'ori, nodes={embeds.shape[0]}, deg={edge_index_ori.shape[1]/embeds.shape[0]}')
            agc = AGC(embeds, edge_index_ori, n_cluster=100, bestpower=5)
            phi = agc.get_phi()
        elif args.compress == 'Grain':
            edge_index_ori = epsilon_graph(torch.tensor(embeds, dtype=torch.float), 0.95).to(device)
            print(f'ori, nodes={embeds.shape[0]}, deg={edge_index_ori.shape[1]/embeds.shape[0]}')
            grain = Grain(torch.tensor(embeds, dtype=torch.float), edge_index_ori, num_coreset=200)
            phi, n_cluster2 = grain.get_phi()
            print(n_cluster2)
        elif args.compress == 'RSA':
            edge_index_ori = epsilon_graph(torch.tensor(embeds, dtype=torch.float), epsilon).to(device)
            print(f'ori, nodes={embeds.shape[0]}, deg={edge_index_ori.shape[1]/embeds.shape[0]}')
            loukas = Loukas(torch.tensor(embeds, dtype=torch.float), edge_index_ori, 1000)
            phi = loukas.get_phi()
        else: 
            raise NotImplementedError

        phi = torch.tensor(phi, device=device, dtype=torch.float)
        phi = phi / phi.sum(dim=0)

        embeds = torch.tensor(embeds, device=device, dtype=torch.float)
        labels = torch.tensor(labels, device=device, dtype=torch.float)

        X_com = phi.T @ embeds
        Y_com = phi.T @ labels
        edge_index = epsilon_graph(X_com, epsilon).to(device)
        com_graph = Data(x=X_com, y=Y_com, edge_index=edge_index)
        degree = edge_index.shape[1]/X_com.shape[0]
        print(f'com, nodes={X_com.shape[0]}, deg={degree}')
    else:
        embeds = torch.tensor(embeds, device=device, dtype=torch.float)
        labels = torch.tensor(labels, device=device, dtype=torch.float)
        X_com = embeds

    # graph mining
    gnn_model = GNN(hidden_size, hidden_size_gnn, 1, args.gnn_backbone).to(device)

    if args.method == 'ECSeq':
        train_gnn2(lstm_model, gnn_model, com_graph, embeds, labels, embeds_val, labels_val, embeds_test, labels_test)
    elif args.method == 'batchGNN': 
        train_gnn_batch(lstm_model, gnn_model, embeds, labels, embeds_val, labels_val, embeds_test, labels_test)

    gnn_model.load_state_dict(torch.load(f'model/{args.method}_{args.dataset}_{t}_gnn.pkl'))
    print('GNN load succsessfully.')
 
    auc1, ap1, rp1 = eval_gnn(lstm_model, gnn_model, X_com, embeds_test, labels_test)

    return auc0, ap0, rp0, auc1, ap1, rp1


if __name__ == '__main__':
    seed_everything(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='FD2')
    parser.add_argument("--seq_backbone", type=str, default='lstm')  # lstm, transformer
    parser.add_argument("--gnn_backbone", type=str, default='GraphSAGE')  # GraphSAGE, GraphSAGE_max, GCN, GAT
    parser.add_argument("--epsilon", type=float, default=0.99)  # epsilon-graph
    parser.add_argument("--method", type=str, default='ECSeq')  # ECSeq, batchGNN
    parser.add_argument("--compress", type=str, default='kmeans_ba')  # kmeans_ba, kmeans_no, AGC, Grain, RSA
    parser.add_argument("--n_pos", type=int, default=100)  # number of positive compressed nodes
    parser.add_argument("--n_neg", type=int, default=400)  # number of negative compressed nodes
    args = parser.parse_args()

    hidden_size = 256 if args.method == 'lstm' else 32
    hidden_size_gnn = 32
    layer_num = 1
    batch_size = 64
    gnn_batch_size = 1000
    epsilon = args.epsilon
       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(args)

    loss_ce = torch.nn.NLLLoss()
    loss_mse = torch.nn.MSELoss()

    train_loader, train_loader2, val_loader, test_loader, input_size = get_dataset(args, batch_size, device)

    results = []
    t0 = time()
    for t in range(1,6):
        print('times', t)
        res = run()
        results.append(res)
        with open('results.txt', 'a') as f:
            f.write(f'{args.method} final, time: {time() - t0}, {args.dataset}, '
                    f'[lstm] auc={res[0]:.4f}, ap={res[1]:.4f}, rp={res[2]:.4f}, '
                    f'[gnn] auc={res[3]:.4f}, ap={res[4]:.4f}, rp={res[5]:.4f}\n')
            f.flush()
    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    with open('results.txt', 'a') as f:
        f.write(f'{args.method} final avg, time: {time() - t0}, {args.dataset}, '
                f'[lstm] {mean[0]:.4f}±{std[0]:.4f}, {mean[1]:.4f}±{std[1]:.4f}, {mean[2]:.4f}±{std[2]:.4f}, '
                f'[gnn] {mean[3]:.4f}±{std[3]:.4f}, {mean[4]:.4f}±{std[4]:.4f}, {mean[5]:.4f}±{std[5]:.4f}\n')
        f.flush()
    
    print(f'{args.method} final avg, time: {time() - t0}, {args.dataset}, '
          f'[lstm] {mean[0]:.4f}±{std[0]:.4f}, {mean[1]:.4f}±{std[1]:.4f}, {mean[2]:.4f}±{std[2]:.4f}, '
          f'[gnn] {mean[3]:.4f}±{std[3]:.4f}, {mean[4]:.4f}±{std[4]:.4f}, {mean[5]:.4f}±{std[5]:.4f}')
