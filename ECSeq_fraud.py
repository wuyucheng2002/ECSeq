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
        x, _ = self.lstm(x)
        output, _ = pad_packed_sequence(x)
        target_x = []
        for j, length in enumerate(x_length):
            target_x.append(output[length - 1, j, :])
        target_x = torch.stack(target_x, dim=0)
        return target_x, self.fc(target_x).softmax(dim=1)
    

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num)
        self.fc1 = nn.Linear(hidden_size + input_size, 32)
        self.fc2 = nn.Linear(32, output_size)

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
        elif self.backbone == 'GAT':
            self.conv = GATConv(hidden_lstm, hidden_gnn, heads, dropout=0.6)
            # self.conv2 = GATConv(hidden_gnn * heads, out_channels, heads=1, concat=False, dropout=0.6)
            self.fc = torch.nn.Linear(hidden_gnn * heads, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        if self.backbone == 'GCN':
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv(x, edge_index, edge_weight)
        elif self.backbone == 'GraphSAGE':
            x = self.conv(x, edge_index)
        elif self.backbone == 'GAT':
            x = F.dropout(x, p=0.6, training=self.training)
            # x = F.elu(self.conv1(x, edge_index))
            # x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv(x, edge_index)
        return x, self.fc(x).sigmoid().squeeze()


class Ensemble(torch.nn.Module):
    def __init__(self, input_size, out_size, hidden_size=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, out_size)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x.softmax(dim=1)


def train_gnn_full(lstm_model, gnn_model, embeds, labels, embeds_val, labels_val, embeds_test, labels_test):
    gnn_model.train()
    gnn_opt = torch.optim.Adam(gnn_model.parameters(), lr=gnn_lr)
    
    best_auc_val1 = 0
    auc_vals1, auc_tests1 = [], []
    # auc_trains1 = []
    step = 0
    idxs = np.random.permutation(range(embeds.shape[0]))

    for epoch in range(1, gnn_epo + 1):
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
        # auc_train1, _, _ = eval_gnn(lstm_model, gnn_model, embeds, train_loader2)

        auc_vals1.append(auc_val1)
        auc_tests1.append(auc_test1)
        # auc_trains1.append(auc_train1)

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
        # plt.plot(range(len(auc_trains1)), auc_trains1, label='gnn_auc_train')
        plt.legend()
        plt.savefig(f'fig/{args.method}_{args.dataset}_{t}_gnn.jpg')
        plt.close()

    return len(auc_vals1) - step, edge_num/embeds.shape[0]


def train_gnn(lstm_model, gnn_model, graph):
    gnn_opt = torch.optim.Adam(gnn_model.parameters(), lr=gnn_lr)
    
    best_auc_val1 = 0
    auc_vals1, auc_tests1 = [], []
    auc_trains1 = []
    step = 0
    for epoch in range(1, gnn_epo + 1):
        gnn_model.train()
        gnn_model.zero_grad()
        _, prob = gnn_model(graph.x, graph.edge_index)
        loss = loss_mse(prob, graph.y)
        loss.backward()
        gnn_opt.step()
        print(f'[gnn] epoch={epoch}, loss={loss.item()}')
        
        auc_val1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, val_loader)
        auc_test1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, test_loader)
        auc_train1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, train_loader2)

        auc_vals1.append(auc_val1)
        auc_tests1.append(auc_test1)
        auc_trains1.append(auc_train1)

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
        plt.plot(range(len(auc_trains1)), auc_trains1, label='gnn_auc_train')
        plt.legend()
        plt.savefig(f'fig/{args.method}_{args.dataset}_{t}_gnn.jpg')
        plt.close()

    return len(auc_vals1) - step


def train_gnn2(lstm_model, gnn_model, graph, embeds, labels, embeds_val, labels_val, embeds_test, labels_test):
    gnn_opt = torch.optim.Adam(gnn_model.parameters(), lr=gnn_lr)
    
    best_auc_val1 = 0
    auc_vals1, auc_tests1 = [], []
    # auc_trains1 = []
    step = 0

    idx = np.random.permutation(range(embeds.shape[0]))
    batch_num = int(np.ceil(embeds.shape[0] / gnn_batch_size))
    edge_nums = []

    for epoch in range(1, gnn_epo + 1):
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

        # auc_val1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, embeds_val, labels_val)
        # auc_test1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, embeds_test, labels_test)
        _, auc_val1, _ = eval_gnn(lstm_model, gnn_model, graph.x, embeds_val, labels_val)
        _, auc_test1, _ = eval_gnn(lstm_model, gnn_model, graph.x, embeds_test, labels_test)
        # auc_train1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, train_loader2)

        auc_vals1.append(auc_val1)
        auc_tests1.append(auc_test1)
        # auc_trains1.append(auc_train1)

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
        print(f'{args.method}, {args.dataset}, [gnn] T={t}, Epoch={epoch}, loss={train_loss}, edge_num={edge_num}, sparisty={edge_num/(n_pos+n_neg)/embeds.shape[0]}'
              f', val auc={auc_val1:.4f}, test auc={auc_test1:.4f}')

        plt.figure()
        plt.plot(range(len(auc_vals1)), auc_vals1, label='gnn_auc_val')
        plt.plot(range(len(auc_tests1)), auc_tests1, label='gnn_auc_test')
        # plt.plot(range(len(auc_trains1)), auc_trains1, label='gnn_auc_train')
        plt.legend()
        plt.savefig(f'fig/{args.method}_{args.dataset}_{t}_gnn.jpg')
        plt.close()

    print(f'edge_num={np.mean(edge_nums)}, sparisty={np.mean(edge_nums)/(n_pos+n_neg)/embeds.shape[0]}')

    return len(auc_vals1) - step


def train_gnn3(lstm_model, gnn_model, graph, embeds, labels):
    gnn_opt = torch.optim.Adam(gnn_model.parameters(), lr=gnn_lr)
    
    best_auc_val1 = 0
    auc_vals1, auc_tests1 = [], []
    step = 0

    idx = np.random.permutation(range(embeds.shape[0]))
    batch_num = int(np.ceil(embeds.shape[0] / gnn_batch_size))
    edge_nums = []

    for epoch in range(1, gnn_epo + 1):
        train_loss = 0
        edge_num = 0
        for i in range(batch_num):
            embed = embeds[idx[(i-1)*gnn_batch_size: i*gnn_batch_size]]
            label = labels[idx[(i-1)*gnn_batch_size: i*gnn_batch_size]]

            gnn_model.train()
            gnn_model.zero_grad()
            num = graph.x.shape[0]

            new_edge_index = torch.cat([graph.edge_index, epsilon_graph_add(graph.x, embed, num, epsilon)], dim=1)
            edge_num += new_edge_index.shape[1]

            _, prob = gnn_model(torch.cat([graph.x, embed], dim=0), new_edge_index)

            loss = loss_mse(prob, torch.cat([graph.y, label], dim=0))
            loss.backward()
            gnn_opt.step()
            train_loss += loss.item()/batch_num

        auc_val1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, val_loader)
        auc_test1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, test_loader)

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

        edge_nums.append(edge_num)
        print(f'{args.method}, {args.dataset}, [gnn] T={t}, Epoch={epoch}, loss={train_loss}, edge_num={edge_num}, sparisty={edge_num/(n_pos+n_neg+embeds.shape[0])/embeds.shape[0]}'
              f', val auc={auc_val1:.4f}, test auc={auc_test1:.4f}')

        plt.figure()
        plt.plot(range(len(auc_vals1)), auc_vals1, label='gnn_auc_val')
        plt.plot(range(len(auc_tests1)), auc_tests1, label='gnn_auc_test')
        plt.legend()
        plt.savefig(f'fig/{args.method}_{args.dataset}_{t}_gnn.jpg')
        plt.close()

    print(f'edge_num={np.mean(edge_nums)}, sparisty={np.mean(edge_nums)/(n_pos+n_neg)/embeds.shape[0]}')

    return len(auc_vals1) - step


def train_gnn4(lstm_model, gnn_model, graph, embeds, labels):
    gnn_opt = torch.optim.Adam(gnn_model.parameters(), lr=gnn_lr)

    # compress graph pretrain
    best_auc_val1 = 0
    auc_vals1, auc_tests1 = [], []
    auc_trains1 = []
    step = 0
    for epoch in range(1, gnn_epo + 1):
        gnn_model.train()
        gnn_model.zero_grad()
        _, prob = gnn_model(graph.x, graph.edge_index)
        loss = loss_mse(prob, graph.y)
        loss.backward()
        gnn_opt.step()

        auc_val1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, val_loader)
        auc_test1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, test_loader)
        auc_train1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, train_loader2)

        auc_vals1.append(auc_val1)
        auc_tests1.append(auc_test1)
        auc_trains1.append(auc_train1)

        if auc_val1 > best_auc_val1:
            best_auc_val1 = auc_val1
            step = 0
            torch.save(gnn_model.state_dict(), f'model/{args.method}_{args.dataset}_{t}_gnn.pkl')
            print('GNN save succsessfully.')
        else:
            step += 1
        
        if step == 40:
            break

        print(f'{args.method}, {args.dataset}, [gnn-pretrain] T={t}, Epoch={epoch}, loss={loss.item()}, train_auc={auc_train1:.4f}, val auc={auc_val1:.4f}, test auc={auc_test1:.4f}, '
              f'edge_num={graph.edge_index.shape[1]}, sparisty={graph.edge_index.shape[1]/(n_pos+n_neg)/(n_pos+n_neg)}')

        plt.figure()
        plt.plot(range(len(auc_vals1)), auc_vals1, label='gnn_auc_val')
        plt.plot(range(len(auc_tests1)), auc_tests1, label='gnn_auc_test')
        plt.plot(range(len(auc_trains1)), auc_trains1, label='gnn_auc_train')
        plt.legend()
        plt.savefig(f'fig/{args.method}_{args.dataset}_{t}_gnn.jpg')
        plt.close()
    
    # inference graph finetune
    auc_val1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, val_loader)
    auc_test1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, test_loader)
    auc_train1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, train_loader2)

    auc_vals1.append(auc_val1)
    auc_tests1.append(auc_test1)
    auc_trains1.append(auc_train1)

    print('='*20)
    step = 0

    gnn_model.load_state_dict(torch.load(f'model/{args.method}_{args.dataset}_{t}_gnn.pkl'))
    gnn_opt = torch.optim.Adam(gnn_model.parameters(), lr=gnn_lr)

    idx = np.random.permutation(range(embeds.shape[0]))
    batch_num = int(np.ceil(embeds.shape[0] / gnn_batch_size))

    for epoch in range(1, gnn_epo + 1):
        train_loss = 0
        edge_num = 0

        for i in range(batch_num):
            embed = embeds[idx[(i-1)*gnn_batch_size: i*gnn_batch_size]]
            label = labels[idx[(i-1)*gnn_batch_size: i*gnn_batch_size]]

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
            
        auc_val1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, val_loader)
        auc_test1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, test_loader)
        auc_train1, _, _ = eval_gnn(lstm_model, gnn_model, graph.x, train_loader2)

        auc_vals1.append(auc_val1)
        auc_tests1.append(auc_test1)
        auc_trains1.append(auc_train1)

        if auc_val1 > best_auc_val1:
            best_auc_val1 = auc_val1
            step = 0
            torch.save(gnn_model.state_dict(), f'model/{args.method}_{args.dataset}_{t}_gnn.pkl')
            print('GNN save succsessfully.')
        else:
            step += 1
        
        if step == 40:
            break
        
        print(f'{args.method}, {args.dataset}, [gnn-finetune] T={t}, Epoch={epoch}, loss={loss.item()}, train auc={auc_train1:.4f}, val auc={auc_val1:.4f}, test auc={auc_test1:.4f}, '
              f'edge_num={edge_num}, sparisty={edge_num/(n_pos+n_neg)/(embeds.shape[0])}')

        plt.figure()
        plt.plot(range(len(auc_vals1)), auc_vals1, label='gnn_auc_val')
        plt.plot(range(len(auc_tests1)), auc_tests1, label='gnn_auc_test')
        plt.plot(range(len(auc_trains1)), auc_trains1, label='gnn_auc_train')
        plt.legend()
        plt.savefig(f'fig/{args.method}_{args.dataset}_{t}_gnn.jpg')
        plt.close()

    return len(auc_vals1) - step
    

def train_ensemble(lstm_model, gnn_model, ens_model, X_com):
    ens_opt = optim.Adam(ens_model.parameters(), lr=ens_lr)
    ens_model.train()
    lstm_model.eval()
    gnn_model.eval()

    best_auc_val2 = 0
    auc_vals2, auc_tests2 = [], []
    step = 0
    for epoch in range(1, ens_epo + 1):
        for batch in train_loader:
            with torch.no_grad():
                input, label, length, end, _ = batch
                x_lstm, _ = lstm_model(input, length, end)

                num = X_com.shape[0]
                new_edge_index = epsilon_graph_add(X_com, x_lstm, num, epsilon)

                x_gnn, _ = gnn_model(torch.cat([X_com, x_lstm], dim=0), new_edge_index)
                x_gnn = x_gnn[num:]

            ens_model.zero_grad()
            out3 = ens_model(x_lstm, x_gnn)
            loss = loss_ce(out3.log(), label)
            loss.backward()
            ens_opt.step()

        _, _, (auc_val2, _, _) = eval_all(lstm_model, gnn_model, ens_model, X_com, val_loader)
        _, _, (auc_test2, _, _) = eval_all(lstm_model, gnn_model, ens_model, X_com, test_loader)

        auc_vals2.append(auc_val2)
        auc_tests2.append(auc_test2)

        if auc_val2 > best_auc_val2:
            best_auc_val2 = auc_val2
            step = 0
            torch.save(ens_model.state_dict(), f'model/{args.method}_{args.dataset}_{t}_ens.pkl')
            print('ENS save succsessfully.')
        else:
            step += 1
        
        if step == 5:
            break

        print(f'{args.method}, {args.dataset}, [ens] T={t}, Epoch={epoch}, val auc={auc_val2:.4f}, test auc={auc_test2:.4f}')

        plt.figure()
        plt.plot(range(len(auc_vals2)), auc_vals2, label='ens_auc_val')
        plt.plot(range(len(auc_tests2)), auc_tests2, label='ens_auc_test')
        plt.legend()
        plt.savefig(f'fig/{args.method}_{args.dataset}_{t}_ens.jpg')
        plt.close()
    return len(auc_vals2) - step


@torch.no_grad()
def eval_gnn(lstm_model, gnn_model, X_com, embeds_eva, labels_eva):
    lstm_model.eval()
    gnn_model.eval()

    prob2s = np.empty(0)

    batch_num = int(np.ceil(embeds_eva.shape[0] / gnn_batch_size))
    # for batch in eval_loader:
    for i in range(batch_num):
        x_lstm = embeds_eva[i*gnn_batch_size: (i+1)*gnn_batch_size]

        # input, label, length, end, _ = batch
        # x_lstm, _ = lstm_model(input, length, end)

        num = X_com.shape[0]
        new_edge_index = epsilon_graph_add(X_com, x_lstm, num, epsilon)

        _, prob2 = gnn_model(torch.cat([X_com, x_lstm], dim=0), new_edge_index)
        prob2 = prob2[num:]

        # labels = np.concatenate([labels, label.cpu().numpy()], axis=0)
        prob2s = np.concatenate([prob2s, prob2.cpu().numpy()], axis=0)
    # print(labels_eva.shape, prob2s.shape)

    return get_auc_ap_rp(labels_eva, prob2s)


@torch.no_grad()
def eval_all(lstm_model, gnn_model, ens_model, X_com, eval_loader):
    lstm_model.eval()
    gnn_model.eval()
    ens_model.eval()

    labels = np.empty(0)
    prob1s = np.empty(0)
    prob2s = np.empty(0)
    prob3s = np.empty(0)

    for batch in eval_loader:
        input, label, length, end, _ = batch
        x_lstm, out1 = lstm_model(input, length, end)
        prob1 = out1[:, 1].detach()

        num = X_com.shape[0]
        new_edge_index = epsilon_graph_add(X_com, x_lstm, num, epsilon)

        x_gnn, prob2 = gnn_model(torch.cat([X_com, x_lstm], dim=0), new_edge_index)
        x_gnn, prob2 = x_gnn[num:], prob2[num:]

        out3 = ens_model(x_lstm, x_gnn)
        prob3 = out3[:, 1].detach()

        labels = np.concatenate([labels, label.cpu().numpy()], axis=0)
        prob1s = np.concatenate([prob1s, prob1.cpu().numpy()], axis=0)
        prob2s = np.concatenate([prob2s, prob2.cpu().numpy()], axis=0)
        prob3s = np.concatenate([prob3s, prob3.cpu().numpy()], axis=0)

    return get_auc_ap_rp(labels, prob1s), get_auc_ap_rp(labels, prob2s), get_auc_ap_rp(labels, prob3s)


def run():
    # LSTM
    if args.backbone == 'lstm':
        lstm_model = LSTM(input_size, hidden_size, 2, layer_num).to(device)
    elif args.backbone == 'xlstm':
        lstm_model = xLSTM(input_size, hidden_size, 2, layer_num).to(device)
    elif args.backbone == 'transformer':
        lstm_model = vanilla_transformer_encoder(input_dim=input_size, hidden_dim=32, d_model=32,  MHD_num_head=4, d_ff=2*32, output_dim=2).to(device)
    else:
        raise NotImplementedError
    
    lstm_model.load_state_dict(torch.load(f'model/{args.backbone}_{args.dataset}_{t}.pkl'))
    print(f'{args.backbone} load succsessfully.')

    # lstm_model = LSTM(input_size, hidden_size, 2, layer_num).to(device)
    # lstm_model.load_state_dict(torch.load(f'model/lstm_{args.dataset}_{t}.pkl'))
    # print('LSTM load succsessfully.')

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

    with torch.no_grad():
        lstm_model.eval()
        embeds_val = np.empty((0, hidden_size))
        labels_val = np.empty(0)

        for batch in val_loader:
            input, label, length, end, _ = batch
            lstm_model.zero_grad()
            embed, _ = lstm_model(input, length, end)
            embeds_val = np.concatenate([embeds_val, embed.cpu().numpy()], axis=0)
            labels_val = np.concatenate([labels_val, label.cpu().numpy()], axis=0)

        embeds_val = torch.tensor(embeds_val).float().to(device)
        # labels_val = torch.tensor(labels_val).float().to(device)

    with torch.no_grad():
        lstm_model.eval()
        embeds_test = np.empty((0, hidden_size))
        labels_test = np.empty(0)

        for batch in test_loader:
            input, label, length, end, _ = batch
            lstm_model.zero_grad()
            embed, _ = lstm_model(input, length, end)
            embeds_test = np.concatenate([embeds_test, embed.cpu().numpy()], axis=0)
            labels_test = np.concatenate([labels_test, label.cpu().numpy()], axis=0)

        embeds_test = torch.tensor(embeds_test).float().to(device)
        # labels_test = torch.tensor(labels_test).float().to(device)

    if 'ECSeq' in args.method:
        # compress
        if args.compress == 'kmeans_ba':
            phi = kmeans_ba(embeds, labels, n_pos=n_pos, n_neg=n_neg)
        elif args.compress == 'kmeans_no':
            phi = kmeans_no(embeds, n_cluster=n_cluster)
        elif args.compress == 'AGC':
            edge_index_ori = epsilon_graph(torch.tensor(embeds, dtype=torch.float), epsilon).to(device)
            print(f'ori, nodes={embeds.shape[0]}, deg={edge_index_ori.shape[1]/embeds.shape[0]}')
            agc = AGC(embeds, edge_index_ori, n_cluster=100, bestpower=5)
            phi = agc.get_phi()
        elif args.compress == 'Grain':
            edge_index_ori = epsilon_graph(torch.tensor(embeds, dtype=torch.float), 0.95).to(device)
            print(f'ori, nodes={embeds.shape[0]}, deg={edge_index_ori.shape[1]/embeds.shape[0]}')
            grain = Grain(torch.tensor(embeds, dtype=torch.float), edge_index_ori, num_coreset=n_cluster)
            phi, n_cluster2 = grain.get_phi()
            print(n_cluster2)
        elif args.compress == 'Loukas':
            edge_index_ori = epsilon_graph(torch.tensor(embeds, dtype=torch.float), epsilon).to(device)
            print(f'ori, nodes={embeds.shape[0]}, deg={edge_index_ori.shape[1]/embeds.shape[0]}')
            loukas = Loukas(torch.tensor(embeds, dtype=torch.float), edge_index_ori, n_cluster)
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

    # GNN
    gnn_model = GNN(hidden_size, hidden_size_gnn, 1, 'GraphSAGE').to(device)

    if not os.path.exists(f'model/{args.method}_{args.dataset}_5_gnn.pkl') or update:
        if args.method == 'ECSeq':  # compress graph
            best_epoch1 = train_gnn(lstm_model, gnn_model, com_graph)
        elif args.method == 'ECSeq2':  # inference graph
            best_epoch1 = train_gnn2(lstm_model, gnn_model, com_graph, embeds, labels, embeds_val, labels_val, embeds_test, labels_test)
        elif args.method == 'ECSeq3':  # compress graph + inference graph
            best_epoch1 = train_gnn3(lstm_model, gnn_model, com_graph, embeds, labels)
        elif args.method == 'ECSeq4':  # compress graph pretrain + inference graph finetune
            best_epoch1 = train_gnn4(lstm_model, gnn_model, com_graph, embeds, labels)
        elif args.method == 'fullGNN':  # full graph
            best_epoch1, degree = train_gnn_full(lstm_model, gnn_model, embeds, labels, embeds_val, labels_val, embeds_test, labels_test)

    gnn_model.load_state_dict(torch.load(f'model/{args.method}_{args.dataset}_{t}_gnn.pkl'))
    print('GNN load succsessfully.')

    if args.ensemble:
        # Ensemble
        ens_model = Ensemble(hidden_size + hidden_size_gnn, 2).to(device)
        # if not os.path.exists(f'model/{args.method}_{args.dataset}_5_ens.pkl') or update:
        #     best_epoch2 = train_ensemble(lstm_model, gnn_model, ens_model, X_com)
        # ens_model.load_state_dict(torch.load(f'model/{args.method}_{args.dataset}_{t}_ens.pkl'))
        # print('ENS load succsessfully.')
        # # evaluate
        best_epoch2 = 0
        (auc0, f10, rp0), (auc1, f11, rp1), (auc2, f12, rp2) = eval_all(lstm_model, gnn_model, ens_model, X_com, test_loader)
    
    else:
        auc1, f11, rp1 = eval_gnn(lstm_model, gnn_model, X_com, embeds_test, labels_test)
        auc0, f10, rp0, best_epoch2, auc2, f12, rp2 = 0, 0, 0, 0, 0, 0, 0

    return auc0, f10, rp0, best_epoch1, auc1, f11, rp1, best_epoch2, auc2, f12, rp2, degree


if __name__ == '__main__':
    seed_everything(42)
    hidden_size = 256
    hidden_size_gnn = 32
    layer_num = 1
    batch_size = 64
    gnn_batch_size = 1000

    # gnn_lr = 0.0005
    gnn_lr = 0.005  # ECSeq
    # gnn_lr = 0.01  # fullGNN
    ens_lr = 0.0001

    ens_epo = 50
    gnn_epo = 300
    update = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='HK')  # HK, LZD
    parser.add_argument("--compress", type=str, default='kmeans_ba')  # kmeans_ba, kmeans_no, AGC, Grain, Loukas
    parser.add_argument("--method", type=str, default='ECSeq2') 
    parser.add_argument("--backbone", type=str, default='transformer')  # lstm
    parser.add_argument("--ensemble", type=bool, default=True) 
    parser.add_argument("--ratio", type=float, default=1.0) 
    args = parser.parse_args()

    n_cluster = 1000

    # epsilon = 0.98 if args.dataset == 'LZD' else 0.99 
    # epsilon = 0.95 if args.dataset == 'LZD' else 0.99  
    epsilon = 0.98 

    with open('params.json', 'r', encoding='utf-8') as f:
        params = json.load(f)
        n_pos = params[args.dataset]["n_pos"]
        n_neg = params[args.dataset]["n_neg"]
    
    n_pos = int(n_pos*args.ratio)
    n_neg = int(n_neg*args.ratio)
       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.backbone == 'transformer':
        epsilon = 0.99
        hidden_size = 32
        def collate_fn(batch):
            inputs, labels, lengths, ends, idxs = zip(*batch)
            inputs_pad = pad_sequence(inputs, batch_first=True)
            return inputs_pad.float().to(device), torch.LongTensor(labels).to(device), torch.LongTensor(lengths), \
                torch.stack(ends, dim=0).float().to(device), torch.LongTensor(idxs).to(device)
        train_loader, train_loader2, val_loader, test_loader, input_size = get_dataset(args, batch_size, device, collate_fn=collate_fn)
    else:
        train_loader, train_loader2, val_loader, test_loader, input_size = get_dataset(args, batch_size, device)

    loss_ce = torch.nn.NLLLoss()
    loss_mse = torch.nn.MSELoss()

    results = []
    t0 = time()
    for t in range(1,6):
        print('times', t)
        res = run()
        results.append(res)
        with open('results.txt', 'a') as f:
            f.write(f'{args.method} final, time: {time() - t0}, {args.dataset}, deg={res[11]:.4f}, '
                    f'[lstm] auc={res[0]:.4f}, f1={res[1]:.4f}, rp={res[2]:.4f}, '
                    f'[gnn] epoch={res[3]}, auc={res[4]:.4f}, f1={res[5]:.4f}, rp={res[6]:.4f}, '
                    f'[ens] epoch={res[7]}, auc={res[8]:.4f}, f1={res[9]:.4f}, rp={res[10]:.4f}\n')
            f.flush()
    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    with open('results.txt', 'a') as f:
        f.write(f'{args.method} final avg, time: {time() - t0}, {args.dataset}, deg={mean[11]:.4f}, '
                f'[lstm] {mean[0]:.4f}±{std[0]:.4f}, {mean[1]:.4f}±{std[1]:.4f}, {mean[2]:.4f}±{std[2]:.4f}, '
                f'[gnn] epoch={mean[3]}, {mean[4]:.4f}±{std[4]:.4f}, {mean[5]:.4f}±{std[5]:.4f}, {mean[6]:.4f}±{std[6]:.4f}, '
                f'[ens] epoch={mean[7]}, {mean[8]:.4f}±{std[8]:.4f}, {mean[9]:.4f}±{std[9]:.4f}, {mean[10]:.4f}±{std[10]:.4f}\n')
        f.flush()
    
    
    print(f'{args.method} final avg, time: {time() - t0}, {args.dataset}, deg={mean[11]:.4f}, '
          f'[lstm] {mean[0]:.4f}±{std[0]:.4f}, {mean[1]:.4f}±{std[1]:.4f}, {mean[2]:.4f}±{std[2]:.4f}, '
          f'[gnn] epoch={mean[3]}, {mean[4]:.4f}±{std[4]:.4f}, {mean[5]:.4f}±{std[5]:.4f}, {mean[6]:.4f}±{std[6]:.4f}, '
          f'[ens] epoch={mean[7]}, {mean[8]:.4f}±{std[8]:.4f}, {mean[9]:.4f}±{std[9]:.4f}, {mean[10]:.4f}±{std[10]:.4f}')
