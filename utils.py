import numpy as np
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, asin, sqrt
from scipy.stats import pearsonr


class STTPMetric:
    def __init__(self, scalar_min, scalar_max, thres=1.):
        self.scalar_min = scalar_min
        self.scalar_max = scalar_max
        self.thres = thres

    def get_rmse_smape(self, labels, probs):
        labels = labels * (self.scalar_max - self.scalar_min) + self.scalar_min
        probs = probs * (self.scalar_max - self.scalar_min) + self.scalar_min
        rmse = mean_squared_error(labels, probs, squared=False)
        labels[labels < self.thres] = self.thres
        probs[probs < self.thres] = self.thres
        smape = (2 * np.abs(labels - probs) / (labels + probs)).mean()
        return rmse, smape


def get_auc_ap_rp(labels, probs):
    if np.isnan(probs).sum() > 0:
        print('nan', np.isnan(probs).sum())
        probs[np.isnan(probs)] = 0
        
    auc = roc_auc_score(labels, probs)

    ap = average_precision_score(labels, probs)

    precision, recall, _ = precision_recall_curve(labels, probs)
    rp = recall[precision>=0.9][0]
    return auc, ap, rp

    
def kNN_graph(X, k):
    X_norm = F.normalize(X, p=2, dim=1)
    sim = X_norm @ X_norm.T

    _, indices = torch.topk(sim, k + 1, dim=0, largest=False, sorted=True)
    row = torch.cat([indices[1:, i] for i in range(len(X))], dim=0)
    col = torch.tensor([i for i in range(len(X)) for _ in range(k)], dtype=torch.long, device=row.device)
    edge_index = torch.stack([row, col], dim=0)
    edge_index = torch.cat([edge_index, edge_index[[1,0], :]], dim=1)

    return edge_index


def knn_graph_add(X, X_new, num, k):
    X_norm = F.normalize(X, p=2, dim=1)
    X_new_norm = F.normalize(X_new, p=2, dim=1)
    sim = X_norm @ X_new_norm.T

    _, indices = torch.topk(sim, k + 1, dim=0, largest=False, sorted=True)
    row = torch.cat([indices[1:, i] for i in range(len(X_new))], dim=0)
    col = torch.tensor([i for i in range(len(X_new)) for _ in range(k)], dtype=torch.long, device=row.device) + num
    edge_index = torch.stack([row, col], dim=0)
    edge_index = torch.cat([edge_index, edge_index[[1,0], :]], dim=1)
    return edge_index


def epsilon_graph(X, epsilon):
    X_norm = F.normalize(X, p=2, dim=1)
    sim = X_norm @ X_norm.T
    edge_index = torch.argwhere(sim > epsilon).T
    return edge_index


def epsilon_graph_add(X, X_new, num, epsilon):
    X_norm = F.normalize(X, p=2, dim=1)
    X_new_norm = F.normalize(X_new, p=2, dim=1)
    sim = X_norm @ X_new_norm.T

    new_edge_index = torch.argwhere(sim > epsilon).T
    new_edge_index[1, :] = new_edge_index[1, :] + num
    # new_edge_index = torch.cat([new_edge_index, new_edge_index[[1,0], :]], dim=1)
    return new_edge_index


def get_neigh(all_edge_index, num1, num2, num_hops=2, num_neighs=10):
    row, col = all_edge_index
    node_mask = row.new_empty(num1 + num2, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    subsets = [torch.tensor([num1], device=row.device)]
    batch1 = [torch.tensor([num1], device=row.device)]
    batch2 = []
    for _ in range(num_hops):
        for target_node in torch.cat(batch1).unique().tolist():
            neighs = col[row == target_node]
            if neighs.size(0) > num_neighs:
                idx = torch.randint(0, neighs.size(0), (num_neighs,))
                neighs = neighs[idx]
            batch2.append(neighs)
        subsets += batch2
        batch1 = batch2.copy()
        batch2 = []

    subset = torch.cat(subsets).unique()
    node_mask.fill_(False)
    node_mask[subset] = True

    edge_mask = node_mask[row] & node_mask[col]
    edge_index = edge_index[:, edge_mask]
    return edge_index


def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000


def distance_edge(X1, X2, threshold=1000):
    edges = []
    for i in range(len(X1)):
        for j in range(len(X2)):
            if haversine(X1[i][0], X1[i][1], X2[j][0], X2[j][1]) <= threshold:
                edges.append((i, j))
    return torch.tensor(edges).T.long()


def correlation_edge(X1, X2, threshold):
    X1 = X1.cpu().numpy()
    X2 = X2.cpu().numpy()
    edges = []
    for i in range(len(X1)):
        for j in range(len(X2)):
            if pearsonr(X1[i], X2[j])[0] >= threshold:
                edges.append((i, j))
    return torch.tensor(edges).T.long()


class EncodedDataset0(Dataset):
    def __init__(self, df):
        super().__init__()
        drop_features = ['target_event_id', 'label', 'target_gmt_occur_cn', 'event_id', 'gmt_occur_cn', 'rn']
        id_feature = 'target_event_id'
        label_feature = 'label'
        self.data = []
        self.label = []
        self.length = []
        df_group = df.groupby([id_feature])
        with tqdm(df_group, desc='loading data...') as loop:
            for _, frame in loop:
                frame.sort_values(by=['rn'], ascending=False, inplace=True)
                self.label.append(frame[label_feature].iloc[-1])
                x = torch.from_numpy(frame.drop(drop_features, axis=1).to_numpy())
                self.data.append(x)
                self.length.append(len(frame))
        print(len(self.label), np.sum(self.label))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item], self.length[item], self.data[item][-1, :], item
    
    

def get_dataset(args, batch_size, device, collate_fn='default', EncodedDataset='default'):
    if EncodedDataset == 'default':
        EncodedDataset = EncodedDataset0
    
    df_train = pd.read_csv('data/' + args.dataset + '_train.csv')
    df_val = pd.read_csv('data/' + args.dataset + '_val.csv')
    df_test = pd.read_csv('data/' + args.dataset + '_test.csv')

    train_dataset = EncodedDataset(df_train)
    val_dataset = EncodedDataset(df_val)
    test_dataset = EncodedDataset(df_test)
    
    input_size = df_train.shape[1] - 6

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    if collate_fn == 'default':
        def collate_fn0(batch):
            inputs, labels, lengths, ends, idxs = zip(*batch)
            inputs_pad = pack_sequence(inputs, enforce_sorted=False)
            return inputs_pad.float().to(device), torch.LongTensor(labels).to(device), torch.LongTensor(lengths).to(device), \
                torch.stack(ends, dim=0).float().to(device), torch.LongTensor(idxs).to(device)
        collate_fn = collate_fn0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    train_loader2 = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, train_loader2, val_loader, test_loader, input_size

