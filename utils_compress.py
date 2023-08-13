from sklearn.cluster import KMeans
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
import scipy
import numpy.linalg as la
from graph_coarsening.coarsening_utils import coarsen
import pygsp


def kmeans_no(X, n_cluster):
    C = KMeans(n_clusters=n_cluster, random_state=42, n_init='auto').fit_predict(X)
    onehot = OneHotEncoder(sparse_output=False)
    return onehot.fit_transform(C.reshape(-1, 1))


def kmeans_ba(X, Y, n_pos, n_neg):
    idx_pos = np.argwhere(Y==1).flatten()
    idx_neg = np.argwhere(Y==0).flatten()
    X_pos = X[idx_pos, :]
    X_neg = X[idx_neg, :]
    # print(X_pos.shape, X_neg.shape)
    C_pos = KMeans(n_clusters=n_pos, random_state=42, n_init='auto').fit_predict(X_pos)
    C_neg = KMeans(n_clusters=n_neg, random_state=42, n_init='auto').fit_predict(X_neg)

    phi = np.zeros((X.shape[0], n_pos + n_neg))
    for i in range(n_pos):
        c_idx = idx_pos[C_pos == i]
        phi[c_idx, i] = 1
    for i in range(n_neg):
        c_idx = idx_neg[C_neg == i]
        phi[c_idx, i + n_pos] = 1
    return phi
 

class Loukas:
    def __init__(self, features, edge_index, new_node, r=None):
        num_node = features.shape[0]
        if r is None:
            self.r = 1.0 - new_node/num_node
        else:
            self.r = r
        row, col = edge_index.cpu()
        edge_attr = torch.ones(row.size(0))
        adj = scipy.sparse.coo_matrix((edge_attr.numpy(), (row.numpy(), col.numpy())), (num_node, num_node))
        self.G = pygsp.graphs.Graph(adj)

    def get_phi(self):
        C, _, _, _ = coarsen(self.G, r=self.r, method='variation_neighborhoods')
        return C.todense().T


class Grain:
    def __init__(self, features, edge_index, num_coreset, radium=0.05):
        self.num_node = features.shape[0]
        self.num_coreset = num_coreset
        self.radium = radium
        row, col = edge_index.cpu()
        edge_attr = torch.ones(row.size(0))
        self.adj = scipy.sparse.coo_matrix((edge_attr.numpy(), (row.numpy(), col.numpy())), (self.num_node, self.num_node))

        #compute and store normalized distance in A*A*X
        self.adj = self.aug_normalized_adjacency(self.adj)
        adj_matrix = torch.FloatTensor(self.adj.todense()).cuda()
        self.adj_matrix2 = torch.mm(adj_matrix,adj_matrix).cuda()
        features = features.cuda()
        self.features_aax = np.array(torch.mm(self.adj_matrix2,features).cpu())
        self.adj_matrix2 = np.array(self.adj_matrix2.cpu())

        self.distance_aax = np.zeros((self.num_node,self.num_node))
        for i in range(self.num_node-1):
            for j in range(i+1,self.num_node):
                self.distance_aax[i][j] = self.compute_distance(i,j)
                self.distance_aax[j][i] = self.distance_aax[i][j]
        dis_range = np.max(self.distance_aax) - np.min(self.distance_aax)
        self.distance_aax = (self.distance_aax - np.min(self.distance_aax))/dis_range

        #compute the balls
        balls = np.zeros((self.num_node,self.num_node))
        self.balls_dict=dict()
        # self.covered_balls = set()
        for i in range(self.num_node):
            for j in range(self.num_node):
                if self.distance_aax[i][j] <= radium:
                    balls[i][j]=1

        for node in range(self.num_node):
            neighbors_tmp = self.get_current_neighbors_dense([node])
            neighbors_tmp = neighbors_tmp[:,np.newaxis]
            dot_result = np.matmul(balls,neighbors_tmp).T
            tmp_set = set()
            for i in range(self.num_node):
                if dot_result[0,i]!=0:
                    tmp_set.add(i)
            self.balls_dict[node]=tmp_set
        print('initialize ok!')
        
    def aug_normalized_adjacency(self, adj):
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def get_current_neighbors_dense(self, cur_nodes):
        if np.array(cur_nodes).shape[0]==0:
            return np.ones(self.num_node)
        neighbors=(self.adj_matrix2[list(cur_nodes)].sum(axis=0)!=0)+0
        return neighbors

    def compute_distance(self, _i,_j):
        return la.norm(self.features_aax[_i,:]-self.features_aax[_j,:])

    def get_phi(self):
        count = 0
        idx_train = []
        covered_balls = set()
        idx_avaliable_tmp = list(range(self.num_node))
        while True:	
            ball_num_max = 0
            node_max = 0
            for node in idx_avaliable_tmp:
                tmp_num = len(covered_balls.union(self.balls_dict[node]))
                if tmp_num > ball_num_max:
                    ball_num_max = tmp_num
                    node_max = node
            res_ball_num = self.num_node - ball_num_max
            count+=1
            print('the number '+str(count)+' is selected, with the balls '+str(ball_num_max-len(covered_balls))+' covered and the rest balls is '+str(res_ball_num))	
            idx_train.append(node_max)
            idx_avaliable_tmp.remove(node_max)
            covered_balls = covered_balls.union(self.balls_dict[node_max])
            if count >= self.num_coreset or res_ball_num==0:
                break
        phi = np.zeros((self.num_node, len(idx_train)))
        for i,j in enumerate(idx_train):
            phi[j, i] = 1
        return phi, len(idx_train)


class AGC:
    def __init__(self, feature, edge_index, n_cluster, bestpower=None):
        row, col = edge_index.cpu()
        edge_attr = torch.ones(row.size(0))
        N = feature.shape[0]
        self.adj = scipy.sparse.coo_matrix((edge_attr.numpy(), (row.numpy(), col.numpy())), (N, N))
        self.feature = feature  #.detach().cpu().numpy()
        self.adj_normalized = self.preprocess_adj()
        self.n_cluster = n_cluster

        if bestpower is not None:
            self.bestpower = bestpower
        else:
            self.bestpower = 0
            intra_list = [100000]
            self.feature_power = self.feature
            for i in range(1, 61):
                self.feature_power = self.adj_normalized.dot(self.feature_power)
                u, s, v = sp.linalg.svds(self.feature_power, k=self.n_cluster, which='LM')
                intraD = []
                for j in range(10):
                    kmeans = KMeans(n_clusters=self.n_cluster, n_init='auto').fit(u)
                    predict_labels = kmeans.predict(u)
                    intraD.append(self.square_dist(predict_labels, self.feature_power))
                intramean = np.mean(intraD)
                intra_list.append(intramean)
                print('power: {}'.format(i), intramean)
                if intra_list[i] > intra_list[i - 1]:
                    print('bestpower: {}'.format(i - 1))
                    self.bestpower = i - 1
                    break

    def get_phi(self):
        self.feature_power = self.feature
        for _ in range(self.bestpower):
            self.feature_power = self.adj_normalized.dot(self.feature_power)
        u, s, v = sp.linalg.svds(self.feature_power, k=self.n_cluster, which='LM')
        kmeans = KMeans(n_clusters=self.n_cluster, n_init='auto').fit(u)
        C = kmeans.predict(u)
        onehot = OneHotEncoder(sparse_output=False)
        return onehot.fit_transform(C.reshape(-1, 1))

    def preprocess_adj(self, loop=True):
        adj = self.adj
        if loop:
            adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
        return (sp.eye(adj_normalized.shape[0]) + adj_normalized) / 2

    def to_onehot(self, prelabel):
        k = len(np.unique(prelabel))
        label = np.zeros([prelabel.shape[0], k])
        label[range(prelabel.shape[0]), prelabel] = 1
        label = label.T
        return label

    def square_dist(self, prelabel, feature):
        if sp.issparse(feature):
            feature = feature.todense()
        feature = np.array(feature)
        onehot = self.to_onehot(prelabel)

        m, n = onehot.shape
        count = onehot.sum(1).reshape(m, 1)
        count[count==0] = 1

        mean = onehot.dot(feature)/count
        a2 = (onehot.dot(feature*feature)/count).sum(1)
        pdist2 = np.array(a2 + a2.T - 2*mean.dot(mean.T))

        intra_dist = pdist2.trace()
        inter_dist = pdist2.sum() - intra_dist
        intra_dist /= m
        inter_dist /= m * (m - 1)
        return intra_dist
