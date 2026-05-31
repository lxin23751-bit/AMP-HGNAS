import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import numpy as np
import random
import dgl
import pickle
# from data_loader import data_loader  # 不在此处加载数据

class MetaPathSampler:
    """
    基于最大独立集的元路径采样器。
    """
    def __init__(self, coverage_dict, threshold=2.0, rounds=10):
        self.coverage_dict = coverage_dict
        self.threshold = threshold
        self.D, self.metapaths = compute_dependency_matrix(coverage_dict)
        self.adj = build_redundancy_graph(self.D, threshold)
        self.rounds = rounds
        self.sampling_results = self.multi_round_sampling()
        self.cur_round = 0

    def multi_round_sampling(self):
        results = []
        for _ in range(self.rounds):
            order = np.random.permutation(self.adj.shape[0])
            indep_set = set()
            nodes = set(order)
            while nodes:
                v = nodes.pop()
                indep_set.add(v)
                neighbors = set(np.where(self.adj[v] == 1)[0])
                nodes -= neighbors
            results.append(sorted(list(indep_set)))
        return results

    def sample(self):
        idxs = self.sampling_results[self.cur_round % self.rounds]
        self.cur_round += 1
        return idxs

class LMSPS_Se(nn.Module):
    def __init__(self, hidden, nclass, feat_keys, label_feat_keys, tgt_key, dropout, 
                 input_drop, device, num_final, residual=False, bns=False, data_size=None, num_sampled=1,
                 metapath_sampler=None):
        
        super(LMSPS_Se, self).__init__()
        assert data_size is not None, "data_size不能为空，请传入特征维度信息。"

        self.feat_keys = feat_keys
        self.label_feat_keys = label_feat_keys
        self.num_feats = len(feat_keys)
        self.all_meta_path = list(self.feat_keys) + list(self.label_feat_keys)
        self.num_sampled = num_sampled
        self.num_channels = self.num_sampled
        self.num_paths = len(self.all_meta_path)
        self.num_final = num_final
        self.num_res = self.num_paths - self.num_final  #剩余可采样个数
        self.tgt_key = tgt_key
        self.residual = residual

        # 新增可学习参数beta
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # 根据beta动态计算num_sampled
        self.num_sampled = int(self.num_paths * torch.sigmoid(self.beta))

        print("number of paths", len(feat_keys), len(label_feat_keys))

        self.embeding = nn.ParameterDict({})
        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(
                torch.Tensor(v, hidden).uniform_(-0.5, 0.5))

        if len(label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in label_feat_keys:
                self.labels_embeding[k] = nn.Parameter(
                    torch.Tensor(nclass, hidden).uniform_(-0.5, 0.5))

        self.lr_output = nn.Sequential(
            nn.Linear(hidden, nclass, bias=False),
            nn.BatchNorm1d(nclass, affine=bns, track_running_stats=bns)
        )

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

        # 初始化alpha
        alpha = np.random.rand(self.num_paths)  # 随机初始化
        for i in range(self.num_paths):
            alpha[i] = self.tent_map(alpha[i])  # 应用Tent混沌映射

        # 将alpha转换为PyTorch张量
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device).requires_grad_(True)
        print("self.alpha:",self.alpha)

        """self.alpha = torch.ones(self.num_paths).to(device)
        self.alpha.requires_grad_(True)"""

        if self.residual:
            self.res_fc = nn.Linear(hidden, hidden)

        self.metapath_sampler = metapath_sampler

        self.init_params()


    def init_params(self):

        gain = nn.init.calculate_gain("relu")
        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


    def alphas(self):
        alphas= [self.alpha]
        return alphas

    # Tent混沌映射函数
    def tent_map(self,x, a=0.5):
        if x < a:
            return (1 / a) * x
        else:
            return (1 - (1 / a) * (1 - x))

    def chaos_magnitude(self, magnitude=0.1):
        # 混沌扰动的幅度
        return magnitude

    def needs_chaos(self, patience=10, threshold=0.01):
        # 检查是否需要混沌扰动
        if self.best_loss is None or self.current_loss - self.best_loss > threshold:
            self.best_loss = self.current_loss
            self.not_improved_count = 0
        else:
            self.not_improved_count += 1
        return self.not_improved_count >= patience

    def epoch_sample(self, eps=None):
        """
        使用最大独立集采样器进行元路径采样。
        """
        if self.metapath_sampler is not None:
            sampled = self.metapath_sampler.sample()
            print(f"MetaPathSampler采样结果: {sampled}")
            return sampled
        else:
            # 兼容原有逻辑
            indices = torch.argsort(self.alpha, dim=-1, descending=True)[:self.num_sampled]
            sampled = sorted(list(indices.cpu().numpy()))
            print(f"原采样: {sampled}")
            return sampled

    def get_gumbel_prob(self,xins):  # get_gumbel_prob函数的输出，包含了每个元路径的采样概率
        while True:
            gumbels = -torch.empty_like(xins).exponential_().log()
            logits = (xins.log_softmax(dim=-1) + gumbels) / self.tau
            probs = nn.functional.softmax(logits, dim=-1)
            index = probs.max(-1, keepdim=True)[-1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if (
                    (torch.isinf(gumbels).any())
                    or (torch.isinf(probs).any())
                    or (torch.isnan(probs).any())
            ):
                continue
            else:
                break

        return probs

    def forward(self, epoch_sampled, feats_dict, label_feats_dict, meta_path_sampled, label_meta_path_sampled):
        #if self.training and self.needs_chaos():
            #self.alpha.data += self.tent_map(torch.rand_like(self.alpha)) * self.chaos_magnitude()
        self.num_sampled = int(self.num_paths * self.beta)

        if isinstance(feats_dict[meta_path_sampled[-1]], torch.Tensor):
            for k, v in feats_dict.items():
                feats_dict[k] = self.input_drop(v @ self.embeding[k])
            
        elif isinstance(feats_dict[meta_path_sampled[-1]], SparseTensor):
            for k, v in feats_dict.items():
                feats_dict[k] = self.input_drop(v @ self.embeding[k[-1]])
            
        else:
            assert 0

        for k, v in label_feats_dict.items():

            label_feats_dict[k] = self.input_drop(v @ self.labels_embeding[k])
            
        x = [feats_dict[k] for k in meta_path_sampled] + [label_feats_dict[k] for k in label_meta_path_sampled]
        x = torch.stack(x, dim=1) # [B, C, D]

        ws = [self.alpha[idx] for idx in epoch_sampled]

        ws = self.get_gumbel_prob(torch.stack(ws))  #ws参数是get_gumbel_prob函数的输出，它包含了每个元路径的采样概率
        x = torch.einsum('bcd,c->bd', x, ws)

        if self.residual:
            k = self.tgt_key
            tgt_feat = feats_dict[k]
            x = x + self.res_fc(tgt_feat)

        x = self.dropout(self.prelu(x))
        x = self.lr_output(x)
        
        return x

    def set_tau(self, tau):
        self.tau = tau

    def adjust_num_sampled(self, new_num_sampled):
        self.num_sampled = new_num_sampled
        print(f"Adjusted num_sampled to {self.num_sampled}")

    def sample(self, keys, label_keys, lam, topn, all_path=False):
        '''
        to sample one candidate edge type per link
        '''
        length = len(self.alpha)
        seq_softmax = None if self.alpha is None else F.softmax(self.alpha, dim=-1)
        max = torch.max(seq_softmax, dim=0).values
        min = torch.min(seq_softmax, dim=0).values
        threshold = lam * max + (1 - lam) * min


        _, idxl = torch.sort(seq_softmax, descending=True)  # descending为alse，升序，为True，降序

        idx = idxl[:self.num_sampled]

        if all_path:
            path = []
            label_path = []
            for i, index in enumerate(idxl):
                if index < len(keys):
                    path.append((keys[index], i))
                else:
                    label_path.append((label_keys[index - len(keys)], i))
            return [path, label_path], idx

        if topn:
            id_paths = idxl[:topn]
        else:
            id_paths = [k for k in range(length) if seq_softmax[k].item() >= threshold]
        path = [keys[i] for i in range(len(keys)) if i in id_paths]
        label_path = [label_keys[i] for i in range(len(label_keys)) if i+len(keys) in id_paths]

        return [path, label_path], idx

def compute_dependency_matrix(coverage_dict):
    metapaths = list(coverage_dict.keys())
    K = len(metapaths)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                D[i, j] = 0
                continue
            Ni, Ei = coverage_dict[metapaths[i]]['nodes'], coverage_dict[metapaths[i]]['edges']
            Nj, Ej = coverage_dict[metapaths[j]]['nodes'], coverage_dict[metapaths[j]]['edges']
            node_union = len(Ni | Nj)
            node_inter = len(Ni & Nj) or 1
            edge_union = len(Ei | Ej)
            edge_inter = len(Ei & Ej) or 1
            D[i, j] = (node_union / node_inter) * (edge_union / edge_inter)
    return D, metapaths

def build_redundancy_graph(D, threshold):
    K = D.shape[0]
    adj = np.zeros((K, K), dtype=int)
    for i in range(K):
        for j in range(i+1, K):
            if D[i, j] > threshold:
                adj[i, j] = adj[j, i] = 1
    return adj

def max_independent_set(adj):
    K = adj.shape[0]
    nodes = set(range(K))
    indep_set = set()
    while nodes:
        v = nodes.pop()
        indep_set.add(v)
        # 移除与v相邻的所有节点
        neighbors = set(np.where(adj[v] == 1)[0])
        nodes -= neighbors
    return indep_set

def multi_round_sampling(adj, rounds=10):
    results = []
    for _ in range(rounds):
        order = np.random.permutation(adj.shape[0])
        indep_set = set()
        nodes = set(order)
        while nodes:
            v = nodes.pop()
            indep_set.add(v)
            neighbors = set(np.where(adj[v] == 1)[0])
            nodes -= neighbors
        results.append(indep_set)
    return results

# 1. 定义元路径
archs = {
    "dblp": [
        ['APA', 'APV', 'APAP', 'APTP', 'APVP', 'APTPA', 'APTPV', 'APVPA', 'APVPV', 'APAPTP', 'APAPVP', 'APTPAP',
         'APTPTP', 'APVPTP', 'APAPAPT', 'APAPAPV', 'APAPTPA', 'APAPVPA', 'APTPTPT', 'APTPVPA', 'APTPVPV', 'APVPAPA',
         'APVPAPT', 'APVPTPA', 'APVPTPV', 'APVPVPV'], []],
    # 其他数据集同理
}

# 2. 字母到节点类型id的映射（根据你的数据集实际情况调整）
node_type_map = {'A': 0, 'P': 1, 'V': 2, 'T': 3}

def your_str2etype_list(mp_str, dl):
    """
    将如'APA'的字符串元路径转为data_loader的边类型id序列。
    """
    etype_list = []
    for i in range(len(mp_str) - 1):
        src = node_type_map[mp_str[i]]
        dst = node_type_map[mp_str[i+1]]
        etype = dl.get_edge_type((src, dst))
        etype_list.append(etype)
    return etype_list

def get_metapath_coverage(dl, metapath):
    """
    dl: data_loader对象
    metapath: 边类型id序列
    return: (set of节点, set of边)
    """
    meta_dict = dl.get_full_meta_path(meta=metapath)
    nodes = set()
    edges = set()
    for start, paths in meta_dict.items():
        for path in paths:
            nodes.update(path)
            for i in range(len(path)-1):
                edges.add((path[i], path[i+1]))
    return nodes, edges





