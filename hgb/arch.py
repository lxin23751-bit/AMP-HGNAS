import pickle
from data_loader import data_loader
import random
import scipy.sparse as sp
import os
import numpy as np

archs1 = {
    "dblp": [
        [
            # 2-hop (length 3)
            'APA', 'APV', 'APT',
            # 3-hop (length 4)
            'APAP', 'APVP', 'APTP',
            # 4-hop (length 5)
            'APAPA', 'APAPV', 'APAPT',
            'APVPA', 'APVPV', 'APVPT',
            'APTPA', 'APTPV', 'APTPT',
            # 5-hop (length 6)
            'APAPAP', 'APAPVP', 'APAPTP',
            'APVPAP', 'APVPVP', 'APVPTP',
            'APTPAP', 'APTPVP', 'APTPTP'
        ], []],
    "imdb": [
        ['M', 'MA', 'MK', 'MAM', 'MDM', 'MKM', 'MAMK', 'MDMK', 'MKMD', 'MDMKM', 'MKMAM', 'MKMDM', 'MKMKM', 'MAMAMK',
         'MAMDMA', 'MAMDMK', 'MAMKMD', 'MAMKMK', 'MDMAMA', 'MDMAMD', 'MDMDMA', 'MDMDMK', 'MDMKMA', 'MKMAMA', 'MKMAMD',
         'MKMAMK', 'MKMDMA', 'MKMDMK', 'MKMKMD', 'MKMKMK'], []],
    "acm": [
        ['PPP', 'PAPP', 'PCPA', 'PCPP', 'PPPC', 'PPPP', 'PAPAP', 'PAPPP', 'PCPAP', 'PCPPA', 'PPAPA', 'PPAPC', 'PPAPP',
         'PAPAPA', 'PAPCPA', 'PAPPAP', 'PAPPCP', 'PAPPPP', 'PCPAPP', 'PCPCPP', 'PCPPAP', 'PCPPPP', 'PPAPAP', 'PPAPCP',
         'PPAPPA', 'PPAPPP', 'PPCPAP', 'PPPAPA', 'PPPCPA', 'PPPPPP'], []],
}

node_type_map = {'A': 0, 'P': 1, 'V': 2, 'T': 3}

def your_str2etype_list(mp_str, dl):
    etype_list = []
    for i in range(len(mp_str) - 1):
        src = node_type_map[mp_str[i]]
        dst = node_type_map[mp_str[i+1]]
        etype = dl.get_edge_type((src, dst))
        etype_list.append(etype)
    return etype_list

def get_metapath_coverage(dl, metapath, sample_ratio=0.5, random_seed=42):
    """
    只用部分节点（子图）统计元路径覆盖，极大降低内存消耗。
    nodes/edges 必须是全局节点ID和全局边！
    """
    random.seed(random_seed)
    all_nodes = list(range(dl.nodes['total']))
    sample_size = max(1, int(len(all_nodes) * sample_ratio))
    sample_nodes = random.sample(all_nodes, sample_size)
    ini = sp.eye(dl.nodes['total']).tocsr()
    ini = ini[sample_nodes, :]
    spmat = ini
    for x in metapath:
        spmat = spmat.dot(dl.links['data'][x]) if x >= 0 else spmat.dot(dl.links['data'][-x - 1].T)
    row_idx, col_idx = spmat.nonzero()
    # 断言 col_idx 不超出全局节点数
    if len(col_idx) > 0:
        assert max(col_idx) < dl.nodes['total'], f"col_idx 超出范围: max={max(col_idx)}, total={dl.nodes['total']}"
    # 断言 row_idx 不超出采样节点数
    if len(row_idx) > 0:
        assert max(row_idx) < len(sample_nodes), f"row_idx 超出范围: max={max(row_idx)}, sample_size={len(sample_nodes)}"
    # nodes: 全局ID
    nodes = set([int(sample_nodes[i]) for i in row_idx]) | set([int(j) for j in col_idx])
    # edges: (全局起点, 全局终点)
    edges = set((int(sample_nodes[i]), int(j)) for i, j in zip(row_idx, col_idx))
    return nodes, edges

if __name__ == '__main__':
    dl = data_loader('F:/LMSPS-O - 副本/data/DBLP')
    metapaths = archs1['dblp'][0]
    coverage_dict = {}
    filtered_metapaths = []
    for mp in metapaths:
        metapath = your_str2etype_list(mp, dl)
        nodes, edges = get_metapath_coverage(dl, metapath, sample_ratio=0.1)
        print(f"元路径 {mp} 覆盖的节点数: {len(nodes)}，边数: {len(edges)}", end='')
        if len(nodes) < 10 or len(edges) < 10:
            print("  [被过滤]")
            filtered_metapaths.append(mp)
            continue
        else:
            print()
        coverage_dict[mp] = {'nodes': nodes, 'edges': edges}
    print(f"被过滤掉的元路径: {filtered_metapaths}")
    COVERAGE_PATH = 'F:/LMSPS-O - 副本/hgb/coverage_dict.pkl'
    print('arch.py 当前工作目录:', os.getcwd())
    print('COVERAGE_PATH 即将保存到:', COVERAGE_PATH)
    with open(COVERAGE_PATH, 'wb') as f:
        pickle.dump(coverage_dict, f)
    print('COVERAGE_PATH 是否存在:', os.path.exists(COVERAGE_PATH))
    print("coverage_dict 已保存到", COVERAGE_PATH)
    print('当前工作目录:', os.getcwd())

import pickle

with open('F:/LMSPS-O - 副本/hgb/coverage_dict.pkl', 'rb') as f:
    coverage_dict = pickle.load(f)

for k, v in coverage_dict.items():
    assert len(v['nodes']) == len(set(v['nodes'])), f"{k} nodes 有重复"
    assert len(v['edges']) == len(set(v['edges'])), f"{k} edges 有重复"
    for n in v['nodes']:
        assert isinstance(n, int) and n >= 0
    for e in v['edges']:
        assert isinstance(e, tuple) and len(e) == 2


archs = {
    "dblp": [
        ['APA', 'APV', 'APAP', 'APTP', 'APVP', 'APTPA', 'APTPV', 'APVPA', 'APVPV', 'APAPTP', 'APAPVP', 'APTPAP',
         'APTPTP', 'APVPTP', 'APAPAPT', 'APAPAPV', 'APAPTPA', 'APAPVPA', 'APTPTPT', 'APTPVPA', 'APTPVPV', 'APVPAPA',
         'APVPAPT','APVPTPA', 'APVPTPV', 'APVPVPV'], []],
    "imdb": [
        ['M', 'MA', 'MK', 'MAM', 'MDM', 'MKM', 'MAMK', 'MDMK', 'MKMD', 'MDMKM', 'MKMAM', 'MKMDM', 'MKMKM', 'MAMAMK',
         'MAMDMA', 'MAMDMK', 'MAMKMD', 'MAMKMK', 'MDMAMA', 'MDMAMD', 'MDMDMA', 'MDMDMK', 'MDMKMA', 'MKMAMA', 'MKMAMD',
         'MKMAMK', 'MKMDMA', 'MKMDMK', 'MKMKMD', 'MKMKMK'], []],
    "acm": [
        ['PPP', 'PAPP', 'PCPA', 'PCPP', 'PPPC', 'PPPP', 'PAPAP', 'PAPPP', 'PCPAP', 'PCPPA', 'PPAPA', 'PPAPC', 'PPAPP',
         'PAPAPA', 'PAPCPA', 'PAPPAP', 'PAPPCP', 'PAPPPP', 'PCPAPP', 'PCPCPP', 'PCPPAP', 'PCPPPP', 'PPAPAP', 'PPAPCP',
         'PPAPPA', 'PPAPPP', 'PPCPAP', 'PPPAPA', 'PPPCPA', 'PPPPPP'], []],
}
