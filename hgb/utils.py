import os
import sys
import gc
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse import remove_diag, set_diag

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not installed, will use t-SNE instead. Install with: pip install umap-learn")
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../data')
from data_loader import data_loader
# import src.utils_lib

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

#选择最佳元路径集合
def project_op(keys, model, criterion, eval_loader, device, trainval_point, valtest_point, labels, repeat):  #
    ''' operation '''
    # num_ops = self.num_sampled
    # candidate_flags = model.candidate_flags[cell_type]
    # proj_crit = args.proj_crit[cell_type]
    compare = lambda x, y: x < y   ######################attention########################
    crit_extrema = None
    best_index = 0
    for opid in range(repeat): #重复次数repeat
        index_sampled = model.epoch_sample(0)  #采样
        crit = infer_eval(model, criterion, eval_loader, device, index_sampled, trainval_point, valtest_point, labels)  #weights=weights  计算采样到的元路径集合损失
        print("loss {}\n".format(crit))
        if crit_extrema is None or compare(crit, crit_extrema):
            crit_extrema = crit
            best_index = index_sampled
    path = []
    label_path = []
    for i, index in enumerate(best_index):
        if index < len(keys):
            path.append(keys[index])
        # else:
        #     label_path.append((label_keys[index - len(keys)], i))
    # import code
    # code.interact(local=locals())
    return [path, label_path]  #返回最佳元路径集合


def infer_eval(model, criterion, eval_loader, device, index_sampled, trainval_point, valtest_point, labels):
    # objs = src.utils_lib.utils.AvgrageMeter()
    # top1 = src.utils_lib.utils.AvgrageMeter()
    model.eval() #模型评估模式 dropout和batch normalization层将按照评估模式运行，而不是训练模式
    raw_preds = []
    meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats) if i in index_sampled]
    label_meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats,model.num_paths) if i in index_sampled]
    with torch.no_grad():#无需梯度计算

        for batch, batch_feats, batch_labels_feats, batch_mask in eval_loader:
            # # batch = batch.to(device)
            batch_feats = {k: x.to(device) for k, x in batch_feats.items() if
                           k in meta_path_sampled or (model.residual and k == model.tgt_key)}
            batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items() if k in label_meta_path_sampled}

            raw_preds.append(model(index_sampled, batch_feats, batch_labels_feats, meta_path_sampled,
                                   label_meta_path_sampled).cpu())  # id_paths
        # logits = model(data, weights_dict=weights_dict)
    raw_preds = torch.cat(raw_preds, dim=0)
    loss_train = criterion(raw_preds[:trainval_point], labels[:trainval_point]).item()
    loss_val = criterion(raw_preds[trainval_point:valtest_point], labels[trainval_point:valtest_point]).item()


    return loss_train + loss_val #返回总损失



def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluator(gt, pred):
    gt = gt.cpu().squeeze()
    pred = pred.cpu().squeeze()
    return f1_score(gt, pred, average='micro'), f1_score(gt, pred, average='macro')


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path, dataset_name=''):
    """绘制混淆矩阵"""
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(range(num_classes)),
                yticklabels=list(range(num_classes)))
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_accuracy_curves(train_accs, val_accs, test_accs, save_path, dataset_name=''):
    """绘制准确率曲线"""
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [acc[0]*100 for acc in train_accs], 'b-', label='Train Micro-F1', linewidth=2)
    plt.plot(epochs, [acc[0]*100 for acc in val_accs], 'r-', label='Val Micro-F1', linewidth=2)
    plt.plot(epochs, [acc[0]*100 for acc in test_accs], 'g-', label='Test Micro-F1', linewidth=2)
    plt.plot(epochs, [acc[1]*100 for acc in train_accs], 'b--', label='Train Macro-F1', linewidth=2)
    plt.plot(epochs, [acc[1]*100 for acc in val_accs], 'r--', label='Val Macro-F1', linewidth=2)
    plt.plot(epochs, [acc[1]*100 for acc in test_accs], 'g--', label='Test Macro-F1', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Training Progress - {dataset_name}', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy curves saved to {save_path}")


def plot_class_performance(y_true, y_pred, num_classes, save_path, dataset_name=''):
    """绘制每个类别的分类性能"""
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    report = classification_report(y_true, y_pred, output_dict=True, 
                                   labels=list(range(num_classes)), zero_division=0)
    
    classes = []
    precision = []
    recall = []
    f1 = []
    
    for i in range(num_classes):
        if str(i) in report:
            classes.append(f'Class {i}')
            precision.append(report[str(i)]['precision'] * 100)
            recall.append(report[str(i)]['recall'] * 100)
            f1.append(report[str(i)]['f1-score'] * 100)
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'Per-Class Performance - {dataset_name}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class performance plot saved to {save_path}")


def save_classification_report(y_true, y_pred, num_classes, save_path, dataset_name=''):
    """保存分类报告到文件"""
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    report = classification_report(y_true, y_pred, 
                                   labels=list(range(num_classes)), 
                                   zero_division=0)
    
    with open(save_path, 'w') as f:
        f.write(f"Classification Report - {dataset_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    
    print(f"Classification report saved to {save_path}")


def plot_embedding_visualization(embeddings, labels, save_path, dataset_name='', method='tsne', n_components=2,
                                 random_state=42):
    """
    绘制节点嵌入的降维可视化图（类似论文中的t-SNE/UMAP可视化）
    """
    embeddings = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    # 如果是多维标签，转换为单维
    if len(labels.shape) > 1:
        if labels.shape[1] > 1:
            labels = labels.argmax(axis=1)
        else:
            labels = labels.squeeze()

    print(f"Reducing {embeddings.shape[0]} nodes from {embeddings.shape[1]}D to {n_components}D using {method}...")

    # 降维
    if method.lower() == 'umap' and HAS_UMAP:
        reducer = UMAP(n_components=n_components, random_state=random_state, n_neighbors=15, min_dist=0.1)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        if embeddings.shape[1] > 50:
            print(f"First reducing to 50D using PCA...")
            pca = PCA(n_components=50, random_state=random_state)
            embeddings = pca.fit_transform(embeddings)

        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=30, n_iter=1000)
        embeddings_2d = reducer.fit_transform(embeddings)

    # 获取唯一标签和对应的颜色
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # 使用seaborn的调色板
    if num_classes <= 10:
        colors = sns.color_palette("husl", num_classes)
    else:
        colors = sns.color_palette("tab20", num_classes)

    # 绘制散点图 - 关键修改：调整点的大小、透明度和边框
    plt.figure(figsize=(12, 9))  # 增大画布尺寸

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[colors[i]],
                    label=f'Class {int(label)}',
                    alpha=0.8,  # 提高透明度，让点更明显
                    s=15,  # 减小点的大小，可以显示更多点
                    edgecolors='none',  # 去掉边框，减少视觉干扰
                    linewidths=0)  # 边框宽度设为0

    plt.title(f'Node Embedding Visualization ({method.upper()}) - {dataset_name}', fontsize=14, fontweight='bold')
    plt.xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Dimension 2', fontsize=12)

    # 优化图例
    if num_classes <= 15:
        plt.legend(loc='best', fontsize=9, ncol=2, framealpha=0.8, markerscale=1.5)
    else:
        # 如果类别太多，可以省略图例或放在图外
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=1.2)

    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Embedding visualization saved to {save_path}")


def visualize_results(y_true, y_pred, num_classes, save_dir, dataset_name='', 
                     train_accs=None, val_accs=None, test_accs=None,
                     embeddings=None, embedding_labels=None):
    """生成所有可视化结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 混淆矩阵
    cm_path = os.path.join(save_dir, f'confusion_matrix_{dataset_name}.png')
    plot_confusion_matrix(y_true, y_pred, num_classes, cm_path, dataset_name)
    
    # 每个类别的性能
    class_perf_path = os.path.join(save_dir, f'class_performance_{dataset_name}.png')
    plot_class_performance(y_true, y_pred, num_classes, class_perf_path, dataset_name)
    
    # 分类报告
    report_path = os.path.join(save_dir, f'classification_report_{dataset_name}.txt')
    save_classification_report(y_true, y_pred, num_classes, report_path, dataset_name)
    
    # 准确率曲线（如果有历史数据）
    if train_accs and val_accs and test_accs:
        acc_curve_path = os.path.join(save_dir, f'accuracy_curves_{dataset_name}.png')
        plot_accuracy_curves(train_accs, val_accs, test_accs, acc_curve_path, dataset_name)
    
    # 节点嵌入可视化（如果有嵌入向量）
    if embeddings is not None and embedding_labels is not None:
        # t-SNE可视化
        tsne_path = os.path.join(save_dir, f'embedding_tsne_{dataset_name}.png')
        plot_embedding_visualization(embeddings, embedding_labels, tsne_path, dataset_name, method='tsne')
        
        # UMAP可视化（如果可用）
        if HAS_UMAP:
            umap_path = os.path.join(save_dir, f'embedding_umap_{dataset_name}.png')
            plot_embedding_visualization(embeddings, embedding_labels, umap_path, dataset_name, method='umap')


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def hg_propagate_feat_dgl(g, tgt_type, num_hops, max_length, echo=False):
    for hop in range(1, max_length):
        #reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]

        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)
            # if hop == args.num_hops and dtype != tgt_type: continue
            for k in list(g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and dtype != tgt_type ) \
                      or (hop > num_hops):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

        # remove no-use items
        for ntype in g.ntypes:
            if ntype == tgt_type: continue
            removes = []
            for k in g.nodes[ntype].data.keys():
                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo: print(f'-- hop={hop} ---')
        for ntype in g.ntypes:
            for k, v in g.nodes[ntype].data.items():
                print(f'{ntype} {k} {v.shape}', v[:,-1].max(), v[:,-1].mean())

        if echo: print(f'------\n')

    return g


def hg_propagate_feat_dgl_path(g, tgt_type, num_hops, max_length, meta_path, echo=False):
    for hop in range(1, max_length):
        #reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]

        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)

            for k in list(g.nodes[stype].data.keys()):
                if len(k) == hop:
                    # if hop == max_length - 1:
                    #     import code
                    #     code.interact(local=locals())
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and dtype != tgt_type ) \
                      or (hop > num_hops):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

        # remove no-use items
        for ntype in g.ntypes:

            if ntype == tgt_type: continue
            removes = []
            for k in g.nodes[ntype].data.keys():

                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo:
            print(f'-- hop={hop} ---')
            for ntype in g.ntypes:
                for k, v in g.nodes[ntype].data.items():
                    print(f'{ntype} {k} {v.shape}', v[:,-1].max(), v[:,-1].mean())

        if echo: print(f'------\n')

    return g


def hg_propagate_sparse_pyg(adjs, tgt_types, num_hops, max_length, extra_metapath, prop_feats=False, echo=False, prop_device='cpu'):
    store_device = 'cpu'
    if type(tgt_types) is not list:
        tgt_types = [tgt_types]

    label_feats = {k: v.clone() for k, v in adjs.items() if prop_feats or k[-1] in tgt_types} # metapath should start with target type in label propagation
    adjs_g = {k: v.to(prop_device) for k, v in adjs.items()}

    for hop in range(2, max_length):
        reserve_heads = [ele[-(hop+1):] for ele in extra_metapath if len(ele) > hop]
        new_adjs = {}
        for rtype_r, adj_r in label_feats.items():
            metapath_types = list(rtype_r)
            if len(metapath_types) == hop:
                dtype_r, stype_r = metapath_types[0], metapath_types[-1]
                for rtype_l, adj_l in adjs_g.items():
                    dtype_l, stype_l = rtype_l
                    if stype_l == dtype_r:
                        name = f'{dtype_l}{rtype_r}'
                        if (hop == num_hops and dtype_l not in tgt_types and name not in reserve_heads) \
                          or (hop > num_hops and name not in reserve_heads):
                            continue
                        if name not in new_adjs:
                            if echo: print('Generating ...', name)
                            if prop_device == 'cpu':
                                new_adjs[name] = adj_l.matmul(adj_r)
                            else:
                                with torch.no_grad():
                                    new_adjs[name] = adj_l.matmul(adj_r.to(prop_device)).to(store_device)
                        else:
                            if echo: print(f'Warning: {name} already exists')
        label_feats.update(new_adjs)

        removes = []
        for k in label_feats.keys():
            metapath_types = list(k)
            if metapath_types[0] in tgt_types: continue  # metapath should end with target type in label propagation
            if len(metapath_types) <= hop:
                removes.append(k)
        for k in removes:
            label_feats.pop(k)
        if echo and len(removes): print('remove', removes)
        del new_adjs
        gc.collect()

    if prop_device != 'cpu':
        del adjs_g
        torch.cuda.empty_cache()

    return label_feats


def check_acc(preds_dict, condition, init_labels, train_nid, val_nid, test_nid, show_test=True, loss_type='ce'):
    mask_train, mask_val, mask_test = [], [], []
    remove_label_keys = []
    k = list(preds_dict.keys())[0]
    v = preds_dict[k]
    if loss_type == 'ce':
        na, nb, nc = len(train_nid), len(val_nid), len(test_nid)
    elif loss_type == 'bce':
        na, nb, nc = len(train_nid) * v.size(1), len(val_nid) * v.size(1), len(test_nid) * v.size(1)

    for k, v in preds_dict.items():
        if loss_type == 'ce':
            pred = v.argmax(1)
        elif loss_type == 'bce':
            pred = (v > 0).int()

        a, b, c = pred[train_nid] == init_labels[train_nid], \
                  pred[val_nid] == init_labels[val_nid], \
                  pred[test_nid] == init_labels[test_nid]
        ra, rb, rc = a.sum() / na, b.sum() / nb, c.sum() / nc

        if loss_type == 'ce':
            vv = torch.log(v / (v.sum(1, keepdim=True) + 1e-6) + 1e-6)
            la, lb, lc = F.nll_loss(vv[train_nid], init_labels[train_nid]), \
                         F.nll_loss(vv[val_nid], init_labels[val_nid]), \
                         F.nll_loss(vv[test_nid], init_labels[test_nid])
        else:
            vv = (v / 2. + 0.5).clamp(1e-6, 1-1e-6)
            la, lb, lc = F.binary_cross_entropy(vv[train_nid], init_labels[train_nid].float()), \
                         F.binary_cross_entropy(vv[val_nid], init_labels[val_nid].float()), \
                         F.binary_cross_entropy(vv[test_nid], init_labels[test_nid].float())
        if condition(ra, rb, rc, k):
            mask_train.append(a)
            mask_val.append(b)
            mask_test.append(c)
        else:
            remove_label_keys.append(k)
        # if show_test:
        #     print(k, ra, rb, rc, la, lb, lc, (ra/rb-1)*100, (ra/rc-1)*100, (1-la/lb)*100, (1-la/lc)*100)
        # else:
        #     print(k, ra, rb, la, lb, (ra/rb-1)*100, (1-la/lb)*100)
    print(set(list(preds_dict.keys())) - set(remove_label_keys))

    print((torch.stack(mask_train, dim=0).sum(0) > 0).sum() / na)
    print((torch.stack(mask_val, dim=0).sum(0) > 0).sum() / nb)
    if show_test:
        print((torch.stack(mask_test, dim=0).sum(0) > 0).sum() / nc)


def train_multi_stage(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, enhance_loader, evaluator, predict_prob, gama, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    loss_l1, loss_l2 = 0., 0.
    iter_num = 0
    y_true, y_pred = [], []

    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        idx = torch.cat((idx_1, idx_2), dim=0)
        L1_ratio = len(idx_1) * 1.0 / (len(idx_1) + len(idx_2))
        L2_ratio = len(idx_2) * 1.0 / (len(idx_1) + len(idx_2))

        if isinstance(feats, list):
            batch_feats = [x[idx].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[idx].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[idx].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[idx].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
        batch_y = labels_cuda[idx_1]
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            extra_weight = 2 * torch.abs(predict_prob[idx_2] - 0.5)
            extra_y = (predict_prob[idx_2] > 0.5).float()
        else:
            extra_weight, extra_y = predict_prob[idx_2].max(dim=1)
        extra_weight = extra_weight.to(device)
        extra_y = extra_y.to(device)

        # teacher_soft = predict_prob[idx_2].to(device)
        # teacher_conf = torch.max(teacher_soft, dim=1, keepdim=True)[0]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(None, batch_feats, batch_labels_feats, batch_mask)
                L1 = loss_fcn(output_att[:len(idx_1)], batch_y)
                if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
                    L2 = F.binary_cross_entropy_with_logits(output_att[len(idx_1):], extra_y, reduction='none')
                else:
                    L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
                L2 = (L2 * extra_weight).sum() / len(idx_2)
                loss_train = L1_ratio * L1 + gama * L2_ratio * L2
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(None, batch_feats, batch_labels_feats, batch_mask)
            L1 = loss_fcn(output_att[:len(idx_1)], batch_y)
            if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
                L2 = F.binary_cross_entropy_with_logits(output_att[len(idx_1):], extra_y, reduction='none')
            else:
                L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
            L2 = (L2 * extra_weight).sum() / len(idx_2)
            loss_train = L1_ratio * L1 + gama * L2_ratio * L2
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att[:len(idx_1)].data.cpu() > 0.).int())
        else:
            y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        loss_l1 += L1.item()
        loss_l2 += L2.item()
        iter_num += 1

    print(loss_l1 / iter_num, loss_l2 / iter_num)
    loss = total_loss / iter_num
    approx_acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, approx_acc


def train(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []

    for batch in train_loader:
        ## batch = batch.to(device)
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch, batch_feats, batch_labels_feats, batch_mask)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(batch, batch_feats, batch_labels_feats, batch_mask)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc



def train_search(model, feats, label_feats, labels_cuda, loss_fcn, optimizer_w, optimizer_a, train_loader, val_loader, epoch_sampled, evaluator, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []
    val_total_loss = 0
    val_y_true, val_y_pred = [], []
    ###################  optimize w  ##################
    for batch in train_loader:
        # batch = batch.to(device)
        val_batch = next(iter(val_loader)).to(device)
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
            val_batch_feats = [x[val_batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            val_batch_feats = {k: x[val_batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        val_batch_labels_feats = {k: x[val_batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
            val_batch_mask = {k: x[val_batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
            val_batch_mask = None
        batch_y = labels_cuda[batch]
        val_batch_y = labels_cuda[val_batch]

        ########################################train
        optimizer_w.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch, batch_feats, epoch_sampled, batch_labels_feats, batch_mask)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward(retain_graph=True)
            scalar.step(optimizer_w)
            scalar.update()
        else:
            output_att = model(batch, batch_feats, epoch_sampled, batch_labels_feats, batch_mask)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward(retain_graph=True)
            optimizer_w.step()

        ########################################val  update a
        optimizer_a.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                val_output_att = model(val_batch, val_batch_feats, epoch_sampled, val_batch_labels_feats, val_batch_mask)
                val_loss_train = loss_fcn(val_output_att, val_batch_y)
            scalar.scale(val_loss_train).backward()
            scalar.step(optimizer_a)
            scalar.update()
        else:
            val_output_att = model(val_batch, val_batch_feats, epoch_sampled, val_batch_labels_feats, val_batch_mask)
            L1 = loss_fcn(val_output_att, val_batch_y)
            val_loss_train = L1
            val_loss_train.backward()
            optimizer_a.step()

        ########################################
        y_true.append(batch_y.cpu().to(torch.long))
        val_y_true.append(val_batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
            val_y_pred.append((val_output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            val_y_pred.append(val_output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        val_total_loss += val_loss_train.item()
        iter_num += 1


    loss_train = total_loss / iter_num
    acc_train = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    loss_val = val_total_loss / iter_num
    acc_val = evaluator(torch.cat(val_y_true, dim=0), torch.cat(val_y_pred, dim=0))

    return loss_train, loss_val, acc_train, acc_val


def train_search_new(model, feats, label_feats, labels_cuda, loss_fcn, optimizer_w, optimizer_a,train_loader, val_loader,
                     epoch_sampled, meta_path_sampled, label_meta_path_sampled, evaluator, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []
    val_total_loss = 0
    val_y_true, val_y_pred = [], []


    ###################  optimize w  ##################
    for batch in train_loader:
        # batch = batch.to(device)
        val_batch = next(iter(val_loader))
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
            val_batch_feats = [x[val_batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            # import code
            # code.interact(local=locals())
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            val_batch_feats = {k: x[val_batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        val_batch_labels_feats = {k: x[val_batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
            val_batch_mask = {k: x[val_batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
            val_batch_mask = None
        batch_y = labels_cuda[batch]
        val_batch_y = labels_cuda[val_batch]

        ########################################train
        optimizer_w.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(epoch_sampled, batch_feats, batch_labels_feats, meta_path_sampled, label_meta_path_sampled)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward(retain_graph=True)
            scalar.step(optimizer_w)
            scalar.update()
        else:
            output_att = model(epoch_sampled, batch_feats, batch_labels_feats, meta_path_sampled, label_meta_path_sampled)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward(retain_graph=True)
            optimizer_w.step()

        ########################################val  update a
        optimizer_a.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                val_output_att = model(epoch_sampled, val_batch_feats, val_batch_labels_feats, meta_path_sampled, label_meta_path_sampled)
                val_loss_train = loss_fcn(val_output_att, val_batch_y)
            scalar.scale(val_loss_train).backward()
            scalar.step(optimizer_a)
            scalar.update()
        else:
            val_output_att = model(epoch_sampled, val_batch_feats, val_batch_labels_feats, meta_path_sampled, label_meta_path_sampled)
            L1 = loss_fcn(val_output_att, val_batch_y)
            val_loss_train = L1
            val_loss_train.backward()
            optimizer_a.step()

        ########################################
        y_true.append(batch_y.cpu().to(torch.long))
        val_y_true.append(val_batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
            val_y_pred.append((val_output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            val_y_pred.append(val_output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        val_total_loss += val_loss_train.item()
        iter_num += 1


    loss_train = total_loss / iter_num
    acc_train = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    loss_val = val_total_loss / iter_num
    acc_val = evaluator(torch.cat(val_y_true, dim=0), torch.cat(val_y_pred, dim=0))

    return loss_train, loss_val, acc_train, acc_val





"""def train_search_two(model, feats, label_feats, labels_cuda, loss_fcn, optimizer_w, optimizer_a, train_loader, val_loader, meta_sampled, label_sampled, evaluator, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []
    val_total_loss = 0
    val_y_true, val_y_pred = [], []
    ###################  optimize w  ##################
    for batch in train_loader:
        # batch = batch.to(device)
        val_batch = next(iter(val_loader))
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
            val_batch_feats = [x[val_batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            val_batch_feats = {k: x[val_batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        val_batch_labels_feats = {k: x[val_batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
            val_batch_mask = {k: x[val_batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
            val_batch_mask = None
        batch_y = labels_cuda[batch]
        val_batch_y = labels_cuda[val_batch]

        ########################################train
        optimizer_w.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch, batch_feats, meta_sampled, label_sampled, batch_labels_feats, batch_mask)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward(retain_graph=True)
            scalar.step(optimizer_w)
            scalar.update()
        else:
            output_att = model(batch, batch_feats, meta_sampled, label_sampled, batch_labels_feats, batch_mask)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward(retain_graph=True)
            optimizer_w.step()

        ########################################val  update a
        optimizer_a.zero_grad()
        # if scalar is not None:
        #     with torch.cuda.amp.autocast():
        #         val_output_att = model(val_batch, val_batch_feats, val_batch_labels_feats, val_batch_mask)
        #         loss_train = loss_fcn(val_output_att, val_batch_y)
        #     scalar.scale(loss_train).backward()
        #     scalar.step(optimizer_a)
        #     scalar.update()
        # else:
        val_output_att = model(val_batch, val_batch_feats, meta_sampled, label_sampled, val_batch_labels_feats, val_batch_mask)
        L1 = loss_fcn(val_output_att, val_batch_y)
        val_loss_train = L1
        val_loss_train.backward()
        optimizer_a.step()

        ########################################
        y_true.append(batch_y.cpu().to(torch.long))
        val_y_true.append(val_batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
            val_y_pred.append((val_output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            val_y_pred.append(val_output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        val_total_loss += val_loss_train.item()
        iter_num += 1


    loss_train = total_loss / iter_num
    acc_train = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    loss_val = val_total_loss / iter_num
    acc_val = evaluator(torch.cat(val_y_true, dim=0), torch.cat(val_y_pred, dim=0))

    return loss_train, loss_val, acc_train, acc_val
"""


def train_flag(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, step_size, m, mask=None, scalar=None):

    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    # step_size = 1e-2
    # m = 3
    y_true, y_pred = [], []

    for batch in train_loader:
        # batch = batch.to(device)
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
        batch_y = labels_cuda[batch]


        def forward(k, perturb):

            #perturb_feats = {k: x+perturb[k] for k, x in batch_feats.items()}
            batch_feats[k] = batch_feats[k] + perturb

            #print (batch_feats[k][0][0:10])
            return model(batch, batch_feats, batch_labels_feats, batch_mask)
        model_forward = (model,forward)
        feats_shape = {k: x.shape for k, x in batch_feats.items()}
        if scalar is not None:
            with torch.cuda.amp.autocast():
                loss_train, output_att = flag(model_forward, feats_shape, batch_y, step_size, m, optimizer, device, F.nll_loss, scalar)
        else:
            loss, output_att = flag(model_forward, batch_feats.shape, batch_y, step_size, m, optimizer, device, F.nll_loss)


        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc






def train_2l(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, tgt_type, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []

    for batch in train_loader:
        # batch = batch.to(device)

        layer2_feats = {k: x[batch] for k, x in feats.items() if k[0] == tgt_type}
        batch_labels_feats = {k: x[batch] for k, x in label_feats.items()}

        involved_keys = {}
        for k, v in layer2_feats.items():
            src = k[-1]
            if src not in involved_keys:
                involved_keys[src] = []
            involved_keys[src].append(torch.unique(v.storage.col()))
        involved_keys = {k: torch.unique(torch.cat(v)) for k, v in involved_keys.items()}

        for k, v in layer2_feats.items():
            src = k[-1]
            old_nnz = v.nnz()
            layer2_feats[k] = v[:, involved_keys[src]]
            assert layer2_feats[k].nnz() == old_nnz

        layer1_feats = {k: v[involved_keys[k[0]]] for k, v in feats.items() if k[0] in involved_keys}

        batch1 = {k: v.to(device) for k,v in involved_keys.items()}
        layer1_feats = {k: v.to(device) for k,v in layer1_feats.items()}
        batch2 = batch.to(device)
        layer2_feats = {k: v.to(device) for k,v in layer2_feats.items()}
        batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(layer1_feats, batch1, layer2_feats, batch2, batch_labels_feats)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(layer1_feats, batch1, layer2_feats, batch2, batch_labels_feats)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc

def edge_mask(etypes,adjs,edge_mask_ratio):
    from collections import Counter
    etypes_ditc ={}

    generator = torch.Generator().manual_seed(1)

    for etype, adj in zip(etypes, adjs): 
        etypes_ditc[etype] = [adj,False]

    for key in etypes: 
        if not etypes_ditc[key][1]:
            etypes_ditc[key][1] = True
            row = etypes_ditc[key][0].storage._row
            col = etypes_ditc[key][0].storage._col

            row_list,col_list = row.cpu().tolist(),col.cpu().tolist()
            row_Counter = Counter(row_list)
            col_Counter = Counter(col_list)
            row_Counter = set([key for key in row_Counter.keys() if row_Counter[key] == 1])
            col_Counter = set([key for key in col_Counter.keys() if col_Counter[key] == 1])

            single_edge = torch.zeros(row.shape) == 0.0
            for index in range(len(row_list)):
                if row_list[index] in row_Counter or col_list[index] in col_Counter:
                    single_edge[index] = False

            edge_mask = torch.zeros(row.shape) != 0.0
            mid_mask = torch.rand((single_edge.sum().item()), generator=generator) < edge_mask_ratio
            edge_mask[single_edge] = mid_mask

            edge_mask = ~edge_mask

            row = row[edge_mask]
            col = col[edge_mask]

            etypes_ditc[key][0].storage._row,etypes_ditc[key][0].storage._col = row,col
            new_key = (key[2],key[1][::-1],key[0])
            if new_key in etypes_ditc and not etypes_ditc[new_key][1]:
                etypes_ditc[new_key][1] = True
                etypes_ditc[new_key][0].storage._row,etypes_ditc[new_key][0].storage._col = col,row


def load_dataset(args):
    dl = data_loader(f'{args.root}/{args.dataset}')

    edge_mask_ratio = args.edge_mask_ratio

    # use one-hot index vectors for nods with no attributes
    # === feats ===
    features_list = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features_list.append(torch.eye(dl.nodes['count'][i]))
        else:
            features_list.append(torch.FloatTensor(th))

    idx_shift = np.zeros(len(dl.nodes['count'])+1, dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        idx_shift[i+1] = idx_shift[i] + dl.nodes['count'][i]

    # === labels ===
    num_classes = dl.labels_train['num_classes']
    init_labels = np.zeros((dl.nodes['count'][0], num_classes), dtype=int)

    val_ratio = 0.2
    train_nid = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_nid)
    split = int(train_nid.shape[0]*val_ratio)
    val_nid = train_nid[:split]
    train_nid = train_nid[split:]
    train_nid = np.sort(train_nid)
    val_nid = np.sort(val_nid)
    test_nid = np.nonzero(dl.labels_test['mask'])[0]
    test_nid_full = np.nonzero(dl.labels_test_full['mask'])[0]

    init_labels[train_nid] = dl.labels_train['data'][train_nid]
    init_labels[val_nid] = dl.labels_train['data'][val_nid]
    init_labels[test_nid] = dl.labels_test['data'][test_nid]
    if args.dataset != 'IMDB':
        init_labels = init_labels.argmax(axis=1)

    print(len(train_nid), len(val_nid), len(test_nid), len(test_nid_full))
    init_labels = torch.LongTensor(init_labels)

    # === adjs ===
    # print(dl.nodes['attr'])
    # for k, v in dl.nodes['attr'].items():
    #     if v is None: print('none')
    #     else: print(v.shape)
    adjs = [] if args.dataset != 'Freebase' else {}
    for i, (k, v) in enumerate(dl.links['data'].items()):
        v = v.tocoo()
        src_type_idx = np.where(idx_shift > v.col[0])[0][0] - 1
        dst_type_idx = np.where(idx_shift > v.row[0])[0][0] - 1
        row = v.row - idx_shift[dst_type_idx]
        col = v.col - idx_shift[src_type_idx]
        sparse_sizes = (dl.nodes['count'][dst_type_idx], dl.nodes['count'][src_type_idx])
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sparse_sizes)
        if args.dataset == 'Freebase':
            name = f'{dst_type_idx}{src_type_idx}'
            assert name not in adjs
            adjs[name] = adj
        else:
            adjs.append(adj)
            #print(adj)


    if args.dataset == 'DBLP':
        # A* --- P --- T
        #        |
        #        V
        # author: [4057, 334]
        # paper : [14328, 4231]
        # term  : [7723, 50]
        # venue(conference) : None
        A, P, T, V = features_list
        AP, PA, PT, PV, TP, VP = adjs

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('P', 'P-A', 'A'),
            ('A', 'A-P', 'P'),
            ('T', 'T-P', 'P'),
            ('V', 'V-P', 'P'),
            ('P', 'P-T', 'T'),
            ('P', 'P-V', 'V'),
        ]
        if edge_mask_ratio != 0:
            edge_mask(etypes,adjs,edge_mask_ratio)


                    
        
        
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
        g = dgl.heterograph(new_edges)

        # for i, etype in enumerate(g.etypes):
        #     src, dst, eid = g._graph.edges(i)
        #     adj = SparseTensor(row=dst.long(), col=src.long())
        #     print(etype, adj)

        # g.ndata['feat']['A'] = A # not work
        g.nodes['A'].data['A'] = A
        g.nodes['P'].data['P'] = P
        g.nodes['T'].data['T'] = T
        g.nodes['V'].data['V'] = V
    elif args.dataset == 'IMDB':
        # A --- M* --- D
        #       |
        #       K
        # movie    : [4932, 3489]
        # director : [2393, 3341]
        # actor    : [6124, 3341]
        # keywords : None
        M, D, A, K = features_list
        MD, DM, MA, AM, MK, KM = adjs
        assert torch.all(DM.storage.col() == MD.t().storage.col())
        assert torch.all(AM.storage.col() == MA.t().storage.col())
        assert torch.all(KM.storage.col() == MK.t().storage.col())

        assert torch.all(MD.storage.rowcount() == 1) # each movie has single director

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('D', 'D-M', 'M'),
            ('M', 'M-D', 'D'),
            ('A', 'A-M', 'M'),
            ('M', 'M-A', 'A'),
            ('K', 'K-M', 'M'),
            ('M', 'M-K', 'K'),
        ]
        if edge_mask_ratio != 0:
            edge_mask(etypes,adjs,edge_mask_ratio)
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
        g = dgl.heterograph(new_edges)

        g.nodes['M'].data['M'] = M
        g.nodes['D'].data['D'] = D
        g.nodes['A'].data['A'] = A
        if args.num_hops > 2 :#or args.two_layer:
            g.nodes['K'].data['K'] = K
    elif args.dataset == 'ACM':
        # A --- P* --- C
        #       |
        #       K
        # paper     : [3025, 1902]
        # author    : [5959, 1902]
        # conference: [56, 1902]
        # field     : None
        P, A, C, K = features_list
        PP, PP_r, PA, AP, PC, CP, PK, KP = adjs
        row, col = torch.where(P)
        assert torch.all(row == PK.storage.row()) and torch.all(col == PK.storage.col())
        assert torch.all(AP.matmul(PK).to_dense() == A)
        assert torch.all(CP.matmul(PK).to_dense() == C)

        assert torch.all(PA.storage.col() == AP.t().storage.col())
        assert torch.all(PC.storage.col() == CP.t().storage.col())
        assert torch.all(PK.storage.col() == KP.t().storage.col())

        row0, col0, _ = PP.coo()
        row1, col1, _ = PP_r.coo()
        PP = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=PP.sparse_sizes())
        PP = PP.coalesce()
        PP = PP.set_diag()
        adjs = [PP] + adjs[2:]

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('P', 'P-P', 'P'),
            ('A', 'A-P', 'P'),
            ('P', 'P-A', 'A'),
            ('C', 'C-P', 'P'),
            ('P', 'P-C', 'C'),
        ]

        if edge_mask_ratio != 0:
            edge_mask(etypes,adjs,edge_mask_ratio)
        if args.ACM_keep_F:
            etypes += [
                ('K', 'K-P', 'P'),
                ('P', 'P-K', 'K'),
            ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)

        g = dgl.heterograph(new_edges)

        g.nodes['P'].data['P'] = P # [3025, 1902]
        g.nodes['A'].data['A'] = A # [5959, 1902]
        g.nodes['C'].data['C'] = C # [56, 1902]
        if args.ACM_keep_F:
            g.nodes['K'].data['K'] = K # [1902, 1902]
    elif args.dataset == 'Freebase':
        # 0*: 40402  2/4/7 <-- 0 <-- 0/1/3/5/6
        #  1: 19427  all <-- 1
        #  2: 82351  4/6/7 <-- 2 <-- 0/1/2/3/5
        #  3: 1025   0/2/4/6/7 <-- 3 <-- 1/3/5
        #  4: 17641  4 <-- all
        #  5: 9368   0/2/3/4/6/7 <-- 5 <-- 1/5
        #  6: 2731   0/4 <-- 6 <-- 1/2/3/5/6/7
        #  7: 7153   4/6 <-- 7 <-- 0/1/2/3/5/7
        for i in range(8):
            kk = str(i)
            print(f'==={kk}===')
            for k, v in adjs.items():
                t, s = k
                assert s == t or f'{s}{t}' not in adjs
                if s == kk or t == kk:
                    if s == t:
                        print(k, v.sizes(), v.nnz(),
                              f'symmetric {v.is_symmetric()}; selfloop-ratio: {v.get_diag().sum()}/{v.size(0)}')
                    else:
                        print(k, v.sizes(), v.nnz())

        adjs['00'] = adjs['00'].to_symmetric()
        g = None
    else:
        assert 0

    if args.dataset == 'DBLP':
        adjs = {'AP': AP, 'PA': PA, 'PT': PT, 'PV': PV, 'TP': TP, 'VP': VP}
    elif args.dataset == 'ACM':
        adjs = {'PP': PP, 'PA': PA, 'AP': AP, 'PC': PC, 'CP': CP}
    elif args.dataset == 'IMDB':
        adjs = {'MD': MD, 'DM': DM, 'MA': MA, 'AM': AM, 'MK': MK, 'KM': KM}
    elif args.dataset == 'Freebase':
        new_adjs = {}
        for rtype, adj in adjs.items():
            dtype, stype = rtype
            if dtype != stype:
                new_name = f'{stype}{dtype}'
                assert new_name not in adjs
                new_adjs[new_name] = adj.t()
        adjs.update(new_adjs)
    else:
        assert 0

    return g, adjs, init_labels, num_classes, dl, train_nid, val_nid, test_nid, test_nid_full


class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
