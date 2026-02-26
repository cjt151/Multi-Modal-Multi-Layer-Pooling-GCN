import os
import re
import logging
import numpy as np
import torch
from scipy.io import loadmat

# 数据路径配置（可按需修改）
dataset_config = {
    'NC': {
        'fc1': '/root/Project/data/ADNI/CU/FC1',
        'fc2': '/root/Project/data/ADNI/CU/FC2',
        'sc1': '/root/Project/data/ADNI/CU/SC1',
        'sc2': '/root/Project/data/ADNI/CU/SC2'
    },
    'MCI': {
        'fc1': '/root/Project/data/ADNI/MCI/FC1',
        'fc2': '/root/Project/data/ADNI/MCI/FC2',
        'sc1': '/root/Project/data/ADNI/MCI/SC1',
        'sc2': '/root/Project/data/ADNI/MCI/SC2'
    }
}


class DataLoaderHelper:
    def __init__(self, dataset_config, modalities=('sc1','fc1','sc2','fc2')):
        self.dataset_config = dataset_config
        self.modalities = modalities
        valid_modalities = ['sc1','fc1','sc2','fc2']
        if len(modalities) != 4 or any(m not in valid_modalities for m in modalities):
            raise ValueError("modalities must be ('sc1','fc1','sc2','fc2')")
        logging.info(f"数据加载器初始化：模态={self.modalities}")

    def load_fc_mat_data(self, folder_path, label):
        all_fc = []
        subj_ids = []
        if not os.path.isdir(folder_path):
            logging.warning(f"{folder_path} 不存在或不是目录")
            return np.array([]), np.array([]), []
        for fn in os.listdir(folder_path):
            if not fn.lower().endswith('.mat'):
                continue
            fp = os.path.join(folder_path, fn)
            try:
                mat = loadmat(fp)
            except Exception as e:
                logging.warning(f"loadmat 失败 {fp}: {e}")
                continue
            # 取第一个非内置变量
            keys = [k for k in mat.keys() if not k.startswith('__')]
            if not keys:
                continue
            arr = mat[keys[0]]
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                logging.warning(f"{fp} 变量不是方阵，跳过，shape={arr.shape}")
                continue
            all_fc.append(arr.flatten())
            m = re.search(r'\d+', fn)
            subj_ids.append(m.group(0) if m else os.path.splitext(fn)[0])
        if not all_fc:
            return np.array([]), np.array([]), []
        labels = np.full(len(all_fc), label, dtype=int)
        return np.vstack(all_fc), labels, subj_ids

    def load_all_data(self):
        fc1_all=[]; fc2_all=[]; sc1_all=[]; sc2_all=[]
        labels_all=[]
        for cat in ['NC','MCI']:
            lab = 0 if cat=='NC' else 1
            logging.info(f"加载类别 {cat}")
            fc1, labs1, ids1 = self.load_fc_mat_data(self.dataset_config[cat]['fc1'], lab)
            fc2, _, ids2 = self.load_fc_mat_data(self.dataset_config[cat]['fc2'], lab)
            sc1, _, ids3 = self.load_fc_mat_data(self.dataset_config[cat]['sc1'], lab)
            sc2, _, ids4 = self.load_fc_mat_data(self.dataset_config[cat]['sc2'], lab)
            if len(fc1)==0 or len(fc2)==0 or len(sc1)==0 or len(sc2)==0:
                logging.warning(f"{cat} 部分模态无数据，跳过该类")
                continue
            common = set(ids1) & set(ids2) & set(ids3) & set(ids4)
            if not common:
                logging.warning(f"{cat} 无共同受试者，跳过")
                continue
            idx1 = [i for i,idv in enumerate(ids1) if idv in common]
            idx2 = [ids2.index(idv) for idv in [ids1[i] for i in idx1]]
            idx3 = [ids3.index(idv) for idv in [ids1[i] for i in idx1]]
            idx4 = [ids4.index(idv) for idv in [ids1[i] for i in idx1]]
            fc1_sel = fc1[idx1]; fc2_sel = fc2[idx2]; sc1_sel = sc1[idx3]; sc2_sel = sc2[idx4]
            labels_sel = np.full(len(fc1_sel), lab, dtype=int)
            fc1_all.append(fc1_sel); fc2_all.append(fc2_sel); sc1_all.append(sc1_sel); sc2_all.append(sc2_sel); labels_all.append(labels_sel)
            logging.info(f"{cat} 加载并对齐样本数: {len(fc1_sel)}")
        if not fc1_all:
            raise RuntimeError("没有加载到任何样本")
        fc1_all = np.vstack(fc1_all); fc2_all = np.vstack(fc2_all); sc1_all = np.vstack(sc1_all); sc2_all = np.vstack(sc2_all)
        labels_all = np.hstack(labels_all)
        perm = np.random.permutation(len(labels_all))
        return fc1_all[perm], fc2_all[perm], sc1_all[perm], sc2_all[perm], labels_all[perm]


def spatial_dimensionality_reduction(matrix, target_size=64):
    if isinstance(matrix, torch.Tensor):
        arr = matrix.cpu().numpy()
    else:
        arr = matrix
    n = arr.shape[0]
    if n <= target_size:
        return arr if not isinstance(matrix, torch.Tensor) else torch.from_numpy(arr).float()
    idx = np.linspace(0, n-1, target_size, dtype=int)
    reduced = arr[np.ix_(idx, idx)]
    return reduced


def preprocess_data(fc1_data, fc2_data, sc1_data, sc2_data, labels, target_size=64):
    assert len(fc1_data)==len(fc2_data)==len(sc1_data)==len(sc2_data)==len(labels)
    orig_n = int(np.sqrt(fc1_data.shape[1]))
    logging.info(f"原矩阵边长: {orig_n}, 降到 {target_size}")
    fc1_r=[]; fc2_r=[]; sc1_r=[]; sc2_r=[]
    for i in range(len(labels)):
        a1 = fc1_data[i].reshape(orig_n, orig_n); a2 = fc2_data[i].reshape(orig_n, orig_n)
        b1 = sc1_data[i].reshape(orig_n, orig_n); b2 = sc2_data[i].reshape(orig_n, orig_n)
        fc1_r.append(spatial_dimensionality_reduction(a1, target_size).flatten())
        fc2_r.append(spatial_dimensionality_reduction(a2, target_size).flatten())
        sc1_r.append(spatial_dimensionality_reduction(b1, target_size).flatten())
        sc2_r.append(spatial_dimensionality_reduction(b2, target_size).flatten())
    fc1_tensor = torch.tensor(np.vstack(fc1_r)).float().reshape(-1, target_size, target_size)
    fc2_tensor = torch.tensor(np.vstack(fc2_r)).float().reshape(-1, target_size, target_size)
    sc1_tensor = torch.tensor(np.vstack(sc1_r)).float().reshape(-1, target_size, target_size)
    sc2_tensor = torch.tensor(np.vstack(sc2_r)).float().reshape(-1, target_size, target_size)
    fc1_tensor = torch.nan_to_num(fc1_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    fc2_tensor = torch.nan_to_num(fc2_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    sc1_tensor = torch.nan_to_num(sc1_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    sc2_tensor = torch.nan_to_num(sc2_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return fc1_tensor, fc2_tensor, sc1_tensor, sc2_tensor, label_tensor


def build_graph_dataset_from_matrices(fc1_tensor, fc2_tensor, sc1_tensor, sc2_tensor,
                                      threshold_prop=0.05,
                                      use_fc1=True, use_fc2=True, use_sc1=True, use_sc2=True,
                                      adj_source='sc_avg',
                                      feature_mode='concat',
                                      use_summary_if_too_large=True, summary_threshold=1024):
    B, N, _ = fc1_tensor.shape
    fc1_np = fc1_tensor.cpu().numpy()
    fc2_np = fc2_tensor.cpu().numpy()
    sc1_np = sc1_tensor.cpu().numpy()
    sc2_np = sc2_tensor.cpu().numpy()

    X_list = []
    A_list = []

    for i in range(B):
        f1 = fc1_np[i]; f2 = fc2_np[i]
        s1 = sc1_np[i]; s2 = sc2_np[i]
        fc_list = []
        sc_list = []
        if use_fc1:
            fc_list.append(f1)
        if use_fc2:
            fc_list.append(f2)
        if use_sc1:
            sc_list.append(s1)
        if use_sc2:
            sc_list.append(s2)
        if adj_source == 'sc_avg':
            sc = np.mean(np.stack(sc_list, axis=0), axis=0) if sc_list else np.zeros((N, N), dtype=float)
        elif adj_source == 'sc1':
            sc = s1.copy()
        elif adj_source == 'sc2':
            sc = s2.copy()
        elif adj_source == 'fc_avg':
            sc = np.mean(np.stack(fc_list, axis=0), axis=0) if fc_list else np.zeros((N, N), dtype=float)
        else:
            sc = np.mean(np.stack(sc_list, axis=0), axis=0) if sc_list else np.zeros((N, N), dtype=float)
        if feature_mode == 'concat':
            node_parts = []
            for m in fc_list:
                node_parts.append(m)
            for m in sc_list:
                node_parts.append(m)
            if node_parts:
                node_feats = np.concatenate(node_parts, axis=1)
            else:
                avg_fc = np.mean(np.stack([f1, f2], axis=0), axis=0)
                avg_sc = np.mean(np.stack([s1, s2], axis=0), axis=0)
                node_feats = np.concatenate([avg_fc, avg_sc], axis=1)
        else:
            summary_parts = []
            for m in fc_list:
                summary_parts.append(np.stack([m.mean(axis=1), m.std(axis=1)], axis=1))
            for m in sc_list:
                summary_parts.append(np.stack([m.mean(axis=1), m.std(axis=1)], axis=1))
            if summary_parts:
                node_feats = np.concatenate(summary_parts, axis=1)
            else:
                avg_fc = np.mean(np.stack([f1, f2], axis=0), axis=0)
                avg_sc = np.mean(np.stack([s1, s2], axis=0), axis=0)
                node_feats = np.concatenate([
                    avg_fc.mean(axis=1, keepdims=True), avg_fc.std(axis=1, keepdims=True),
                    avg_sc.mean(axis=1, keepdims=True), avg_sc.std(axis=1, keepdims=True)
                ], axis=1)
        A = sc.copy()
        if np.allclose(A, 0):
            if fc_list:
                A = np.abs(np.mean(np.stack(fc_list, axis=0), axis=0))
            else:
                A = np.abs(f1)
        A[A < 0] = 0.0
        flat = np.sort(A.ravel())[::-1]
        k = max(1, int(flat.size * threshold_prop))
        thr = flat[k-1] if k < flat.size else flat[-1]
        A_bin = (A >= thr).astype(float)
        A_bin = np.maximum(A_bin, A_bin.T)
        np.fill_diagonal(A_bin, 0.0)
        A_self = A_bin + np.eye(N, dtype=float)
        deg = A_self.sum(axis=1)
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        A_hat = D_inv_sqrt @ A_self @ D_inv_sqrt
        X_list.append(node_feats.astype(np.float32))
        A_list.append(A_hat.astype(np.float32))

    max_f = max(x.shape[1] for x in X_list)
    X_batch = np.zeros((B, N, max_f), dtype=np.float32)
    for i in range(B):
        fd = X_list[i].shape[1]
        X_batch[i, :, :fd] = X_list[i]
    A_batch = np.stack(A_list, axis=0)
    return torch.from_numpy(X_batch), torch.from_numpy(A_batch)
