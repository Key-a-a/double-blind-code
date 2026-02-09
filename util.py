import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch


class DataLoader(object):

    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()


class StandardScaler():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        mask = (data == 0)
        data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    try:
        _, _, adj_mx = load_pickle(pkl_filename)
    except:
        adj_mx = load_pickle(pkl_filename)
    adj = [sym_adj(adj_mx), sym_adj(np.transpose(adj_mx))]
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    # === (1) 标准化通道0（pH）：保留现有 scaler 以便反标准化 ===
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    # === (2) 其余通道逐通道标准化（基于训练集统计），不影响 y ===
    if data['x_train'].shape[-1] > 1:
        # 训练集统计（展平成二维：样本*T*N, C）
        xtr = data['x_train']
        flat = xtr.reshape(-1, xtr.shape[-1])
        mu  = flat.mean(axis=0)
        std = flat.std(axis=0) + 1e-6
        # 只处理 c>=1 的通道（c==0 已由 scaler 处理）
        for category in ['train', 'val', 'test']:
            X = data['x_' + category]
            for c in range(1, X.shape[-1]):
                X[..., c] = (X[..., c] - mu[c]) / std[c]

    # === (3) 追加时间上下文通道：日/周 sin-cos（4小时采样 -> 日6步、周42步）===
    def _append_time_feats(X):
        # X: [S, T, N, C]
        S, T, N, C = X.shape
        t = np.arange(T, dtype=np.float32)[None, :, None]  # [1,T,1]

        daily = 6.0   # 24h / 4h
        weekly = 42.0 # 7d * 6

        sin_d = np.sin(2 * np.pi * t / daily)
        cos_d = np.cos(2 * np.pi * t / daily)
        sin_w = np.sin(2 * np.pi * t / weekly)
        cos_w = np.cos(2 * np.pi * t / weekly)

        # 广播到 [S,T,N,1]
        def _tile(x):
            return np.tile(x, (S, 1, N)).astype(np.float32)[..., None]

        feats = [ _tile(sin_d), _tile(cos_d), _tile(sin_w), _tile(cos_w) ]
        return np.concatenate([X] + feats, axis=-1)

    for category in ['train', 'val', 'test']:
        data['x_' + category] = _append_time_feats(data['x_' + category])

    # === (4) DataLoaders & 返回 ===
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader']   = DataLoader(data['x_val'],   data['y_val'],   valid_batch_size)
    data['test_loader']  = DataLoader(data['x_test'],  data['y_test'],  test_batch_size)
    data['scaler'] = scaler
    torch.Tensor(data['x_train'])
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse



def _kmeans_numpy(X, num_clusters, num_iters=50, seed=1234):
    """
    Simple, dependency-free KMeans on numpy arrays.
    X: [N, D]
    returns labels: [N], centroids: [K, D]
    """
    rng = np.random.RandomState(seed)
    N, D = X.shape
    if num_clusters <= 0 or num_clusters > N:
        raise ValueError(f"num_clusters must be in [1, N], got {num_clusters} with N={N}")
    # init: pick K unique points
    init_idx = rng.choice(N, size=num_clusters, replace=False)
    C = X[init_idx].copy()

    for _ in range(num_iters):
        # assign
        # (N,K)
        dist2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = dist2.argmin(axis=1)
        # update
        C_new = np.zeros_like(C)
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.any():
                C_new[k] = X[mask].mean(axis=0)
            else:
                # empty cluster: re-init to a random point
                C_new[k] = X[rng.randint(0, N)]
        # convergence check
        if np.allclose(C_new, C, rtol=0, atol=1e-6):
            C = C_new
            break
        C = C_new
    return labels.astype(np.int64), C.astype(np.float32)


def build_hyperedges_knn_kmeans(trainx, k_knn=10, num_clusters=10, seed=1234):
    """
    Build two incidence matrices:
      - H_loc_T: [E_loc=N, N] local hyperedges via KNN (one hyperedge per node)
      - H_reg_T: [E_reg=G, N] regional hyperedges via KMeans clusters
    trainx: numpy array [num_samples, seq_len, num_nodes, in_dim]
    """
    if isinstance(trainx, torch.Tensor):
        trainx = trainx.detach().cpu().numpy()
    if trainx.ndim != 4:
        raise ValueError(f"trainx must be 4D [S,T,N,C], got shape {trainx.shape}")
    S, T, N, C = trainx.shape
    # Node feature vectors for clustering: concat mean and std over (S,T)
    mean_feat = trainx.mean(axis=(0, 1))            # [N, C]
    std_feat = trainx.std(axis=(0, 1)) + 1e-6       # [N, C]
    X = np.concatenate([mean_feat, std_feat], axis=1).astype(np.float32)  # [N, 2C]

    # --- KNN local hyperedges ---
    # pairwise distances (N,N)
    dist2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(dist2, np.inf)
    k = int(min(max(k_knn, 1), N - 1))
    nn_idx = np.argpartition(dist2, kth=k-1, axis=1)[:, :k]  # [N, k] unordered
    H_loc_T = np.zeros((N, N), dtype=np.float32)             # [E_loc=N, N]
    for i in range(N):
        H_loc_T[i, i] = 1.0
        H_loc_T[i, nn_idx[i]] = 1.0

    # --- KMeans regional hyperedges ---
    G = int(min(max(num_clusters, 1), N))
    labels, _ = _kmeans_numpy(X, num_clusters=G, num_iters=50, seed=seed)
    H_reg_T = np.zeros((G, N), dtype=np.float32)             # [E_reg=G, N]
    for i in range(N):
        H_reg_T[labels[i], i] = 1.0

    return H_loc_T, H_reg_T


def save_hypergraph_knn_kmeans(save_path, trainx, k_knn=10, num_clusters=10, seed=1234):
    """Build and save hypergraph incidence matrices to a pickle."""
    H_loc_T, H_reg_T = build_hyperedges_knn_kmeans(trainx, k_knn=k_knn, num_clusters=num_clusters, seed=seed)
    obj = {
        "H_loc_T": H_loc_T,
        "H_reg_T": H_reg_T,
        "k_knn": int(k_knn),
        "num_clusters": int(num_clusters),
        "seed": int(seed),
    }
    with open(save_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return save_path


def load_hypergraph_knn_kmeans(hyper_pkl_path):
    """Load saved incidence matrices and return torch tensors H_a (local), H_b (regional)."""
    with open(hyper_pkl_path, "rb") as f:
        obj = pickle.load(f)
    H_loc_T = obj["H_loc_T"].astype(np.float32)  # [E_loc, N]
    H_reg_T = obj["H_reg_T"].astype(np.float32)  # [E_reg, N]
    H_a = torch.from_numpy(H_loc_T)  # local
    H_b = torch.from_numpy(H_reg_T)  # regional
    return H_a, H_b, obj


def load_hadj(pkl_filename, top_k, hypergraph_pkl=None):
    try:
        _, _, adj_mx = load_pickle(pkl_filename)
    except:
        adj_mx = load_pickle(pkl_filename)

    hadj = adj_mx

    # If a precomputed KNN+KMeans hypergraph incidence is provided, use it for H_a/H_b.
    # Otherwise, fall back to the original edge-derived hypergraph construction.
    use_precomputed_hg = hypergraph_pkl is not None and os.path.exists(hypergraph_pkl)


    top = top_k

    hadj = hadj - np.identity(hadj.shape[0])
    hadj = torch.from_numpy(hadj.astype(np.float32))
    _, idx = torch.topk(hadj, top, dim=0)
    _, idy = torch.topk(hadj, top, dim=1)

    base_mx_lie = torch.zeros([hadj.shape[0], hadj.shape[1]])
    for i in range(hadj.shape[0]):
        base_mx_lie[idx[:, i], i] = hadj[idx[:, i], i]
    base_mx_hang = torch.zeros([hadj.shape[0], hadj.shape[1]])
    for j in range(hadj.shape[0]):
        base_mx_hang[j, idy[j, :]] = hadj[j, idy[j, :]]

    base_mx = torch.where(base_mx_lie != 0, base_mx_lie, base_mx_hang)

    hadj = base_mx + torch.eye(hadj.shape[0])
    hadj = hadj.numpy()

    n = hadj.shape[0]
    l = int((len(np.nonzero(hadj)[0])))
    H = np.zeros((l, n))
    H_a = np.zeros((l, n))
    H_b = np.zeros((l, n))
    lwjl = np.zeros((l,1))
    a=0

    for i in range(hadj.shape[0]):
        for j in range(hadj.shape[1]):
            if(hadj[i][j]!=0.0):
                H[a, i] = 1.0
                H[a, j] = 1.0
                H_a[a, i] = 1.0
                H_b[a, j] = 1.0
                if(i==j):
                    lwjl[a, 0] = 1.0
                else:
                    lwjl[a,0] = adj_mx[i,j]
                a = a+1

    lwjl = 1.0-lwjl

    W = np.ones(n)

    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    DE_=np.power(DE, -1)
    DE_[np.isinf(DE_)] = 0.
    invDE = np.mat(np.diag(DE_))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    HT = sp.coo_matrix(HT)
    rowsum = np.array(HT.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    H_T_new = d_mat.dot(HT).astype(np.float32).todense()

    G0 = DV2 * H
    G1 = invDE * HT * DV2

    n = adj_mx.shape[0]
    l = int((len(np.nonzero(adj_mx)[0])))
    H_all = np.zeros((l, n))
    edge_1 = np.array([])
    edge_2 = np.array([])
    a=0

    for i in range(adj_mx.shape[0]):
        for j in range(adj_mx.shape[1]):
            if(adj_mx[i][j]!=0.0):
                H_all[a, i] = 1.0
                H_all[a, j] = 1.0
                edge_1 = np.hstack((edge_1, np.array([i])))
                edge_2 = np.hstack((edge_2, np.array([j])))
                a = a+1

    W_all = np.ones(n)
    DV_all = np.sum(H_all * W_all, axis=1)
    DE_all = np.sum(H_all, axis=0)

    DE__all=np.power(DE_all, -1)
    DE__all[np.isinf(DE__all)] = 0.
    invDE_all = np.mat(np.diag(DE__all))
    DV2_all = np.mat(np.diag(np.power(DV_all, -0.5)))
    W_all = np.mat(np.diag(W_all))
    H_all = np.mat(H_all)
    HT_all = H_all.T

    HT_all = sp.coo_matrix(HT_all)
    rowsum_all = np.array(HT_all.sum(1)).flatten()
    d_inv_all = np.power(rowsum_all, -1).flatten()
    d_inv_all[np.isinf(d_inv_all)] = 0.
    d_mat_all = sp.diags(d_inv_all)
    H_T_new_all = d_mat_all.dot(HT_all).astype(np.float32).todense()

    G0_all = DV2_all * H_all
    G1_all = invDE_all * HT_all * DV2_all

    coo_hadj = adj_mx - np.identity(n)
    coo_hadj = sp.coo_matrix(coo_hadj)
    coo_hadj = coo_hadj.tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack((edge_1, edge_2)).astype(np.int64))

    G0 = G0.astype(np.float32)
    G1 = G1.astype(np.float32)
    H = H.astype(np.float32)
    HT = H.T.astype(np.float32)
    H_T_new = torch.from_numpy(H_T_new.astype(np.float32))
    H_a = torch.from_numpy(H_a.astype(np.float32))
    H_b = torch.from_numpy(H_b.astype(np.float32))
    lwjl = torch.from_numpy(lwjl.astype(np.float32))

    G0_all = G0_all.astype(np.float32)
    G1_all = G1_all.astype(np.float32)

    # Override hypergraph incidence with KNN+KMeans-built hyperedges if provided.
    if use_precomputed_hg:
        H_a0, H_b0, meta = load_hypergraph_knn_kmeans(hypergraph_pkl)
        # Combined incidence H: [E, N]
        H_comb = torch.cat([H_a0, H_b0], dim=0).cpu().numpy().astype(np.float32)
        n = H_comb.shape[1]
        W = np.ones(n, dtype=np.float32)

        DV = np.sum(H_comb * W, axis=1)              # [E]
        DE = np.sum(H_comb, axis=0)                  # [N]
        DE_ = np.power(DE, -1)
        DE_[np.isinf(DE_)] = 0.
        invDE = np.mat(np.diag(DE_))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        DV2[np.isinf(DV2)] = 0.
        Hm = np.mat(H_comb)
        HTm = Hm.T                                   # [N, E]

        # Row-normalized HT for edge->node projection (same style as original code)
        HT_coo = sp.coo_matrix(HTm)
        rowsum = np.array(HT_coo.sum(1)).flatten()
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        H_T_new_np = d_mat.dot(HT_coo).astype(np.float32).todense()

        G0_np = (DV2 * Hm).astype(np.float32)
        G1_np = (invDE * HTm * DV2).astype(np.float32)

        # Move back to torch tensors matching return types
        H_a = H_a0.float()
        H_b = H_b0.float()
        HT = torch.from_numpy(np.array(HTm, dtype=np.float32))
        H_T_new = torch.from_numpy(np.array(H_T_new_np, dtype=np.float32))
        lwjl = torch.ones((H_comb.shape[0], 1), dtype=torch.float32)
        G0 = np.array(G0_np, dtype=np.float32)
        G1 = np.array(G1_np, dtype=np.float32)


    return H_a, H_b, HT, lwjl ,G0,G1,indices, G0_all,G1_all


def feature_node_to_edge(feature_node,H_a,H_b,operation="concat"):
    feature_edge_a = torch.einsum('ncvl,wv->ncwl', (feature_node, H_a))
    feature_edge_b = torch.einsum('ncvl,wv->ncwl', (feature_node, H_b))
    if operation == "concat":
        feature_edge = torch.cat([feature_edge_a, feature_edge_b], dim=1)
    elif  operation == "sum":
        feature_edge = feature_edge_a + feature_edge_b
    elif operation == "subtract":
        feature_edge = feature_edge_a - feature_edge_b
    return feature_edge


def fusion_edge_node(x, x_h, H_T_new):
    x_h_new = torch.einsum('ncvl,wv->ncwl', (x_h, H_T_new))
    x = torch.cat([x, x_h_new], dim=1)
    return x