import random, h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, csc_matrix
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import StepLR
from sklearn.neighbors import NearestNeighbors
class LazyCSR:
    def __init__(self, csr_mat: csr_matrix, dtype=torch.float32, device="cpu",
                 pin_memory=False, max_pin_bytes=256 * 1024 * 1024, verbose=False):
        assert isinstance(csr_mat, csr_matrix)
        self.csr = csr_mat.tocsr()
        self.shape = self.csr.shape
        self.dtype = dtype
        self.device = torch.device(device)
        self.pin_memory = bool(pin_memory)
        self.max_pin_bytes = int(max_pin_bytes)
        self.verbose = verbose

    def to(self, device):
        self.device = torch.device(device)
        return self

    def get_csr(self):
        return self.csr

    def __len__(self):
        return self.shape[0]

    def _maybe_pin(self, t: torch.Tensor) -> torch.Tensor:
        if (not self.pin_memory) or (self.device.type != "cuda") or (not torch.cuda.is_available()):
            return t
        size_bytes = t.numel() * t.element_size()
        if size_bytes > self.max_pin_bytes:
            return t
        if not t.is_contiguous():
            t = t.contiguous()
        try:
            return t.pin_memory()
        except RuntimeError as e:
            if self.verbose:
                print(f"[LazyCSR] pin_memory failed -> fallback. reason: {e}")
            return t

    def _rows_to_tensor(self, row_idx_1d: np.ndarray):
        sub = self.csr[row_idx_1d, :]
        dense = sub.toarray()
        t = torch.as_tensor(dense, dtype=self.dtype)
        t = self._maybe_pin(t)
        non_block = self.pin_memory and (self.device.type == "cuda") and torch.cuda.is_available()
        return t.to(self.device, non_blocking=non_block)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.detach().cpu().numpy()
        if isinstance(idx, slice):
            row_idx = np.arange(self.shape[0])[idx]
            return self._rows_to_tensor(row_idx.astype(np.int64))

        idx = np.asarray(idx)
        if idx.ndim == 1:
            return self._rows_to_tensor(idx.astype(np.int64))
        elif idx.ndim == 2:
            B, K = idx.shape
            flat = idx.astype(np.int64).ravel()
            t = self._rows_to_tensor(flat)
            return t.reshape(B, K, -1)
        else:
            raise IndexError("LazyCSR only supports 1D/2D row indexing.")

# -------------------------
# 设备
# -------------------------
def get_device(use_gpu=True, gpu_id=0):
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("CPU")
    return device
class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_attention_heads=4,
                 attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2):
        super(SelfAttention, self).__init__()
        self.AE = nn.Sequential(
            nn.Linear(input_size, hidden_size),
        )
        self.LN = nn.LayerNorm(hidden_size)
        self.relation_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=4 * hidden_size,
                activation='gelu',
                batch_first=True,
                dropout=0.2
            ),
            num_layers=1
        )
    def forward(self, input_tensor, return_attention=False, is_drop=False):
        input_tensor = self.AE(input_tensor)  # (B, H)
        hidden_states = self.relation_encoder(input_tensor.unsqueeze(1)).squeeze(1)
        hidden_states = self.LN(hidden_states + input_tensor)
        return hidden_states
class EncoderWithProjector(nn.Module):
    def __init__(self, input_size, hidden_size=128, proj_dim=128,
                 num_heads=4, attn_dropout=0.2, hidden_dropout=0.2):
        super().__init__()
        self.backbone = SelfAttention(input_size, hidden_size,
                                      num_attention_heads=num_heads,
                                      attention_probs_dropout_prob=attn_dropout,
                                      hidden_dropout_prob=hidden_dropout)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_size, proj_dim)
        )
    def forward(self, x, return_proj=True):
        h = self.backbone(x)
        if return_proj:
            z = F.normalize(self.projector(h), dim=1)
            return h, z
        else:
            return h
class ConstrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.mmd_loss = MMDLoss()
    def group_by_domain(self, domain_label, eta_counts):
        unique_domains = torch.unique(domain_label)
        domain_features_list = []
        for domain_id in unique_domains:
            mask = (domain_label == domain_id)
            domain_features = eta_counts[mask]  # shape: [n_i, gene]
            if len(domain_features) > 0:
                domain_features_list.append(domain_features)
        return domain_features_list
    def forward(self, z_i, z_j, h1, h2, domain_label):
        B = z_i.size(0)
        z = torch.cat((z_i, z_j), dim=0)                 # (2B, D)
        sim = torch.matmul(z, z.T) / self.temperature    # (2B, 2B)
        mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float('-inf'))
        targets = torch.cat([torch.arange(B, 2*B, device=z.device),
                             torch.arange(0, B, device=z.device)], dim=0)  # (2B,)
        sim = sim - sim.max(dim=1, keepdim=True)[0]
        mmd_total = 0.
        pairs = 0
        if domain_label is not None:
            domain_features_list_1 = self.group_by_domain(domain_label, h1)
            domin_num = len(domain_features_list_1)
            for i in range(domin_num):
                for j in range(domin_num):
                    if i != j:
                        mmd_total += self.mmd_loss(domain_features_list_1[i], domain_features_list_1[j])
                        pairs += 1
            return self.criterion(sim, targets) + 0.2 * (mmd_total/pairs)
        return self.criterion(sim, targets)

def data_augmentations(X, rate=0.4, lib_jitter=0.0, noise_std=0.0):
    X_aug = X.clone()
    rows, cols = torch.nonzero(X_aug > 0, as_tuple=True)
    if rows.numel() > 0 and rate > 0:
        k = int(rows.numel() * rate)
        idx = torch.randperm(rows.numel(), device=X.device)[:k]
        X_aug[rows[idx], cols[idx]] = 0.0
    if lib_jitter > 0:
        scales = (1.0 + lib_jitter * torch.randn(X.size(0), device=X.device)).clamp_min(0.1)
        X_aug = X_aug * scales.unsqueeze(1)
    if noise_std > 0:
        X_aug = X_aug + noise_std * torch.randn_like(X_aug)
        X_aug = torch.clamp(X_aug, min=0.0)
    return X_aug

def train_simclr(encoder: EncoderWithProjector, x_lazy: LazyCSR, device, batch_labels,
                 epochs=200, batch_size=512, drop_rate=0.4, lr=3e-3, weight_decay=5e-3,
                 temperature=0.1, patience=5):
    N = len(x_lazy)
    if N < batch_size:
        batch_size = N
    ds = TensorDataset(torch.arange(N))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        pin_memory=torch.cuda.is_available(), drop_last=True)

    opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = ConstrastiveLoss(temperature=temperature)

    best, bad = None, 0
    best_state = None

    encoder.train()
    for ep in range(epochs):
        tot, cnt = 0.0, 0
        for (idx_batch,) in loader:
            idx_batch = idx_batch.to(device, non_blocking=True)
            X = x_lazy[idx_batch.cpu().numpy()]  # (B, G)  log1p 归一域
            x1 = data_augmentations(X, rate=drop_rate)
            x2 = data_augmentations(X, rate=drop_rate)
            h1, z1 = encoder(x1, return_proj=True)
            h2, z2 = encoder(x2, return_proj=True)
            batch_label = batch_labels[idx_batch.detach().cpu().numpy()].to(
                device) if batch_labels is not None else None
            if batch_labels is not None:
                loss = criterion(z1, z2, h1, h2, batch_label)
            else:
                loss = criterion(z1, z2, None, None, None)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            opt.step()

            tot += loss.item() * X.size(0)
            cnt += X.size(0)

        avg = tot / max(cnt, 1)
        print(f"[SimCLR(SelfAttn)] Epoch {ep+1:03d} | Loss: {avg:.4f}")

        if (best is None) or (avg < best - 1e-4):
            best = avg
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k,v in encoder.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("[SimCLR] Early stop")
                break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder

@torch.no_grad()
def extract_embeddings(encoder: EncoderWithProjector, x_lazy: LazyCSR, device, bs=1024):
    N = len(x_lazy)
    embs = []
    for i in range(0, N, bs):
        rows = np.arange(i, min(i+bs, N))
        xb = x_lazy[rows]
        h = encoder(xb, return_proj=False)
        h = F.normalize(h, dim=1)
        embs.append(h.detach().cpu())
    return torch.cat(embs, dim=0).numpy()
@torch.no_grad()
def build_neighbor_pool(embeddings: np.ndarray, threshold=0.9, topk=30):
    N, _ = embeddings.shape
    nbrs = NearestNeighbors(n_neighbors=min(topk+1, max(2, N)), metric='cosine')
    nbrs.fit(embeddings)
    dist, idx = nbrs.kneighbors(embeddings)
    dist, idx = dist[:,1:], idx[:,1:]  # 去自身
    sim = 1.0 - dist
    mask = (sim >= threshold)
    idx_pool = np.full((N, topk), -1, dtype=np.int64)
    sim_pool = np.zeros((N, topk), dtype=np.float32)
    for i in range(N):
        cand = idx[i][mask[i]]
        s = sim[i][mask[i]]
        if cand.size > 0:
            order = np.argsort(-s)[:topk]
            take_idx = cand[order]
            take_sim = s[order]
            L = len(take_idx)
            idx_pool[i,:L] = take_idx
            sim_pool[i,:L] = take_sim
    return idx_pool, sim_pool

class ImputerModel(nn.Module):
    def __init__(self, frozen_encoder: EncoderWithProjector, n_genes, n_cells, batch_label,
                 neighbor_idx_pool: np.ndarray, cell_factors: np.ndarray, num_domains,
                 K_train=10, d_model=512, device = 'cuda'):
        super().__init__()
        self.encoder = frozen_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.K_train = K_train
        self.batch_label = batch_label
        self.register_buffer('neighbor_idx_pool', torch.from_numpy(neighbor_idx_pool.astype(np.int64)))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=4,
                                                   dim_feedforward=4*d_model,
                                                   batch_first=False,
                                                   dropout=0.1,
                                                   activation='gelu')
        self.decode = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.decoder = nn.Linear(d_model, n_genes)

    def _sample_neighbors(self, idx_batch: torch.Tensor):
        pool = self.neighbor_idx_pool
        B = idx_batch.size(0)
        K = self.K_train
        pos_idx = torch.empty((B, K), dtype=torch.long, device=idx_batch.device)
        for b in range(B):
            i = idx_batch[b].item()
            cand = pool[i]
            cand_valid = cand[cand >= 0]
            if cand_valid.numel() >= K:
                choice = cand_valid[torch.randperm(cand_valid.numel(), device=idx_batch.device)[:K]]
                pos_idx[b] = choice
            elif cand_valid.numel() > 0:
                need = K - cand_valid.numel()
                pad = torch.full((need,), i, dtype=torch.long, device=idx_batch.device)
                pos_idx[b] = torch.cat([cand_valid, pad], dim=0)
            else:
                pos_idx[b] = torch.full((K,), i, dtype=torch.long, device=idx_batch.device)
        return pos_idx

    def forward(self, x_norm_lazy: LazyCSR, indices: torch.Tensor, flag):
        X = x_norm_lazy[indices.detach().cpu().numpy()]
        h_anchor = self.encoder(X, return_proj=False)
        pos_idx = self._sample_neighbors(indices)
        positives1 = x_norm_lazy[pos_idx.detach().cpu().numpy()]
        positives = torch.cat([X.unsqueeze(1), positives1], dim=1)
        B, K, G = positives.shape
        positives_flat = positives.view(B * K, G)
        h_pos = self.encoder(positives_flat, return_proj=False)
        h_pos = h_pos.view(B, K, -1)
        tgt = h_anchor.unsqueeze(0)  # (1,B,D)
        memory = h_pos.permute(1, 0, 2)  # (10, B, D)
        attn_out =  self.decode(tgt, memory).squeeze(0)   # (B, D)
        eta_norm = self.decoder(h_anchor+attn_out) # (B, D)
        if flag:
            if self.batch_label is not None:
                z_cycle = self.encoder(eta_norm, return_proj=False)
                return eta_norm, (h_anchor, z_cycle)
            else:
                return eta_norm, (h_anchor, None)
        else:
            return eta_norm
class MMDLoss(nn.Module):
    """最大均值差异(MMD)损失"""
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss
    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
class ImputationLoss(nn.Module):
    def __init__(self, device, gene):
        super(ImputationLoss, self).__init__()
    def forward(self, eta_counts, y_true_counts, y_true_log, domain_feature, domain_label):
        log_p = torch.log_softmax(eta_counts, dim=1)
        y_true_log1p = torch.log1p(y_true_log)
        target_prob = torch.softmax(y_true_log1p, dim=1)
        loss1 = F.kl_div(log_p, target_prob, reduction='batchmean')
        loss2 = F.mse_loss(eta_counts, y_true_log1p)
        nz_mask = (y_true_counts > 0)
        loss_nz = F.l1_loss(eta_counts[nz_mask], y_true_log1p[nz_mask])
        h_anchor, z_cycle = domain_feature
        if z_cycle is not None:
            loss3 = F.mse_loss(z_cycle, h_anchor)
            return 0.4*loss1 + loss2 + loss3 + 0.4*loss_nz
        else:
            return 0.4*loss1 + loss2 + 0.4*loss_nz


# preprocessing
def preprocess(data: csr_matrix):
    assert isinstance(data, csr_matrix)
    print("preprocessing（CSR, UMI + log1p）")
    cell_sums = data.sum(axis=0).A1
    zero_mask = (cell_sums == 0)
    if np.any(zero_mask):
        median_nonzero = np.median(cell_sums[~zero_mask]) if (~zero_mask).any() else 1.0
        cell_sums[zero_mask] = median_nonzero
    scale_factors = np.median(cell_sums) / cell_sums
    data_norm = data.multiply(scale_factors).tocsr()
    data_log = data_norm.tocsr(copy=True)
    data_log.data = np.log1p(data_log.data)
    return data_log, scale_factors

class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best = None
        self.stop = False
        self.delta = delta
        self.best_state = None
    def step(self, val_loss, model):
        score = -val_loss
        if self.best is None or score > self.best + self.delta:
            self.best = score
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.verbose:
                print(f"[Impute] EarlyStopping {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True

def train_imputer(model: ImputerModel, train_loader, val_loader, batch_labels, gene,
                  x_norm_lazy: LazyCSR, x_raw_lazy: LazyCSR, scale_factors, device,
                  num_epochs=200, lr=1e-2, wd=1e-4):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, momentum=0.9, lr=lr, weight_decay=wd)
    sch = StepLR(opt, step_size=10, gamma=0.7)
    stopper = EarlyStopping(patience=5, verbose=True)
    impute_loss = ImputationLoss(device,gene)
    scale_factors = torch.tensor(scale_factors, dtype=torch.float32).to(device)
    for ep in range(num_epochs):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for (idx_batch,) in train_loader:
            idx_batch = idx_batch.to(device, non_blocking=True)
            eta_counts, z = model(x_norm_lazy, idx_batch, True)                    # (B,G)
            y_true = x_raw_lazy[idx_batch.detach().cpu().numpy()]        # (B,G)
            batch_label = batch_labels[idx_batch.detach().cpu().numpy()].to(device) if batch_labels is not None else None
            loss = impute_loss(eta_counts, y_true, y_true*scale_factors[idx_batch][:, None], z, batch_label)
            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf loss detected (train)")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tr_loss += loss.item() * idx_batch.size(0)
            tr_n += idx_batch.size(0)
        tr_avg = tr_loss / max(tr_n, 1)

        model.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for (idx_batch,) in val_loader:
                idx_batch = idx_batch.to(device, non_blocking=True)
                eta_counts, z = model(x_norm_lazy, idx_batch, True)
                y_true = x_raw_lazy[idx_batch.detach().cpu().numpy()]
                batch_label = batch_labels[idx_batch.detach().cpu().numpy()].to(device) if batch_labels is not None else None
                loss = impute_loss(eta_counts, y_true, y_true*scale_factors[idx_batch][:, None], z, batch_label)
                va_loss += loss.item() * idx_batch.size(0)
                va_n += idx_batch.size(0)
        va_avg = va_loss / max(va_n, 1)
        print(f"[Impute] Epoch {ep:03d} | Train {tr_avg:.4f} | Val {va_avg:.4f}")

        sch.step()
        stopper.step(va_avg, model)
        if stopper.stop:
            print("[Impute] Early stop")
            break
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    return model
@torch.no_grad()
def stream_impute_to_h5(model: ImputerModel, dataset, x_norm_lazy: LazyCSR, target_csc: csc_matrix, cell_scal,
                        save_file_path: str, batch_size: int = 256, device='cpu'):
    C, G = x_norm_lazy.shape
    G_, C_ = target_csc.shape
    assert (C == C_) and (G == G_), "Inconsistent shape"
    with h5py.File(save_file_path, 'w') as f:
        dset = f.create_dataset('data', shape=(C, G), dtype='float32',
                                chunks=(1, min(G, 4096)), compression="gzip")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=torch.cuda.is_available())
    model.eval()
    with h5py.File(save_file_path, 'a') as f:
        dset = f['data']
        for (idx_batch,) in loader:
            idx_cpu = idx_batch.cpu().numpy()
            idx_batch = idx_batch.to(device, non_blocking=True)
            out = model(x_norm_lazy, idx_batch, False).detach().cpu().numpy().astype(np.float32)                     # (B,G) log(mu_counts)
            np.expm1(out, out=out)
            out /= cell_scal[idx_cpu][:, None]
            out = np.clip(out, 0, None)  # mu_counts

            for b, col in enumerate(idx_cpu):
                col_vec = out[b]
                col_sparse = target_csc.getcol(col)
                if col_sparse.nnz > 0:
                    col_vec[col_sparse.indices] = col_sparse.data.astype(np.float32)
                dset[col, :] = col_vec
    print(f"finish：{save_file_path}")
#读取文件
def read_h5(file_path, data_key, gene_key, cell_key, label_key, time_key, batch_key):
    with h5py.File(file_path, 'r') as f:
        data_dense = f[data_key][:].T
        expression_matrix = csr_matrix(data_dense)
        processed_labels = f[label_key][:] if label_key and label_key in f else None
        gene_names = f[gene_key][:] if gene_key else None
        cell_barcodes = f[cell_key][:] if cell_key else None
        time = f[time_key][:] if time_key and time_key in f else None
        batch = f[batch_key][:] if batch_key and batch_key in f else None
    return expression_matrix, gene_names, cell_barcodes, processed_labels, time, batch

def main(use_gpu=True, gpu_id=0):
    contrastive_batch = 1024
    impute_batch = 256
    sim_threshold = 0.5
    pool_topk = 15
    train_neighbors_K = 10
    d_model = 512
    num = 4
    device = get_device(use_gpu, gpu_id)
    target_data, gene, cell, label_key, time, batch_labels = read_h5(
        '/media/swust123/DATA1/qiu_data/project/real_data/filter/'+ 'GSM2230759_human3' +'.h5',
        'data', 'gene_name', 'cell', 'label', 'time','batch_label'
    )
    target_scaled_csr, scale_factors = preprocess(target_data)
    x_norm = target_scaled_csr.T.tocsr()
    x_raw  = target_data.T.tocsr()
    if batch_labels is not None:
        batch_labels = np.array([x.decode('utf-8') for x in batch_labels])
        batch_labels = batch_labels.astype(np.int64)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    x_norm_lazy = LazyCSR(x_norm, dtype=torch.float32, device=device,
                          pin_memory=torch.cuda.is_available())
    x_raw_lazy  = LazyCSR(x_raw,  dtype=torch.float32, device=device,
                          pin_memory=torch.cuda.is_available())
    n_cells, n_genes = x_norm_lazy.shape

    all_idx = torch.arange(n_cells)
    dataset_all = TensorDataset(all_idx)
    train_size = int(0.9 * n_cells)
    val_size   = n_cells - train_size
    train_dataset, val_dataset = random_split(dataset_all, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=impute_batch, shuffle=True,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_dataset,   batch_size=impute_batch, shuffle=False,
                              pin_memory=torch.cuda.is_available())
    encoder = EncoderWithProjector(
        input_size=n_genes, hidden_size=d_model, proj_dim=128,
        num_heads=4, attn_dropout=0.1, hidden_dropout=0.1
    ).to(device)
    encoder = train_simclr(
        encoder, x_norm_lazy, device, batch_labels,
        epochs=500, batch_size=contrastive_batch, drop_rate=0.4,
        lr=3e-4, weight_decay=1e-4, temperature=0.1, patience=5
    )
    emb_h = extract_embeddings(encoder, x_norm_lazy, device, bs=1024)
    try:
        labels_np = label_key if label_key is not None else None
    except:
        labels_np = None

    # KNN pool（cosine≥0.5，max 15）
    idx_pool, sim_pool = build_neighbor_pool(emb_h, threshold=sim_threshold, topk=pool_topk)
    print(f"邻居池完成：阈值 {sim_threshold}，top {pool_topk}")

    # Freeze encoder + decoder training
    imputer = ImputerModel(
        frozen_encoder=encoder, n_genes=n_genes, n_cells=n_cells, batch_label = batch_labels,
        neighbor_idx_pool=idx_pool, cell_factors=scale_factors.astype(np.float32), num_domains = num,
        K_train=train_neighbors_K, d_model=d_model, device=device
    ).to(device)

    model_impute = train_imputer(
        imputer, train_loader, val_loader, batch_labels, n_genes,
        x_norm_lazy=x_norm_lazy, x_raw_lazy=x_raw_lazy, scale_factors=scale_factors.astype(np.float32),
        device=device, num_epochs=500
    )

    # save file
    save_file_path = '/media/swust123/DATA1/qiu_data/project/test/num_1_mid_1.h5'
    target_csc = target_data.tocsc()
    stream_impute_to_h5(
        model=model_impute,
        dataset=dataset_all,
        x_norm_lazy=x_norm_lazy,
        target_csc=target_csc,
        cell_scal=scale_factors.astype(np.float32),
        save_file_path=save_file_path,
        batch_size=impute_batch,
        device=device
    )
    with h5py.File(save_file_path, 'a') as f:
        if gene is not None: f.create_dataset('gene_name', data=gene)
        if cell is not None: f.create_dataset('cell', data=cell)
        if label_key is not None: f.create_dataset('label', data=label_key)
        if time is not None: f.create_dataset('time', data=time)
    print(f"file {save_file_path} ")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(24)
    main(use_gpu=True, gpu_id=0)
