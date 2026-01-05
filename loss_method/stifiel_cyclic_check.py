# -*- coding: utf-8 -*-
"""
CWS Quantum Codes on Asymmetric Channels
- Switchable error sets: 'orig' / 'E1' / 'E2' / 'E3'
- Switchable KL levels:  'detect' (检测级) / 'correct' (纠错级)
- Unified selector from (r, n, K, d) table spec
- General K-dimensional code subspace (CODE_K = TABLE_K)
- NEW: Optional cyclic-symmetry (k=0) constraint via subspace embedding

依赖: numpy, torch, scipy.sparse, matplotlib, numqi
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from itertools import combinations

# ============ Threads ============
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['OMP_NUM_THREADS'] = '12'
if torch.get_num_threads() != 12:
    torch.set_num_threads(12)

import numqi

# ============ Config ============
# —— 新风格（按表格）——
USE_TABLE_SPEC = True
TABLE_N = 5          # n（比特数） -> L
TABLE_K = 2          # K（码空间维）
TABLE_D = '2'     # '2'（对称）或 'asym'（非对称）
TABLE_R = None          # r；表中“–”用 None

# —— 旧风格（手选误差集，可不用）——
ERROR_SET_MODE = 'orig'   # 'orig'|'E1'|'E2'|'E3'

# —— 统一尺寸与K ——
L = TABLE_N if USE_TABLE_SPEC else 6
CODE_K = TABLE_K         # 码空间维度

# —— KL级别/绘图/优化 ——
KL_LEVEL = 'detect'      # 'detect' 或 'correct'
DROP_IDENTITY_IN_DETECT = True
PRINT_SIZES = True

lambda2_list = np.linspace(0.0, 1.5, 100)
if 0.6 not in lambda2_list:
    lambda2_list = np.insert(lambda2_list, np.searchsorted(lambda2_list, 0.6), 0.6)
if 1.0 not in lambda2_list:
    lambda2_list = np.insert(lambda2_list, np.searchsorted(lambda2_list, 1.0), 1.0)

optim_kwargs = dict(theta0='uniform', num_repeat=50, tol=1e-15,
                    print_freq=0, early_stop_threshold=1e-14)
constraint_threshold = 1e-12

# ============ Symmetry options (NEW) ============
# 是否把码空间限制在循环对称(k=0)子空间；False=不限制（原始行为）
USE_CYCLIC_SYM = True
# 预留接口：当前仅支持 k=0
CYCLIC_SECTOR = 'k0'

# ============ Pauli & Embedding ============
sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

_s = np.zeros((4, 2, 2), dtype=complex)  # 0:I, 1:X, 2:Y, 3:Z
_s[0, :, :] = np.eye(2, dtype=complex)
_s[1, :, :] = sx
_s[2, :, :] = sy
_s[3, :, :] = sz

def identity_n(L):
    return csr_matrix(sp.eye(2**L, dtype=complex, format='csr'))

def sigma(i, j, L):
    """ i:0->I,1->X,2->Y,3->Z; j=1..L; j==0 => I^{⊗L} """
    if j == 0:
        return identity_n(L)
    I2 = np.eye(2, dtype=complex)
    mat = _s[i] if j == 1 else I2
    for k in range(2, L+1):
        mat = np.kron(mat, _s[i] if k == j else I2)
    return csr_matrix(mat)

def sigmax(j, L): return sigma(1, j, L)
def sigmay(j, L): return sigma(2, j, L)
def sigmaz(j, L): return sigma(3, j, L)

# ============ Base Error Sets (old) ============
def error_set_orig(L):
    E = [identity_n(L)]
    for i in range(1, L+1):
        E += [sigmax(i, L), sigmay(i, L), sigmaz(i, L)]
    return E

def error_set_E1(L):
    E = [identity_n(L)]
    for i in range(1, L+1):
        E += [sigmax(i, L), sigmay(i, L), sigmaz(i, L)]
    for i, j in combinations(range(1, L+1), 2):
        E.append(sigmax(i, L) @ sigmax(j, L))
        E.append(sigmay(i, L) @ sigmay(j, L))
        E.append(sigmax(i, L) @ sigmay(j, L))
        E.append(sigmay(i, L) @ sigmax(j, L))
    return E

def error_set_E2(L):
    E1 = error_set_E1(L)
    return [A @ B for A in E1 for B in E1]

def error_set_E3(L, r):
    E = [identity_n(L)]
    for i in range(1, L+1):
        E += [sigmax(i, L), sigmay(i, L)]
    for w in range(1, min(r, L)+1):
        for idxs in combinations(range(1, L+1), w):
            Zprod = identity_n(L)
            for j in idxs:
                Zprod = Zprod @ sigmaz(j, L)
            E.append(Zprod)
    return E

# ============ Table-spec Error Sets (new) ============
def error_set_asym_mixed(L):
    """ {I} ∪ {X_i,Y_i,Z_i} ∪ {XX, ZZ, XZ, ZX (i<j)} """
    E = [identity_n(L)]
    for i in range(1, L+1):
        E += [sigmax(i, L), sigmay(i, L), sigmaz(i, L)]
    for i, j in combinations(range(1, L+1), 2):
        E.append(sigmax(i, L) @ sigmax(j, L))  # XX
        E.append(sigmaz(i, L) @ sigmaz(j, L))  # ZZ
        E.append(sigmax(i, L) @ sigmaz(j, L))  # XZ
        E.append(sigmaz(i, L) @ sigmax(j, L))  # ZX
    return E

def build_error_set_from_table_spec(L, K, d, r):
    """
    d='2'    -> {I}∪{X_i,Y_i,Z_i}
    d='asym' & r is None -> {I}∪{X_i,Y_i,Z_i}∪{XX,ZZ,XZ,ZX}
    d='asym' & r>=1     -> {I}∪{X_i,Y_i}∪Z_{≤r}
    """
    d = str(d).lower()
    if d in ['2', 'two', 'sym', 'symmetric']:
        return error_set_orig(L)
    if d in ['asym', 'asymmetric']:
        if (r is None) or (str(r).strip() == '' or str(r).lower() == 'none'):
            return error_set_asym_mixed(L)
        r = int(r)
        if r < 1: raise ValueError("For d='asym', r must be >=1 or None.")
        return error_set_E3(L, r)
    raise ValueError("TABLE_D must be '2' or 'asym'.")

def build_error_set(mode, L, r=None):
    mode = mode.lower()
    if mode == 'orig': return error_set_orig(L)
    if mode == 'e1':   return error_set_E1(L)
    if mode == 'e2':   return error_set_E2(L)
    if mode == 'e3':
        if r is None: raise ValueError("E3 requires integer r.")
        return error_set_E3(L, r)
    raise ValueError(f"Unknown ERROR_SET_MODE: {mode}")

# ============ KL Operator Lists ============
def distance_3_error_set(error_set):
    ret = []
    n = len(error_set)
    for i in range(n):
        for j in range(i+1, n):
            ret.append(error_set[i] @ error_set[j])
    return ret

def is_identity(A, L):
    return (A - identity_n(L)).nnz == 0

def build_op_list_for_KL(Error_set, L, KL_LEVEL, drop_I_in_detect=True):
    level = KL_LEVEL.lower()
    if level == 'detect':
        return [A for A in Error_set if not (drop_I_in_detect and is_identity(A, L))]
    if level == 'correct':
        return distance_3_error_set(Error_set)
    raise ValueError("KL_LEVEL must be 'detect' or 'correct'.")

# ============ Cyclic-invariant (k=0) embedding (NEW) ============
def _int_to_bits(i: int, L: int) -> tuple:
    return tuple((i >> (L-1-p)) & 1 for p in range(L))

def _bits_to_int(bits: tuple) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v

def _rotate_tuple(t: tuple, r: int = 1) -> tuple:
    n = len(t); r %= n
    return t[-r:] + t[:-r] if r else t

def _orbit_indices(i0: int, L: int) -> list:
    """返回整数索引组成的循环轨道（去重、有序）"""
    x = _int_to_bits(i0, L)
    seen = {}
    cur = x; r = 0
    while cur not in seen:
        seen[cur] = r
        r += 1
        cur = _rotate_tuple(cur, 1)
    # 按首次出现次序还原
    orb = [w for w,_ in sorted(seen.items(), key=lambda kv: kv[1])]
    return [_bits_to_int(w) for w in orb]

def build_cyclic_k0_embedding(L: int) -> np.ndarray:
    """
    返回 S ∈ C^{2^L × N_c}，其列为各循环轨道的等幅叠加 |Φ_orbit>
    列与列正交归一，张成 T 的本征值=1 的循环不变(k=0)子空间。
    """
    N = 2**L
    used_reps = set()
    columns = []
    for i in range(N):
        orb = _orbit_indices(i, L)
        rep = min(orb)
        if rep in used_reps:
            continue
        used_reps.add(rep)
        s = len(orb)
        col = np.zeros(N, dtype=np.complex128)
        amp = 1.0/np.sqrt(s)
        for idx in orb:
            col[idx] = amp
        columns.append(col)
    S = np.stack(columns, axis=1)  # [N, N_c]
    # 数值稳健：再做一次 QR 正交化（理论上已经互正交）
    Q, _ = np.linalg.qr(S)
    return Q.astype(np.complex128)

def describe_cyclic_embedding(L: int) -> str:
    S = build_cyclic_k0_embedding(L)
    return f"cyclic-k0 subspace (n={L}, dim={S.shape[1]})"

# ============ Model (General K) with optional embedding (NEW) ============
class DummyModel(torch.nn.Module):
    """
    U_eff ∈ St(N_eff, CODE_K). 若给定嵌入 S (N×N_eff)，则 U_full = S @ U_eff。
    对每个 O_k：
      若使用嵌入：O_eff = S^† O_k S；A_k = U_eff^† O_eff U_eff
      否则：       A_k = U_full^† O_k U_full
    Loss:
      - offdiag: 所有非对角元的 |·|^2 总和
      - diag   : 对角实部按“每个算符的对角均值”一致化（零均差平方和）
      - lambda : 以每个算符对角均值 λ_k 进行目标塑形
    """
    def __init__(self, op_list, penalty=1.0, embedding: np.ndarray = None):
        super().__init__()
        # 稀疏 -> 稠密张量
        op_dense = torch.stack([torch.tensor(op.toarray(), dtype=torch.complex128) for op in op_list])  # [M,N,N]
        self.embedding = None
        if embedding is not None:
            # S: [N, N_c]
            S_np = embedding
            S = torch.tensor(S_np, dtype=torch.complex128)
            Sdag = S.T.conj()
            # 降维：O_eff = S^† O S -> [M, N_c, N_c]
            op_eff = torch.einsum('ab,mbc,cd->mad', Sdag, op_dense, S)
            self.op_list = op_eff
            self.embedding = S                      # 保存以便导出全空间码基
            N_eff = S.shape[1]
        else:
            self.op_list = op_dense
            N_eff = op_dense.shape[-1]

        self.code_k = int(CODE_K)
        if self.code_k > N_eff:
            raise ValueError(f"CODE_K={self.code_k} 大于有效子空间维度 N_eff={N_eff}。请降低 K 或关闭对称性限制。")

        # 在有效维度上做 Stiefel 优化
        self.manifold = numqi.manifold.Stiefel(N_eff, self.code_k, dtype=torch.complex128)
        self.lambda_target = None
        self.penalty = penalty

    def set_lambda_target(self, x):
        if x is None:
            self.lambda_target = None
        elif isinstance(x, str):
            assert x in ['min', 'max']
            self.lambda_target = x
        else:
            t = torch.tensor(x, dtype=torch.float64).reshape(-1)
            self.lambda_target = t[0] if t.numel() == 1 else t

    def forward(self, return_info=False):
        U_eff = self.manifold()                               # [N_eff, K]
        A = U_eff.T.conj() @ self.op_list @ U_eff             # [M, K, K]

        # Off-diagonal penalty
        diagA = torch.diagonal(A, dim1=1, dim2=2)             # [M, K]
        A_off = A - torch.diag_embed(diagA)
        loss_offdiag = (A_off.abs()**2).sum().real

        # Diagonal-equality penalty
        diag_real = diagA.real
        diag_mean = diag_real.mean(dim=1, keepdim=True)
        loss_diag = ((diag_real - diag_mean)**2).sum()

        # Lambda shaping
        lambdas = diag_mean.squeeze(1)                        # [M]
        if self.lambda_target is None:
            loss_lambda = torch.tensor(0.0, dtype=torch.float64)
        elif isinstance(self.lambda_target, (str,)):
            loss_lambda = torch.dot(lambdas, lambdas) if self.lambda_target == 'min' else -torch.dot(lambdas, lambdas)
        elif self.lambda_target.numel() == 1:
            loss_lambda = (torch.dot(lambdas, lambdas) - self.lambda_target**2)**2
        else:
            loss_lambda = torch.dot(lambdas - self.lambda_target, lambdas - self.lambda_target)

        total = self.penalty*(loss_offdiag + loss_diag) + loss_lambda

        if return_info:
            # 导出“全空间”的码基列向量（便于后验检查/可视化）
            if self.embedding is not None:
                U_full = self.embedding @ U_eff
            else:
                U_full = U_eff
            total = total, dict(
                loss=(total, loss_offdiag, loss_diag, loss_lambda),
                lambda_ab_ij=A,       # A 在有效空间内
                code=U_full,          # 全空间码基
                code_eff=U_eff        # 有效空间码基
            )
        return total

# ============ Run ============
def main():
    print('[setup] PyTorch threads =', torch.get_num_threads())
    print(f'[config] L={L}, CODE_K={CODE_K}, KL={KL_LEVEL}, TABLE(d={TABLE_D}, r={TABLE_R}, K={TABLE_K})')

    if USE_TABLE_SPEC:
        Error_set = build_error_set_from_table_spec(L=L, K=TABLE_K, d=TABLE_D, r=TABLE_R)
        mode_desc = f"table_spec(d={TABLE_D}, r={TABLE_R}, n={L}, K={TABLE_K})"
    else:
        Error_set = build_error_set(ERROR_SET_MODE, L, r=(TABLE_R if ERROR_SET_MODE.lower()=='e3' else None))
        mode_desc = f"{ERROR_SET_MODE}"

    if PRINT_SIZES:
        print(f"[Info] |Error_set| = {len(Error_set)}")

    op_list_sparse = build_op_list_for_KL(Error_set, L, KL_LEVEL, drop_I_in_detect=DROP_IDENTITY_IN_DETECT)
    if PRINT_SIZES:
        print(f"[Info] |op_list| = {len(op_list_sparse)}")

    embedding = None
    if USE_CYCLIC_SYM:
        assert CYCLIC_SECTOR.lower() == 'k0', "当前仅支持 cyclic k=0 扇区。"
        S = build_cyclic_k0_embedding(L)    # np.ndarray [2^L, N_c]
        embedding = S
        if PRINT_SIZES:
            print(f"[Info] cyclic embedding: N={2**L} -> N_eff={S.shape[1]}  ({describe_cyclic_embedding(L)})")
        if CODE_K > S.shape[1]:
            raise ValueError(f"CODE_K={CODE_K} 大于循环不变子空间维度 {S.shape[1]}。请降低 K 或关闭 USE_CYCLIC_SYM。")

    model = DummyModel(op_list_sparse, embedding=embedding)
    model.penalty = 10

    loss_histories = []
    codes = []
    for lambda2 in lambda2_list:
        model.set_lambda_target(np.sqrt(lambda2))
        callback = numqi.optimize.MinimizeCallback(print_freq=500)
        _ = numqi.optimize.minimize(model, callback=callback, **optim_kwargs)

        with torch.no_grad():
            hist = []
            for h in callback.history_state:
                numqi.optimize.set_model_flat_parameter(model, h['optim_x'])
                loss, info = model(return_info=True)
                hist.append([x.item() for x in info['loss']])  # [total, offdiag, diag, lambda]
            print('[scan] lambda2 =', float(lambda2), '| iterates =', len(hist))
            loss_histories.append(hist)
        codes.append(model(return_info=True)[1]['code'])

    # pick best feasible per target
    def feasible(x): return (x[1] + x[2]) < constraint_threshold
    def best(history):
        feas = [x for x in history if feasible(x)]
        return [np.nan]*4 if len(feas)==0 else min(feas, key=lambda x: x[3])

    fval_optim = np.array([best(h) for h in loss_histories])

    fig, ax = plt.subplots()
    ax.plot(lambda2_list, fval_optim[:, 3])
    ax.set_xlabel(r'$\lambda^{*2}$')
    ax.set_ylabel('lambda loss')
    ax.set_yscale('log'); ax.grid(True)

    max_constraint = np.nan_to_num(fval_optim[1:,1] + fval_optim[1:,2], nan=0).max()
    sym_tag = "_cyclic" if USE_CYCLIC_SYM else ""
    out_png = f"n{L}_K{CODE_K}_{TABLE_D}_r{TABLE_R}_KL-{KL_LEVEL}{sym_tag}.png"
    ax.set_title(f"Cyclic Codes vs $\\lambda$, loss_constraint <= {max_constraint:.3g} | mode={mode_desc} | KL={KL_LEVEL} | K={CODE_K}{' | cyclic k=0' if USE_CYCLIC_SYM else ''}")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f'[plot] saved -> {out_png}')

# ============ Optional QC ============
def qc_sample(model: DummyModel, idx_list=None, max_print=3):
    """随机抽样若干 O_k，打印 U^H O_k U 的数值（适配任意 K；兼容嵌入）。"""
    with torch.no_grad():
        loss, info = model(return_info=True)
        A = info['lambda_ab_ij']    # [M,K,K]（有效空间内）
        U = info['code']            # [N,K]   （全空间）
        M, K = A.shape[0], A.shape[-1]
        if idx_list is None:
            idx_list = np.random.choice(M, size=min(max_print, M), replace=False)
        for k in idx_list:
            Mk = A[k].cpu().numpy()
            print(f"[QC] k={k}, shape={Mk.shape}, K={K}")
            sl = slice(0, min(K, 4))
            print(np.round(Mk[sl, sl], 6))
            off = Mk - np.diag(np.diag(Mk))
            print("-- offdiag Fro norm^2:", float((abs(off)**2).sum()))
            d = np.real(np.diag(Mk))
            print("-- diag real:", np.round(d, 6), " (std:", float(d.std()), ")")

# ============ Entry ============
if __name__ == '__main__':
    main()
