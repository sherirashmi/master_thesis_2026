# Settings for deterministic reproducability
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["PYTHONHASHSEED"] = "0"

# =============================================================================
# ERP neural operators: forward (design -> ERP) and invertible (both directions)
# for a simply supported plate carrying tuned mass resonators.
#
# Order     1 device and seed    2 parameters and plate modes
#           3 solver (displacement, velocity, ERP)
#           4 dataset creation   5 normalisation, splits, statistics
#           6 forward models     7 invertible models (iDON, iFNO family)
#           8 training, validation, prediction, plots    9 main
# Folders   dataset/<dataset>/{dataset file(s), stats, plots}   (one folder per dataset, everything inside it)
#           forward_models/{models,plots/<dataset>/<MODEL>}
#           invertible_models/{models,plots/<dataset>/<MODEL>}
# Data      m, f_t, x, y (uniform) and damping ratio zeta (log-uniform) of every resonator by Latin hypercube;
#           k = m (2 pi f_t)^2, c = 2 zeta sqrt(k m)
# Defaults  every model uses the set + f_t-sorted resonator encoder and the
#           plate-mode (physics-aware) features; invertible models use the
#           bounded design, bounded gate, binned readout, cycle/alignment
#           terms and stage-2 pretraining on stage-1 estimates. Their design is
#           [m, f_t, x, y] per resonator with the damping fixed to c = 1 N s/m.
# Run       python erp_neural_operators.py
# =============================================================================

#%% Imports and Configuration
import sys
import math
import time
import copy
import functools
import inspect
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from scipy.stats import qmc
from scipy.signal import find_peaks

#%% Latex properties for Matplotlib
# Switch to True to render all text with LaTeX (needs a LaTeX installation; model names with "_" must then be escaped)
use_latex = False
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12, "axes.labelsize": 13,
                     "axes.titlesize": 13, "legend.fontsize": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight"})
if use_latex and shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": "\\usepackage{amsmath}\\usepackage{amssymb}"})
ERP_LABEL, FREQ_LABEL = "ERP (dB re 1 pW)", "Frequency (Hz)"

#%% 1. Device and seed
# Set random seeds fixed to ensure reproducability
SEED = 727
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For deterministic CUDA GPU settings
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    print(f"Running on {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (no GPU found)"))

#%% Directories to save datasets, models and figures (created next to this file when it is run)
ROOT = Path(os.getcwd())
DATASET_DIR = ROOT / "dataset"
FORWARD_DIR = ROOT / "forward_models"
INVERTIBLE_DIR = ROOT / "invertible_models"


def make_folders():
    """
    Creates the dataset, model and plot folders if they do not exist yet
    """
    for path in (DATASET_DIR, FORWARD_DIR / "models", FORWARD_DIR / "plots",
                 INVERTIBLE_DIR / "models", INVERTIBLE_DIR / "plots"):
        path.mkdir(parents=True, exist_ok=True)


#%% 2. Physical and Simulation Parameters
# --- Plate (aluminium, simply supported), point force, fluid ---
Lx, Ly, h = 1.4, 0.5, 0.005                 # Plate length, width and thickness (m)
E = 7.1e10 * (1 + 1j * 0.001)               # Complex Young's modulus (Pa), loss factor 0.001
rho, nu = 2800.0, 0.3                       # Density (kg/m^3) and Poisson's ratio
D_plate = E * h**3 / (12 * (1 - nu**2))     # Flexural rigidity
M_plate = rho * h * Lx * Ly                 # Total plate mass (kg)
rho_f, c_f, P_ref = 1.21, 343.0, 1e-12      # Air density (kg/m^3), speed of sound (m/s), reference power (W)
F0, xf, yf = 1.0, 0.865, 0.309              # Force amplitude (N) and force location (m)

# --- Frequency range (301 points) ---
f_min, f_max, f_step = 10.0, 160.0, 0.5     # Hz
freqs = np.arange(f_min, f_max + 0.5 * f_step, f_step)
n_freq = freqs.size

# --- ERP integration grid ---
erp_nx, erp_ny = 280, 100
X_grid, Y_grid = np.meshgrid(np.linspace(0, Lx, erp_nx), np.linspace(0, Ly, erp_ny))
dA = (Lx / (erp_nx - 1)) * (Ly / (erp_ny - 1))

# --- Resonators: mass, tuning frequency, position (uniform) and damping ratio (log-uniform) by Latin hypercube,
#     k = m (2 pi f_t)^2, c = 2 zeta sqrt(k m) ---
m_min, m_max = 0.1, 1.0                     # Resonator mass (kg)
edge_margin = 0.05                          # Minimum distance of a resonator to the plate edges (m)
zeta_min, zeta_max = 5e-4, 8e-2             # Damping ratio (-)

# --- Training settings ---
epochs_forward = 200                        # Epochs of the forward models
epochs_idon = 150                           # Epochs of the invertible DeepONet (iDON)
epochs_stages = (100, 25, 50)               # Epochs of stage 1/2/3 of the iFNO family
batch_size = 128                            # Batch size
weight_decay_forward = 1e-4                 # AdamW weight decay of the forward models
lr_min_forward = 1e-6                       # Final learning rate of the cosine schedule (forward models)
grad_clip_forward = 5.0                     # Gradient clipping norm (forward models)
resume_every = 5                            # Epochs between resume checkpoints of the forward models

# --- Invertible training settings ---
STAGE_LR = (5e-4, 1e-3, 3e-4)               # Learning rates of stage 1/2/3 (iFNO family)
KL_TARGET, KL_WARMUP = 0.05, 8              # Final beta of the beta-VAE and number of warm-up epochs
GRAD_CLIP, EARLY_STOP_PATIENCE = 5.0, 10    # Gradient clipping norm and early-stopping patience (epochs)
CYCLE_WEIGHT = ALIGN_WEIGHT = 0.1           # Weights of the cycle and alignment terms
IDON_SLOPE, IDON_PEAK = 0.5, 0.05           # iDON loss weights: spectrum slope and peak terms
IDON_PEAK_START, IDON_INVERSE_WARMUP = 0.8, 10   # Fraction of epochs after which the peak term starts, inverse-loss warm-up epochs

# --- Plate modes (set when a dataset is generated or loaded) ---
Nx, Ny = 6, 3


def set_modes(nx_modes, ny_modes):
    """
    Plate mode basis (nx_modes x ny_modes) used by the solver.
    """
    global Nx, Ny, N, omega_n, m_idx, n_idx
    Nx, Ny = int(nx_modes), int(ny_modes)
    table = np.array([[np.real(np.sqrt(D_plate / (rho * h)) * ((i * np.pi / Lx) ** 2 + (j * np.pi / Ly) ** 2)), i, j]
                      for i in range(1, Nx + 1) for j in range(1, Ny + 1)])
    table = table[np.argsort(table[:, 0])]
    omega_n, m_idx, n_idx = table[:, 0], table[:, 1], table[:, 2]
    N = len(table)


set_modes(Nx, Ny)

#%% 3. Solver: displacement, velocity and ERP


def mode_shapes(x, y):
    """
    Mass-normalised plate mode shapes, (N modes, N points).
    """
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    return 2.0 / np.sqrt(M_plate) * np.sin(np.pi * np.outer(m_idx, x) / Lx) * np.sin(np.pi * np.outer(n_idx, y) / Ly)


def coupled_matrices(resonators):
    """
    Mass, damping, stiffness matrices and force vector of plate modes + resonators.
    """
    n_res = len(resonators)
    M, C, K = (np.zeros((N + n_res, N + n_res), complex) for _ in range(3))
    M[:N, :N] = np.eye(N)
    K[:N, :N] = np.diag(omega_n**2)
    force = np.zeros(N + n_res, complex)
    force[:N] = F0 * mode_shapes(xf, yf)[:, 0]
    if n_res:
        phi = mode_shapes([r["x"] for r in resonators], [r["y"] for r in resonators])
    for i, r in enumerate(resonators):
        p, d = phi[:, i], N + i
        M[d, d] = r["m"]
        for A, value in ((C, r["c"]), (K, r["k"])):
            A[:N, :N] += value * np.outer(p, p)
            A[:N, d] = A[d, :N] = -value * p
            A[d, d] = value
    return M, C, K, force


def solve_modal_response(resonators, frequencies=freqs, block=64):
    """
    Plate modal coordinates q, (n frequencies, N), frequency blocks solved in one batch.
    """
    frequencies = np.atleast_1d(np.asarray(frequencies, float))
    M, C, K, force = coupled_matrices(resonators)
    q = np.empty((frequencies.size, N), complex)
    for start in range(0, frequencies.size, block):
        w = 2 * np.pi * frequencies[start:start + block]
        A = K[None] + 1j * w[:, None, None] * C[None] - (w**2)[:, None, None] * M[None]
        rhs = np.broadcast_to(force, (w.size, force.size))[..., None]
        q[start:start + block] = np.linalg.solve(A, rhs)[..., 0][:, :N]
    return q


def compute_displacement(resonators, x=None, y=None, frequencies=freqs):
    """
    Complex displacement: full grid (ny, nx, n_freq) or at given points (n_freq, n_points).
    """
    q = solve_modal_response(resonators, frequencies)
    if x is None:
        return (q @ mode_shapes(X_grid.ravel(), Y_grid.ravel())).reshape(-1, *X_grid.shape).transpose(1, 2, 0)
    return q @ mode_shapes(x, y)


def compute_velocity(displacement, frequencies=freqs):
    """
    Harmonic velocity from displacement (last axis = frequency).
    """
    return 1j * 2 * np.pi * np.asarray(frequencies) * displacement


def compute_erp(velocity):
    """
    ERP in dB from a velocity field (ny, nx, n_freq).
    """
    power = 0.5 * rho_f * c_f * np.sum(np.abs(velocity) ** 2, axis=(0, 1)) * dA
    return 10 * np.log10(np.maximum(power, np.finfo(float).tiny) / P_ref)


_gram_cache = {}


def compute_erp_spectrum(resonators, frequencies=freqs):
    """
    ERP spectrum in dB (same as compute_erp of the full field, via the modal Gram matrix).
    """
    if (Nx, Ny) not in _gram_cache:
        phi = mode_shapes(X_grid.ravel(), Y_grid.ravel())
        _gram_cache[(Nx, Ny)] = phi @ phi.T * dA
    frequencies = np.atleast_1d(np.asarray(frequencies, float))
    q = solve_modal_response(resonators, frequencies)
    energy = np.einsum("fi,ij,fj->f", q.conj(), _gram_cache[(Nx, Ny)], q).real * (2 * np.pi * frequencies) ** 2
    return 10 * np.log10(np.maximum(0.5 * rho_f * c_f * energy, np.finfo(float).tiny) / P_ref)


def to_resonators(configuration):
    """
    (R, 6) [m, k, f_t, x, y, zeta] rows -> solver dictionaries, c = 2 zeta sqrt(k m).
    """
    return [dict(m=float(m), k=float(k), f_t=float(ft), x=float(x), y=float(y), zeta=float(z), c=float(2 * z * np.sqrt(k * m)))
            for m, k, ft, x, y, z in np.asarray(configuration, float)]


_gram_torch = {}


def _erp_batch(configurations):
    """ERP spectra (B, F) in dB of B configurations (B, R, 6), computed on `device`.
    The resonator degrees of freedom are eliminated and the remaining plate system, diagonal plate modes plus a rank-R
    correction sum_r a_r phi_r phi_r^T with a_r = -s_r w^2 m_r / (s_r - w^2 m_r), s_r = k_r + i w c_r, is solved with the
    Woodbury identity: only R x R systems per frequency. Same result as the full coupled solve."""
    key = (Nx, Ny, str(device))
    if key not in _gram_torch:
        phi_grid = torch.from_numpy(mode_shapes(X_grid.ravel(), Y_grid.ravel())).to(device)
        _gram_torch[key] = phi_grid @ phi_grid.T * dA
    gram = _gram_torch[key]
    cfg = configurations.to(device=device, dtype=torch.float64)
    m, k, x, y, zeta = cfg[..., 0], cfg[..., 1], cfg[..., 3], cfg[..., 4], cfg[..., 5]
    c = 2 * zeta * torch.sqrt(k * m)
    mi, ni = torch.as_tensor(m_idx, dtype=torch.float64, device=device), torch.as_tensor(n_idx, dtype=torch.float64, device=device)
    phi = 2 / math.sqrt(M_plate) * torch.sin(math.pi * mi[None, :, None] * x[:, None, :] / Lx) * torch.sin(math.pi * ni[None, :, None] * y[:, None, :] / Ly)  # (B, N, R)
    f0 = F0 * 2 / math.sqrt(M_plate) * torch.sin(math.pi * mi * xf / Lx) * torch.sin(math.pi * ni * yf / Ly)  # (N,)
    w = torch.as_tensor(2 * np.pi * freqs, device=device)  # (F,)
    on = torch.as_tensor(omega_n, device=device)
    d_inv = 1.0 / (on[None, :] ** 2 - w[:, None] ** 2)  # (F, N) inverse of the diagonal plate matrix
    u = f0[None, :] * d_inv  # (F, N)
    s = k[:, None, :] + 1j * w[None, :, None] * c[:, None, :]  # (B, F, R)
    mass = (w**2)[None, :, None] * m[:, None, :]
    a = -s * mass / (s - mass)
    gmat = torch.einsum("bnr,fn,bns->bfrs", phi, d_inv, phi).to(torch.complex128)  # Phi^T D^-1 Phi
    eye = torch.eye(cfg.shape[1], dtype=torch.complex128, device=device)
    t = torch.linalg.solve(eye + a[..., None] * gmat, (a * torch.einsum("bnr,fn->bfr", phi, u))[..., None])[..., 0]  # (B, F, R)
    correction = torch.einsum("bnr,bfr->bfn", phi.to(torch.complex128), t) * d_inv[None].to(torch.complex128)
    q = u[None].to(torch.complex128) - correction  # (B, F, N) plate modal coordinates
    energy = ((q.real @ gram) * q.real).sum(-1) + ((q.imag @ gram) * q.imag).sum(-1)
    power = 0.5 * rho_f * c_f * w[None] ** 2 * energy
    return 10 * torch.log10(torch.clamp(power, min=torch.finfo(torch.float64).tiny) / P_ref)


def solve_configurations(configurations, batch_size=None, report_every=0):
    """
    ERP spectra (n, F) of physical configurations (n, R, 6), solved in batches on the GPU if available, else on the CPU.
    """
    batch_size = batch_size or (256 if device.type == "cuda" else 32)
    configurations = torch.as_tensor(np.asarray(configurations, np.float64))
    out = np.empty((len(configurations), n_freq), np.float32)
    start = time.time()
    next_report = report_every
    for i in range(0, len(configurations), batch_size):
        out[i:i + batch_size] = _erp_batch(configurations[i:i + batch_size]).cpu().numpy()
        if report_every and (i + batch_size >= next_report or i + batch_size >= len(configurations)):
            done = min(i + batch_size, len(configurations))
            print(f"  {done}/{len(configurations)} configurations solved ({time.time() - start:.0f} s)")
            next_report = done + report_every
    return out


#%% 4. Dataset creation


def sample_configurations(n, num_res, seed=SEED):
    """
    (n, num_res, 6) [m, k, f_t, x, y, zeta]: m, f_t, x, y uniform and zeta log-uniform, all from one Latin hypercube.
    """
    u = qmc.LatinHypercube(d=5 * num_res, seed=seed).random(n).reshape(n, num_res, 5)
    m = m_min + u[..., 0] * (m_max - m_min)
    f_t = f_min + u[..., 1] * (f_max - f_min)
    x = edge_margin + u[..., 2] * (Lx - 2 * edge_margin)
    y = edge_margin + u[..., 3] * (Ly - 2 * edge_margin)
    zeta = np.exp(np.log(zeta_min) + u[..., 4] * (np.log(zeta_max) - np.log(zeta_min)))
    return np.stack([m, m * (2 * np.pi * f_t) ** 2, f_t, x, y, zeta], axis=-1).astype(np.float32)


def dataset_tag(n, num_res, nx_modes, ny_modes):
    label = f"{n // 1000}k" if n % 1000 == 0 else str(n)
    return f"{label}_{num_res}res_{nx_modes}x{ny_modes}modes"


MAX_FILE_BYTES = 15 * 1000 * 1000   # largest dataset file written; bigger datasets are split into parts
DATA_KEYS = ("configuration_features", "responses")


def dataset_folder(tag):
    return DATASET_DIR / tag


def dataset_files(tag):
    """
    (single file, meta file, sorted part files) of dataset/<tag>/ -- existing or not.
    """
    folder = dataset_folder(tag)
    return folder / f"{tag}.pth", folder / f"{tag}_meta.pth", sorted(folder.glob(f"{tag}_part_*.pth"))


def save_dataset(tag, meta, arrays):
    """
    Save dataset/<tag>/<tag>.pth when it fits in MAX_FILE_BYTES, else <tag>_meta.pth + <tag>_part_NNNN.pth, each <= MAX_FILE_BYTES.
    """
    folder = dataset_folder(tag)
    folder.mkdir(parents=True, exist_ok=True)
    single, meta_file, parts = dataset_files(tag)
    for old in [single, meta_file, *parts]:
        old.unlink(missing_ok=True)
    arrays = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}
    n, row_bytes = len(arrays[DATA_KEYS[0]]), sum(v[0].numel() * v.element_size() for v in arrays.values())
    margin = 64 * 1024   # pickle / zip overhead
    if n * row_bytes + margin <= MAX_FILE_BYTES:
        torch.save(dict(meta, **arrays), single)
        written = [single]
    else:
        rows = max(1, (MAX_FILE_BYTES - margin) // row_bytes)
        num_parts = math.ceil(n / rows)
        torch.save(dict(meta, _split=dict(keys=list(arrays), num_parts=num_parts)), meta_file)
        written = [meta_file]
        for i in range(num_parts):
            part = folder / f"{tag}_part_{i:04d}.pth"
            torch.save({k: v[i * rows:(i + 1) * rows].clone() for k, v in arrays.items()}, part)
            written.append(part)
    too_big = [f.name for f in written if f.stat().st_size > MAX_FILE_BYTES]
    if too_big:
        raise RuntimeError(f"dataset files larger than {MAX_FILE_BYTES / 1e6:g} MB: {too_big}")
    return written


def load_dataset_payload(tag):
    """
    Dict with the dataset: dataset/<tag>/<tag>.pth if it exists, else its meta file and all parts joined.
    """
    single, meta_file, parts = dataset_files(tag)
    if single.exists():
        return torch.load(single, weights_only=False)
    if not meta_file.exists():
        raise FileNotFoundError(f"No dataset files for '{tag}' in {dataset_folder(tag)}")
    payload = torch.load(meta_file, weights_only=False)
    split = payload.pop("_split")
    if len(parts) != split["num_parts"]:
        raise FileNotFoundError(f"'{tag}' needs {split['num_parts']} part files but {len(parts)} found in {dataset_folder(tag)}")
    chunks = [torch.load(p, weights_only=False) for p in parts]
    for key in split["keys"]:
        payload[key] = np.concatenate([np.asarray(c[key]) for c in chunks])
    return payload


def generate_dataset(n, num_res, nx_modes, ny_modes):
    """
    Solve n random configurations and save them in dataset/<tag>/ (one file, or parts of at most 15 MB).
    """
    make_folders()
    set_modes(nx_modes, ny_modes)
    tag = dataset_tag(n, num_res, nx_modes, ny_modes)
    print(f"Generating '{tag}': {n} configurations, {num_res} resonators, {N} plate modes, {n_freq} frequencies")
    configurations = sample_configurations(n, num_res)
    responses = solve_configurations(configurations, report_every=max(1, n // 20))
    meta = dict(tag=tag, n=int(n), num_res=int(num_res), nx_modes=int(nx_modes), ny_modes=int(ny_modes), seed=SEED,
                frequency_values=freqs.astype(np.float32))
    files = save_dataset(tag, meta, {"configuration_features": configurations, "responses": responses})
    print(f"Saved dataset/{tag}/ ({len(files)} file{'s' if len(files) > 1 else ''}, each at most {MAX_FILE_BYTES / 1e6:g} MB)")
    return tag


def list_datasets():
    """
    Settings (tag, n, num_res, nx_modes, ny_modes) of every dataset folder in dataset/.
    """
    make_folders()
    found = []
    for folder in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        single, meta_file, _ = dataset_files(folder.name)
        source = single if single.exists() else meta_file if meta_file.exists() else None
        if source is not None:
            payload = torch.load(source, weights_only=False)
            found.append({k: payload[k] for k in ("tag", "n", "num_res", "nx_modes", "ny_modes")})
    return found


#%% 5. Normalisation, splits and statistics
FIELDS = ("m", "k", "f_t", "x", "y", "zeta")
CONF_DIM = len(FIELDS)
# Invertible models: bounded design coordinates [m, f_t, x, y] per resonator (name, index in FIELDS, lower bound, upper bound).
# The damping is not designed: it is fixed to c = 1 N s/m, so the models see the 5 fields [m, k, f_t, x, y].
BOUNDED = (("m", 0, m_min, m_max), ("f_t", 2, f_min, f_max), ("x", 3, 0.0, Lx), ("y", 4, 0.0, Ly))
DESIGN_FIELDS = tuple(b[0] for b in BOUNDED)
N_DESIGN = len(BOUNDED)
INV_CONF_DIM = 5


def log_zeta(configuration):
    """
    Copy of (..., 6) configurations with the damping ratio replaced by its logarithm (it spans two decades).
    """
    out = np.asarray(configuration, np.float32).copy()
    out[..., 5] = np.log(out[..., 5])
    return out


def normalize_config(configuration, norm):
    """
    z-score each field [m, k, f_t, x, y, log zeta] (statistics shared by all resonators).
    """
    out = log_zeta(configuration)
    for i, name in enumerate(FIELDS):
        out[..., i] = (out[..., i] - norm[f"{name}_mean"]) / norm[f"{name}_std"]
    return out


def normalize_erp(erp, norm):
    return ((np.asarray(erp, np.float32) - norm["erp_mean"]) / norm["erp_std"]).astype(np.float32)


def denormalize_erp(erp, norm):
    erp = erp.detach().cpu().numpy() if torch.is_tensor(erp) else np.asarray(erp)
    return erp.astype(np.float32) * norm["erp_std"] + norm["erp_mean"]


def sort_by_ft(configuration):
    """
    Resonators in ascending tuning frequency (the spectrum does not depend on the order).
    """
    configuration = np.asarray(configuration)
    order = np.argsort(configuration[..., 2], axis=-1)
    return np.take_along_axis(configuration, order[..., None], axis=-2)


def _unit(values, lo, hi):
    """
    Value -> (0, 1) inside its bounds.
    """
    return np.clip((np.asarray(values, float) - lo) / (hi - lo), 1e-6, 1 - 1e-6)


def fit_bounded_stats(configuration):
    """
    Logit-space mean/std of the bounded design coordinates (training designs).
    """
    stats = {}
    for name, idx, lo, hi in BOUNDED:
        u = _unit(np.asarray(configuration)[..., idx], lo, hi)
        w = np.log(u) - np.log1p(-u)
        stats[f"b12_{name}_mean"], stats[f"b12_{name}_std"] = float(w.mean()), float(max(w.std(), 1e-6))
    return stats


def encode_bounded(configuration, norm):
    """
    Physical (..., R, 6) -> normalised bounded coordinates (..., R, 4) [m, f_t, x, y].
    """
    out = []
    for name, idx, lo, hi in BOUNDED:
        u = _unit(np.asarray(configuration)[..., idx], lo, hi)
        out.append((np.log(u) - np.log1p(-u) - norm[f"b12_{name}_mean"]) / norm[f"b12_{name}_std"])
    return np.stack(out, axis=-1).astype(np.float32)


def decode_bounded_torch(flat, num_res, mean, std):
    """Differentiable (..., R*4) -> physical (..., R, 6) [m, k, f_t, x, y, zeta]; k = m (2 pi f_t)^2 and the damping is
    fixed to c = 1 N s/m, i.e. zeta = 1 / (2 sqrt(k m)). mean/std: (4,) tensors."""
    lo = torch.tensor([b[2] for b in BOUNDED], device=flat.device, dtype=flat.dtype)
    hi = torch.tensor([b[3] for b in BOUNDED], device=flat.device, dtype=flat.dtype)
    s = torch.sigmoid(flat.reshape(*flat.shape[:-1], num_res, N_DESIGN) * std + mean)
    m, f_t, x, y = (lo + (hi - lo) * s).unbind(-1)
    k = m * (2 * math.pi * f_t) ** 2
    return torch.stack((m, k, f_t, x, y, 1.0 / (2.0 * torch.sqrt(k * m))), dim=-1)


def decode_bounded(flat, num_res, norm):
    mean = torch.tensor([norm[f"b12_{b[0]}_mean"] for b in BOUNDED], dtype=torch.float64)
    std = torch.tensor([norm[f"b12_{b[0]}_std"] for b in BOUNDED], dtype=torch.float64)
    with torch.no_grad():
        return decode_bounded_torch(torch.as_tensor(np.asarray(flat), dtype=torch.float64), num_res, mean, std).numpy()


class SpectrumSet(Dataset):
    """
    (normalised configuration, normalised frequencies, normalised ERP) per configuration.
    """

    def __init__(self, config, erp, freq):
        self.config, self.erp, self.freq = map(torch.from_numpy, (config, erp[..., None], freq[:, None]))

    def __len__(self):
        return len(self.config)

    def __getitem__(self, i):
        return self.config[i], self.freq, self.erp[i]


class DesignSet(Dataset):
    """
    (normalised ERP, normalised bounded design sorted by f_t) per configuration.
    """

    def __init__(self, erp, design):
        self.erp, self.design = torch.from_numpy(erp), torch.from_numpy(design)

    def __len__(self):
        return len(self.erp)

    def __getitem__(self, i):
        return self.erp[i], self.design[i]


class ERPData:
    """
    Loads a dataset, splits it 80/10/10 by configuration and fits the normalisation on the training part.
    """

    def __init__(self, tag):
        payload = load_dataset_payload(tag)
        self.tag, self.num_res = tag, int(payload["num_res"])
        set_modes(payload["nx_modes"], payload["ny_modes"])
        self.config = np.asarray(payload["configuration_features"], np.float32)
        if self.config.shape[-1] == 5:  # dataset without damping column: c = 1 N s/m, zeta = c / (2 sqrt(k m))
            self.config = np.concatenate([self.config, 1.0 / (2 * np.sqrt(self.config[..., 1:2] * self.config[..., 0:1]))], axis=-1).astype(np.float32)
        self.erp = np.asarray(payload["responses"], np.float32)
        self.freqs = np.asarray(payload["frequency_values"], np.float32)
        order = np.random.default_rng(SEED).permutation(len(self.config))
        n_train, n_val = int(0.8 * len(order)), int(0.1 * len(order))
        self.split = {"train": order[:n_train], "val": order[n_train:n_train + n_val], "test": order[n_train + n_val:]}
        train = self.config[self.split["train"]]
        train_log = log_zeta(train)
        self.norm = {"num_res": self.num_res, "freq_mean": float(self.freqs.mean()), "freq_std": float(self.freqs.std()),
                     "erp_mean": float(self.erp[self.split["train"]].mean()), "erp_std": float(self.erp[self.split["train"]].std())}
        for i, name in enumerate(FIELDS):
            self.norm[f"{name}_mean"], self.norm[f"{name}_std"] = float(train_log[..., i].mean()), float(max(train_log[..., i].std(), 1e-8))
        self.norm.update(fit_bounded_stats(sort_by_ft(train)))
        self.freq_norm = ((self.freqs - self.norm["freq_mean"]) / self.norm["freq_std"]).astype(np.float32)

    def with_norm(self, norm):
        """
        Shallow copy that normalises with `norm`: the statistics a checkpoint was trained with (possibly another dataset's).
        """
        other = copy.copy(self)
        other.norm = norm
        other.freq_norm = ((self.freqs - norm["freq_mean"]) / norm["freq_std"]).astype(np.float32)
        return other

    def loaders(self, kind, batch_size):
        """
        kind 'forward': configuration -> ERP;  kind 'inverse': ERP <-> bounded design (used by invertible models).
        """
        out = {}
        for name, ids in self.split.items():
            erp = normalize_erp(self.erp[ids], self.norm)
            if kind == "forward":
                dataset = SpectrumSet(normalize_config(self.config[ids], self.norm), erp, self.freq_norm)
            else:
                dataset = DesignSet(erp, encode_bounded(sort_by_ft(self.config[ids]), self.norm))
            generator = torch.Generator().manual_seed(SEED) if name == "train" else None
            out[name] = DataLoader(dataset, batch_size=batch_size, shuffle=(name == "train"), generator=generator,
                                   pin_memory=device.type == "cuda")
        return out


def dataset_statistics(data):
    """
    Statistics and plots of a dataset -> dataset/<tag>/ (csv, txt and png files next to the dataset file(s)).
    """
    make_folders()
    tag, erp, config = data.tag, data.erp, data.config
    stats_dir = plots_dir = dataset_folder(tag)
    stats_dir.mkdir(parents=True, exist_ok=True)
    q1, med, q3 = np.percentile(erp, [25, 50, 75], axis=0)
    iqr = q3 - q1
    low = np.where(erp >= q1 - 1.5 * iqr, erp, np.inf).min(0)
    high = np.where(erp <= q3 + 1.5 * iqr, erp, -np.inf).max(0)
    columns = ["frequency_hz", "min_db", "lower_whisker_db", "q1_db", "median_db", "q3_db", "upper_whisker_db", "max_db", "mean_db", "std_db"]
    table = np.column_stack([data.freqs, erp.min(0), low, q1, med, q3, high, erp.max(0), erp.mean(0), erp.std(0)])
    np.savetxt(stats_dir / "erp_frequency_statistics.csv", table, delimiter=",", header=",".join(columns), comments="")

    peaks = [len(find_peaks(s, prominence=3.0)[0]) for s in erp[np.random.default_rng(SEED).choice(len(erp), min(2000, len(erp)), replace=False)]]
    natural = [f for f in omega_n / (2 * np.pi) if f_min <= f <= f_max]
    lines = [f"Dataset {tag}", f"configurations: {len(config)}  resonators: {data.num_res}  plate modes: {Nx}x{Ny} ({N})",
             f"split (train/val/test): {len(data.split['train'])}/{len(data.split['val'])}/{len(data.split['test'])}",
             f"frequencies: {n_freq} ({f_min:g}-{f_max:g} Hz, step {f_step:g} Hz)",
             f"ERP over all spectra: min {erp.min():.2f}  mean {erp.mean():.2f}  max {erp.max():.2f}  std {erp.std():.2f} dB",
             f"peaks per spectrum (prominence 3 dB): mean {np.mean(peaks):.2f}  max {np.max(peaks)}",
             f"bare-plate natural frequencies in band: {', '.join(f'{f:.1f}' for f in natural)} Hz", "", "field        mean        std        min        max"]
    lines += [f"{name:<6s} {config[..., i].mean():11.4f} {config[..., i].std():10.4f} {config[..., i].min():10.4f} {config[..., i].max():10.4f}"
              for i, name in enumerate(FIELDS)]
    (stats_dir / "summary.txt").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for j, f in enumerate(natural):
        ax.axvline(f, color="gray", ls="--", lw=0.8, label="Bare-plate natural frequencies" if j == 0 else None)
    ax.fill_between(data.freqs, erp.min(0), erp.max(0), color="C0", alpha=0.10, lw=0, label="Minimum to maximum")
    ax.fill_between(data.freqs, low, high, color="C0", alpha=0.22, lw=0, label="Whiskers (1.5 IQR)")
    ax.fill_between(data.freqs, q1, q3, color="C0", alpha=0.45, lw=0, label="Interquartile range")
    ax.plot(data.freqs, med, "k", lw=1.5, label="Median")
    ax.set(xlabel=FREQ_LABEL, ylabel=ERP_LABEL, xlim=(f_min, f_max), title=f"ERP distribution, {tag}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "erp_frequency_band.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
    for ax, (i, label) in zip(axes, ((0, "Mass $m$ (kg)"), (2, "Tuning frequency $f_t$ (Hz)"), (3, "Position $x$ (m)"), (4, "Position $y$ (m)"),
                                     (5, r"Damping ratio $\zeta$"))):
        values = config[..., i].ravel()
        ax.hist(values, bins=np.geomspace(values.min(), values.max(), 41) if i == 5 else 40, color="C0")
        ax.set(xlabel=label, ylabel="Resonators", xscale="log" if i == 5 else "linear")
    fig.tight_layout()
    fig.savefig(plots_dir / "parameter_distributions.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist2d(config[..., 3].ravel(), config[..., 4].ravel(), bins=(28, 10), range=[[0, Lx], [0, Ly]], cmap="Blues")
    axes[0].set(xlabel="$x$ (m)", ylabel="$y$ (m)", title="Resonator positions", aspect="equal")
    axes[1].hist(peaks, bins=np.arange(-0.5, max(peaks) + 1.5), color="C0")
    axes[1].set(xlabel="Peaks per spectrum", ylabel="Spectra", title="Peaks (prominence 3 dB)")
    fig.tight_layout()
    fig.savefig(plots_dir / "positions_and_peaks.png")
    plt.close(fig)
    print(f"Statistics and plots saved to dataset/{tag}")

#%% 6. Forward models (configuration, frequency -> ERP)
ACT = {"relu": torch.nn.ReLU, "gelu": torch.nn.GELU, "silu": torch.nn.SiLU, "tanh": torch.nn.Tanh}


class MLP(torch.nn.Module):
    def __init__(self, widths, activation=torch.nn.Tanh):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(torch.nn.Linear(a, b))
            if i < len(widths) - 2:
                layers.append(activation())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualMLPBlock(torch.nn.Module):
    def __init__(self, width, activation=torch.nn.Tanh):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(width, width), activation(), torch.nn.Linear(width, width))
        self.norm, self.act = torch.nn.LayerNorm(width), activation()

    def forward(self, x):
        return self.act(self.norm(x + self.net(x)))


class FrequencyRefinement1d(torch.nn.Module):
    """
    Residual local mixing along frequency; input (B, width, F).
    """

    def __init__(self, width):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Conv1d(width, width, 3, padding=1), torch.nn.SiLU(), torch.nn.Conv1d(width, width, 3, padding=1))
        self.norm = torch.nn.GroupNorm(1, width)

    def forward(self, x):
        return torch.nn.functional.silu(self.norm(x + self.net(x)))


# --- plate-mode (physics-aware) features: sin(i pi x/Lx), sin(j pi y/Ly) and detuning f - f_t ---
def modal_features(module, configuration, harmonics):
    """
    [m, k, f_t, x, y, log zeta] + x/y mode shapes + their products; x, y are mapped to x/Lx, y/Ly by the module's coord_affine.
    """
    a = getattr(module, "coord_affine")
    u, v = configuration[..., 3] * a[0, 0] + a[0, 1], configuration[..., 4] * a[1, 0] + a[1, 1]
    xm = [torch.sin(math.pi * i * u)[..., None] for i in range(1, harmonics + 1)]
    ym = [torch.sin(math.pi * j * v)[..., None] for j in range(1, harmonics + 1)]
    return torch.cat([configuration, *xm, *ym, *[a_ * b_ for a_ in xm for b_ in ym]], dim=-1)


def resonance_detuning(module, query, f_t):
    a = module.detuning_affine
    return query - (f_t * a[0] + a[1])


def enable_physical_features(model):
    for module in model.modules():
        if getattr(module, "uses_modal_features", False):
            module.register_buffer("coord_affine", torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
        if getattr(module, "uses_detuning", False):
            module.register_buffer("detuning_affine", torch.tensor([1.0, 0.0]))
    return model


def set_physical_feature_normalization(model, norm):
    coord = torch.tensor([[norm["x_std"] / Lx, norm["x_mean"] / Lx], [norm["y_std"] / Ly, norm["y_mean"] / Ly]])
    detuning = torch.tensor([norm["f_t_std"] / norm["freq_std"], (norm["f_t_mean"] - norm["freq_mean"]) / norm["freq_std"]])
    for module in model.modules():
        if hasattr(module, "coord_affine"):
            module.coord_affine.copy_(coord)
        if hasattr(module, "detuning_affine"):
            module.detuning_affine.copy_(detuning)


def feature_dim(harmonics, conf_dim=CONF_DIM):
    return conf_dim + 2 * harmonics + harmonics**2


# --- resonator encoders ---
class ResonatorSetEncoder(torch.nn.Module):
    """
    Shared per-resonator MLP + mean/max pooling (permutation invariant).
    """
    uses_modal_features = True

    def __init__(self, hidden_dim=128, element_dim=128, output_dim=128, harmonics=10, conf_dim=CONF_DIM):
        super().__init__()
        self.harmonics = harmonics
        self.element_net = MLP([feature_dim(harmonics, conf_dim), hidden_dim, hidden_dim, element_dim])
        self.fusion_net = MLP([2 * element_dim, hidden_dim, output_dim])

    def forward(self, configuration):
        h = self.element_net(modal_features(self, configuration, self.harmonics))
        return self.fusion_net(torch.cat((h.mean(1), h.max(1).values), dim=-1))


class SortedResonatorEncoder(torch.nn.Module):
    """
    Resonators sorted by f_t and concatenated (lossless, order-canonical).
    """
    uses_modal_features = True

    def __init__(self, num_res, hidden_dim=128, output_dim=128, harmonics=10, conf_dim=CONF_DIM):
        super().__init__()
        self.harmonics = harmonics
        self.net = MLP([num_res * feature_dim(harmonics, conf_dim), hidden_dim, hidden_dim, output_dim])

    def forward(self, configuration):
        order = torch.argsort(configuration[..., 2], dim=-1)
        ordered = torch.gather(configuration, 1, order[..., None].expand(-1, -1, configuration.shape[-1]))
        return self.net(modal_features(self, ordered, self.harmonics).flatten(1))


class SetAndSortedEncoder(torch.nn.Module):
    """
    Pooled set branch + f_t-sorted branch, fused (the default configuration encoder).
    """

    def __init__(self, num_res, hidden_dim=128, element_dim=128, output_dim=128, harmonics=10):
        super().__init__()
        self.set_encoder = ResonatorSetEncoder(hidden_dim, element_dim, output_dim, harmonics)
        self.sorted_encoder = SortedResonatorEncoder(num_res, hidden_dim, output_dim, harmonics)
        self.fusion = torch.nn.Linear(2 * output_dim, output_dim)

    def forward(self, configuration):
        return self.fusion(torch.cat((self.set_encoder(configuration), self.sorted_encoder(configuration)), dim=-1))


class ResonanceQueryEncoder(torch.nn.Module):
    """
    Resonator / query-frequency interaction with detuning features, pooled over resonators.
    """
    uses_modal_features = True
    uses_detuning = True

    def __init__(self, hidden_dim=64, element_dim=64, output_dim=64, harmonics=10, conf_dim=CONF_DIM):
        super().__init__()
        self.harmonics = harmonics
        self.element_net = MLP([feature_dim(harmonics, conf_dim) + 4, hidden_dim, hidden_dim, element_dim])
        self.fusion_net = MLP([2 * element_dim, hidden_dim, output_dim])

    def forward(self, configuration, frequency):
        b, n, _ = configuration.shape
        f = frequency.shape[1]
        resonator = modal_features(self, configuration, self.harmonics)[:, None].expand(b, f, n, -1)
        query = frequency[:, :, None, :].expand(b, f, n, 1)
        delta = resonance_detuning(self, query, configuration[:, None, :, 2:3].expand(b, f, n, 1))
        h = self.element_net(torch.cat((resonator, query, delta, delta.abs(), delta.square()), dim=-1))
        return self.fusion_net(torch.cat((h.mean(2), h.max(2).values), dim=-1))


# --- architectures ---
class DON(torch.nn.Module):
    """DeepONet with an orthonormal basis (as in iDON): ERP(f) = sum_q b_q(configuration) psi_q(f) + psi_0(f).
    The trunk maps the frequency alone to Q + 1 functions; psi_1..psi_Q are orthonormalised by QR on every pass
    (Psi^T Psi = F I), psi_0 is a free bias function. The branch (resonator encoder + MLP) gives the coefficients b."""

    def __init__(self, num_res, hidden_dim=54, context_dim=86, q=64, trunk_hidden=155, num_fourier=32):
        super().__init__()
        self.encoder = SetAndSortedEncoder(num_res, hidden_dim, hidden_dim, context_dim)
        self.branch_head = MLP([context_dim, 2 * hidden_dim, q], torch.nn.Tanh)
        self.register_buffer("fourier", torch.arange(1, num_fourier + 1, dtype=torch.float32) * math.pi)
        self.trunk = MLP([1 + 2 * num_fourier, trunk_hidden, trunk_hidden, trunk_hidden, q + 1], torch.nn.SiLU)

    def basis(self, frequency):
        """
        (Psi (F, Q) with Psi^T Psi = F I, psi_0 (F,)) on the frequency grid (shared by the batch).
        """
        f = frequency.reshape(-1)
        f = 2 * (f - f.min()) / (f.max() - f.min()) - 1
        arg = f[:, None] * self.fourier[None]
        out = self.trunk(torch.cat((f[:, None], torch.sin(arg), torch.cos(arg)), dim=-1))
        qmat, _ = torch.linalg.qr(out[:, 1:])
        return qmat * math.sqrt(len(f)), out[:, 0]

    def forward(self, configuration, frequency):
        psi, psi0 = self.basis(frequency[0])
        coefficients = self.branch_head(self.encoder(configuration))
        return (coefficients @ psi.T + psi0).unsqueeze(-1)


class FiLMResidualBlock(torch.nn.Module):
    def __init__(self, width, condition_dim, activation=torch.nn.SiLU):
        super().__init__()
        self.block, self.film, self.act = ResidualMLPBlock(width, activation), torch.nn.Linear(condition_dim, 2 * width), activation()

    def forward(self, x, condition):
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        return self.act((1 + 0.2 * torch.tanh(gamma)) * self.block(x) + 0.1 * beta)


class DNO(torch.nn.Module):
    """
    Deep operator: residual blocks with FiLM conditioning on configuration and detuning.
    """

    def __init__(self, num_res, hidden_dim=54, context_dim=57, frequency_dim=28, query_dim=28, depth=4):
        super().__init__()
        self.encoder = SetAndSortedEncoder(num_res, hidden_dim, hidden_dim, context_dim)
        self.frequency_encoder = MLP([1, frequency_dim, frequency_dim], torch.nn.SiLU)
        self.query = ResonanceQueryEncoder(query_dim, query_dim, query_dim)
        self.lift = torch.nn.Linear(context_dim + frequency_dim + query_dim, hidden_dim)
        self.blocks = torch.nn.ModuleList([FiLMResidualBlock(hidden_dim, context_dim + query_dim) for _ in range(depth)])
        self.refine = FrequencyRefinement1d(hidden_dim)
        self.output = MLP([hidden_dim, hidden_dim // 2, 1], torch.nn.SiLU)

    def forward(self, configuration, frequency):
        context = self.encoder(configuration)[:, None].expand(-1, frequency.shape[1], -1)
        query = self.query(configuration, frequency)
        h = torch.nn.functional.silu(self.lift(torch.cat((context, self.frequency_encoder(frequency), query), dim=-1)))
        condition = torch.cat((context, query), dim=-1)
        for block in self.blocks:
            h = block(h, condition)
        return self.output(self.refine(h.transpose(1, 2)).transpose(1, 2))


class DCO(torch.nn.Module):
    """
    Deep Cat operator: concatenated branch, f_t-sorted branch, trunk and query features, residual blocks.
    """

    def __init__(self, num_res, hidden_dim=67, branch_dim=56, trunk_dim=56, query_dim=28, depth=4):
        super().__init__()
        self.branch = ResonatorSetEncoder(hidden_dim, hidden_dim, branch_dim)
        self.sorted_branch = SortedResonatorEncoder(num_res, hidden_dim, branch_dim)
        self.trunk = MLP([1, hidden_dim, trunk_dim], torch.nn.SiLU)
        self.query = ResonanceQueryEncoder(query_dim, query_dim, query_dim)
        self.lift = torch.nn.Linear(2 * branch_dim + trunk_dim + query_dim, hidden_dim)
        self.blocks = torch.nn.ModuleList([ResidualMLPBlock(hidden_dim, torch.nn.SiLU) for _ in range(depth)])
        self.refine = FrequencyRefinement1d(hidden_dim)
        self.output = MLP([hidden_dim, hidden_dim // 2, 1], torch.nn.SiLU)

    def forward(self, configuration, frequency):
        n_f = frequency.shape[1]
        parts = [self.branch(configuration)[:, None].expand(-1, n_f, -1), self.sorted_branch(configuration)[:, None].expand(-1, n_f, -1),
                 self.trunk(frequency), self.query(configuration, frequency)]
        h = torch.nn.functional.silu(self.lift(torch.cat(parts, dim=-1)))
        for block in self.blocks:
            h = block(h)
        return self.output(self.refine(h.transpose(1, 2)).transpose(1, 2))


class SineLayer(torch.nn.Module):
    """
    SIREN layer with configuration-dependent FiLM modulation.
    """

    def __init__(self, n_in, n_out, context_dim, first=False, omega_0=20.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear, self.film = torch.nn.Linear(n_in, n_out), torch.nn.Linear(context_dim, 2 * n_out)
        with torch.no_grad():
            bound = 1.0 / n_in if first else math.sqrt(6.0 / n_in) / omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
            self.film.weight.zero_()
            self.film.bias.zero_()

    def forward(self, x, context):
        gamma, beta = self.film(context).chunk(2, dim=-1)
        return torch.sin(self.omega_0 * ((1 + 0.25 * torch.tanh(gamma))[:, None] * self.linear(x) + 0.1 * beta[:, None]))


class SIREN(torch.nn.Module):
    """
    SIREN operator: sine layers over frequency and detuning features, modulated by the configuration.
    """

    def __init__(self, num_res, context_dim=57, query_dim=14, hidden_dim=75, config_hidden=57, query_hidden=28, depth=4, omega_0=20.0):
        super().__init__()
        self.encoder = SetAndSortedEncoder(num_res, config_hidden, config_hidden, context_dim)
        self.query = ResonanceQueryEncoder(query_hidden, query_hidden, query_dim)
        self.layers = torch.nn.ModuleList([SineLayer(1 + query_dim, hidden_dim, context_dim, True, omega_0)] +
                                    [SineLayer(hidden_dim, hidden_dim, context_dim, False, omega_0) for _ in range(depth - 1)])
        self.refine, self.output = FrequencyRefinement1d(hidden_dim), torch.nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.output.weight.uniform_(-bound, bound)
            self.output.bias.zero_()

    def forward(self, configuration, frequency):
        context = self.encoder(configuration)
        h = torch.cat((frequency, self.query(configuration, frequency)), dim=-1)
        for layer in self.layers:
            h = layer(h, context)
        return self.output(self.refine(h.transpose(1, 2)).transpose(1, 2))


class SpectralConv1d(torch.nn.Module):
    """
    Learned convolution on the lowest Fourier modes; optional zero padding of the frequency axis.
    """

    def __init__(self, in_channels, out_channels, modes, padding=0):
        super().__init__()
        self.out_channels, self.modes, self.padding = out_channels, modes, padding
        self.weight = torch.nn.Parameter(torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat) / max(1, in_channels * out_channels))

    def forward(self, x):
        n0 = x.shape[-1]
        if self.padding:
            x = torch.nn.functional.pad(x, (0, self.padding))
        x_ft = torch.fft.rfft(x, dim=-1)
        k = min(self.modes, x_ft.shape[-1])
        out = torch.zeros(x.shape[0], self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out[:, :, :k] = torch.einsum("bim,iom->bom", x_ft[:, :, :k], self.weight[:, :, :k])
        return torch.fft.irfft(out, n=x.shape[-1], dim=-1)[..., :n0]


class FNOBlock1d(torch.nn.Module):
    def __init__(self, width, modes, dropout=0.1):
        super().__init__()
        self.spectral, self.local = SpectralConv1d(width, width, modes), torch.nn.Conv1d(width, width, 3, padding=1)
        self.norm, self.dropout = torch.nn.GroupNorm(1, width), torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(torch.nn.functional.gelu(self.norm(x + self.spectral(x) + self.local(x))))


class FNO(torch.nn.Module):
    """
    Fourier neural operator along frequency, with edge padding and detuning features.
    """

    def __init__(self, num_res, width=23, modes=35, depth=4, config_hidden=70, query_dim=26, padding=8):
        super().__init__()
        self.padding = padding
        self.encoder = SetAndSortedEncoder(num_res, config_hidden, config_hidden, width)
        self.query = ResonanceQueryEncoder(query_dim, query_dim, query_dim)
        self.lift = torch.nn.Linear(width + query_dim + 1, width)
        self.blocks = torch.nn.ModuleList([FNOBlock1d(width, modes) for _ in range(depth)])
        self.project = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, 1))

    def forward(self, configuration, frequency):
        context = self.encoder(configuration)[:, None].expand(-1, frequency.shape[1], -1)
        x = self.lift(torch.cat((context, self.query(configuration, frequency), frequency), dim=-1)).transpose(1, 2)
        x = torch.nn.functional.pad(x, (self.padding, self.padding), mode="replicate")
        for block in self.blocks:
            x = block(x)
        return self.project(x[..., self.padding:-self.padding].transpose(1, 2))


class HaarWaveletBlock1d(torch.nn.Module):
    """
    Multi-level Haar analysis, learned mixing per level, synthesis.
    """

    def __init__(self, width, levels=3, dropout=0.1):
        super().__init__()
        self.levels = levels
        self.low_mix = torch.nn.ModuleList([torch.nn.Conv1d(width, width, 3, padding=1) for _ in range(levels)])
        self.high_mix = torch.nn.ModuleList([torch.nn.Conv1d(width, width, 3, padding=1) for _ in range(levels)])
        self.coarse_mix, self.local = torch.nn.Conv1d(width, width, 3, padding=1), torch.nn.Conv1d(width, width, 3, padding=1)
        self.norm, self.dropout = torch.nn.GroupNorm(1, width), torch.nn.Dropout(dropout)

    def forward(self, x):
        s = 1 / math.sqrt(2)
        current, details, lengths = x, [], []
        for level in range(self.levels):
            lengths.append(current.shape[-1])
            padded = torch.nn.functional.pad(current, (0, 1), mode="replicate") if current.shape[-1] % 2 else current
            even, odd = padded[..., 0::2], padded[..., 1::2]
            details.append(torch.nn.functional.gelu(self.high_mix[level]((even - odd) * s)))
            current = torch.nn.functional.gelu(self.low_mix[level]((even + odd) * s))
        current = torch.nn.functional.gelu(self.coarse_mix(current))
        for level in reversed(range(self.levels)):
            even, odd = (current + details[level]) * s, (current - details[level]) * s
            merged = torch.stack((even, odd), dim=-1).flatten(-2)
            current = merged[..., :lengths[level]]
        return self.dropout(torch.nn.functional.gelu(self.norm(x + current + self.local(x))))


class WNO(torch.nn.Module):
    """
    Wavelet neural operator along frequency.
    """

    def __init__(self, num_res, width=31, depth=4, levels=3, config_hidden=58, query_dim=22):
        super().__init__()
        self.encoder = SetAndSortedEncoder(num_res, config_hidden, config_hidden, width)
        self.query = ResonanceQueryEncoder(query_dim, query_dim, query_dim)
        self.lift = torch.nn.Linear(width + query_dim + 1, width)
        self.blocks = torch.nn.ModuleList([HaarWaveletBlock1d(width, levels) for _ in range(depth)])
        self.project = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, 1))

    def forward(self, configuration, frequency):
        context = self.encoder(configuration)[:, None].expand(-1, frequency.shape[1], -1)
        x = self.lift(torch.cat((context, self.query(configuration, frequency), frequency), dim=-1)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.project(x.transpose(1, 2))


def pole_features(f, sigma, omega, r_re, r_im):
    """
    Re/Im of r/(s - p) + conj(r)/(s - conj(p)), s = i f, p = -sigma + i omega; f (..., F), poles (..., 1, P) -> (..., F, 2P).
    """
    s = torch.complex(torch.zeros_like(f), f)[..., None]
    pole, res = torch.complex(-sigma, omega), torch.complex(r_re, r_im)
    term = res / (s - pole) + res.conj() / (s - pole.conj())
    return torch.cat((term.real, term.imag), dim=-1)


class LNO(torch.nn.Module):
    """
    Laplace neural operator: poles and residues predicted from the configuration.
    """

    def __init__(self, num_res, width=86, num_poles=13, config_hidden=62, pole_hidden=62, query_dim=24):
        super().__init__()
        self.num_poles = num_poles
        self.encoder = SetAndSortedEncoder(num_res, config_hidden, config_hidden, width)
        self.query = ResonanceQueryEncoder(query_dim, query_dim, query_dim)
        self.pole_predictor = MLP([width, pole_hidden, pole_hidden, 4 * num_poles], torch.nn.SiLU)
        self.lift, self.dropout = torch.nn.Linear(2 * num_poles + query_dim + 1, width), torch.nn.Dropout(0.1)
        self.refine = FrequencyRefinement1d(width)
        self.project = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, 1))

    def forward(self, configuration, frequency):
        raw = self.pole_predictor(self.encoder(configuration)).view(-1, 1, self.num_poles, 4)
        poles = pole_features(frequency[..., 0], torch.nn.functional.softplus(raw[..., 0]) + 1e-3, torch.nn.functional.softplus(raw[..., 1]), raw[..., 2], raw[..., 3])
        x = self.dropout(self.lift(torch.cat((poles, self.query(configuration, frequency), frequency), dim=-1)))
        return self.project(self.refine(x.transpose(1, 2)).transpose(1, 2))


class GraphMessageLayer(torch.nn.Module):
    """
    Complete-graph message passing between resonators with geometric and tuning edge features.
    """

    def __init__(self, width, dropout=0.0, conf_dim=CONF_DIM):
        super().__init__()
        self.message, self.update = MLP([2 * width + conf_dim + 2, width, width], torch.nn.SiLU), MLP([2 * width, width, width], torch.nn.SiLU)
        self.norm, self.dropout = torch.nn.LayerNorm(width), torch.nn.Dropout(dropout)

    def forward(self, h, features):
        b, n, w = h.shape
        rel = features[:, None] - features[:, :, None]
        edge = torch.cat((rel, torch.linalg.vector_norm(rel[..., 3:5], dim=-1, keepdim=True), rel[..., 2:3].abs()), dim=-1)
        message = self.message(torch.cat((h[:, :, None].expand(b, n, n, w), h[:, None].expand(b, n, n, w), edge), dim=-1))
        mask = (~torch.eye(n, dtype=torch.bool, device=h.device))[None, :, :, None]
        aggregate = (message * mask).sum(2) / max(n - 1, 1)
        return self.dropout(torch.nn.functional.silu(self.norm(h + self.update(torch.cat((h, aggregate), dim=-1)))))


class GNO(torch.nn.Module):
    """
    Graph neural operator: resonator graph, detuning-aware attention over resonators per frequency.
    """
    uses_modal_features = True
    uses_detuning = True

    def __init__(self, num_res, width=68, depth=3, frequency_dim=32, harmonics=4, dropout=0.1):
        super().__init__()
        self.harmonics = harmonics
        self.node_lift = MLP([feature_dim(harmonics), width, width], torch.nn.SiLU)
        self.layers = torch.nn.ModuleList([GraphMessageLayer(width, dropout) for _ in range(depth)])
        self.frequency_encoder = MLP([1, frequency_dim, frequency_dim], torch.nn.SiLU)
        self.query_kernel = MLP([width + CONF_DIM + frequency_dim + 4, width, width, width], torch.nn.SiLU)
        self.attention_score = MLP([width, width // 2, 1], torch.nn.SiLU)
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.refine = FrequencyRefinement1d(width)
        self.output = MLP([width, width // 2, 1], torch.nn.SiLU)

    def forward(self, configuration, frequency):
        h = self.node_lift(modal_features(self, configuration, self.harmonics))
        for layer in self.layers:
            h = layer(h, configuration)
        b, f, _ = frequency.shape
        n = configuration.shape[1]
        query = frequency[:, :, None].expand(b, f, n, 1)
        delta = resonance_detuning(self, query, configuration[:, None, :, 2:3].expand(b, f, n, 1))
        pair = torch.cat((h[:, None].expand(b, f, n, -1), configuration[:, None].expand(b, f, n, -1),
                          self.frequency_encoder(frequency)[:, :, None].expand(b, f, n, -1), query, delta, delta.abs(), delta.square()), dim=-1)
        kernel = self.query_kernel(pair)
        weights = self.attention_dropout(torch.softmax(self.attention_score(kernel), dim=2))
        integral = self.refine((weights * kernel).sum(2).transpose(1, 2)).transpose(1, 2)
        return self.output(integral)


class DetuningCrossAttention(torch.nn.Module):
    """
    Multi-head frequency-to-resonator attention with a learned detuning bias.
    """
    uses_detuning = True

    def __init__(self, width, heads, dropout=0.0, activation=torch.nn.SiLU, conf_dim=CONF_DIM):
        super().__init__()
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (torch.nn.Linear(width, width) for _ in range(4))
        self.bias_net = MLP([conf_dim + 3, width // 2, heads], activation)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query, tokens, configuration, frequency):
        b, f, _ = query.shape
        n = tokens.shape[1]
        q = self.q_proj(query).view(b, f, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(tokens).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        score = torch.einsum("bhfd,bhnd->bhfn", q, k) / math.sqrt(self.head_dim)
        query_f = frequency[:, :, None].expand(b, f, n, 1)
        delta = resonance_detuning(self, query_f, configuration[:, None, :, 2:3].expand(b, f, n, 1))
        bias = self.bias_net(torch.cat((configuration[:, None].expand(b, f, n, -1), query_f, delta, delta.abs()), dim=-1)).permute(0, 3, 1, 2)
        attention = self.dropout(torch.softmax(score + bias, dim=-1))
        return self.out_proj(torch.einsum("bhfn,bhnd->bhfd", attention, v).transpose(1, 2).reshape(b, f, self.width))


class FrequencyMixer(torch.nn.Module):
    """
    Local and dilated mixing along frequency; input (B, width, F).
    """

    def __init__(self, width, activation=torch.nn.GELU):
        super().__init__()
        self.local, self.dilated = torch.nn.Conv1d(width, width, 5, padding=2), torch.nn.Conv1d(width, width, 3, padding=2, dilation=2)
        self.norm, self.act = torch.nn.GroupNorm(1, width), activation()

    def forward(self, x):
        return self.act(self.norm(x + self.local(x) + self.dilated(x)))


class STO(torch.nn.Module):
    """
    Set Transformer operator: resonator self-attention, detuning-biased cross-attention from frequencies.
    """
    uses_modal_features = True

    def __init__(self, num_res, width=64, heads=4, depth=2, ff_dim=168, harmonics=4, dropout=0.1):
        super().__init__()
        self.harmonics = harmonics
        self.node_lift = MLP([feature_dim(harmonics), width, width], torch.nn.GELU)
        layer = torch.nn.TransformerEncoderLayer(width, heads, ff_dim, dropout, activation=torch.nn.GELU(), batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, depth)
        self.frequency_query = MLP([1, width, width], torch.nn.GELU)
        self.cross_attention = DetuningCrossAttention(width, heads, dropout, torch.nn.GELU)
        self.cross_norm, self.mixer = torch.nn.LayerNorm(width), FrequencyMixer(width)
        self.output = MLP([width, width, width // 2, 1], torch.nn.GELU)

    def forward(self, configuration, frequency):
        tokens = self.encoder(self.node_lift(modal_features(self, configuration, self.harmonics)))
        query = self.frequency_query(frequency)
        h = self.cross_norm(query + self.cross_attention(query, tokens, configuration, frequency))
        return self.output(self.mixer(h.transpose(1, 2)).transpose(1, 2))


class NN(torch.nn.Module):
    """
    Plain MLP on the flattened configuration and the frequency (baseline; trained with resonator-order shuffling).
    """

    def __init__(self, num_res, hidden_dim=168, depth=6, dropout=0.1):
        super().__init__()
        self.num_res = num_res
        layers, n_in = [], num_res * CONF_DIM + 1
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            n_in = hidden_dim
        self.mlp, self.output = torch.nn.Sequential(*layers), torch.nn.Linear(hidden_dim, 1)

    def forward(self, configuration, frequency):
        flat = configuration.reshape(configuration.shape[0], 1, -1).expand(-1, frequency.shape[1], -1)
        return self.output(self.mlp(torch.cat((flat, frequency), dim=-1)))


# name -> (class, learning rate); sizes are about 145k parameters each (2 resonators)
# display order everywhere (menus, tables, legends): DON, DNO, DCO, SIREN, FNO, WNO, LNO, GNO, STO, NN
FORWARD_MODELS = {"DON": (DON, 5e-4), "DNO": (DNO, 5e-4), "DCO": (DCO, 5e-4), "SIREN": (SIREN, 2e-4), "FNO": (FNO, 5e-4),
                  "WNO": (WNO, 5e-4), "LNO": (LNO, 5e-4), "GNO": (GNO, 5e-4), "STO": (STO, 5e-4), "NN": (NN, 5e-4)}


def build_forward_model(name, num_res, norm):
    """
    Forward model with the plate-mode features switched on (default for every architecture).
    """
    model = FORWARD_MODELS[name][0](num_res=num_res)
    if name != "NN":
        enable_physical_features(model)
        set_physical_feature_normalization(model, norm)
    return model.to(device)


#%% 7. Invertible models (design <-> ERP with one set of weights)
# Design = bounded coordinates [m, f_t, x, y, zeta] per resonator (k, c derived), resonators sorted by f_t.
class CouplingBlock(torch.nn.Module):
    """
    v1 <- v1 * S(L(v2)),  v2 <- v2 * S(L(v1)),  S(x) = exp(c tanh(a x)) > 0: exactly invertible.
    """

    def __init__(self, gate_net, gate_scale=2.0):
        super().__init__()
        self.gate_net, self.gate_scale, self.gate_gain = gate_net, gate_scale, torch.nn.Parameter(torch.zeros(1))

    def gate(self, x):
        return torch.exp(self.gate_scale * torch.tanh(self.gate_gain * self.gate_net(x)))

    def forward(self, v1, v2):
        v1 = v1 * self.gate(v2)
        return v1, v2 * self.gate(v1)

    def inverse(self, v1, v2):
        v2 = v2 / self.gate(v1)
        return v1 / self.gate(v2), v2


class CouplingStack(torch.nn.Module):
    def __init__(self, gate_factory, num_blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList([CouplingBlock(gate_factory()) for _ in range(num_blocks)])

    def forward(self, v1, v2):
        for block in self.blocks:
            v1, v2 = block(v1, v2)
        return v1, v2

    def inverse(self, v1, v2):
        for block in reversed(self.blocks):
            v1, v2 = block.inverse(v1, v2)
        return v1, v2


class DesignVAE(torch.nn.Module):
    def __init__(self, design_dim, z_dim=6, hidden=64):
        super().__init__()
        self.encoder = MLP([design_dim, hidden, hidden], torch.nn.SiLU)
        self.mu_head, self.log_var_head = torch.nn.Linear(hidden, z_dim), torch.nn.Linear(hidden, z_dim)
        self.decoder = MLP([z_dim, hidden, hidden, design_dim], torch.nn.SiLU)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.log_var_head(h).clamp(-8.0, 4.0)

    def decode(self, z):
        return self.decoder(z)

    @staticmethod
    def reparameterize(mu, log_var):
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)


def kl_standard_normal(mu, log_var):
    return (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(-1)


class InvertibleOperator(torch.nn.Module):
    """Pipeline shared by iDNO, iDCO, iFNO, iLNO, iGNO, iSTO. A subclass builds `self.blocks` and the two lifts
    (`_lift_forward`: configuration -> latent, `_lift_inverse`: spectrum -> latent), both (B, 2*width, F)."""

    def _init_base(self, num_res, width, act, vae_hidden=64, z_dim=6, readout_bins=16):
        super().__init__()
        self.num_res, self.design_dim, self.width = num_res, N_DESIGN * num_res, width
        f = torch.from_numpy(freqs.astype(np.float32))
        self.register_buffer("frequency_hz", f.clone())
        self.register_buffer("frequency_norm", (f - f.mean()) / f.std())
        for name, size, value in (("config_mean", INV_CONF_DIM, 0.0), ("config_std", INV_CONF_DIM, 1.0), ("b12_mean", N_DESIGN, 0.0), ("b12_std", N_DESIGN, 1.0)):
            self.register_buffer(name, torch.full((size,), value))
        self.project_q = torch.nn.Sequential(torch.nn.Linear(2 * width, 2 * width), act(), torch.nn.Linear(2 * width, 1))
        self.readout_pool, self.readout_bin_proj = torch.nn.AdaptiveAvgPool1d(readout_bins), torch.nn.Conv1d(2 * width, 4, 1)
        self.project_qp = MLP([4 * width + 4 * readout_bins, vae_hidden, self.design_dim], torch.nn.SiLU)
        self.vae = DesignVAE(self.design_dim, z_dim, vae_hidden)

    def _standard_inverse_lift(self, freq_dim, act, activate=True, embed_name="freq_embed"):
        self._embed_name = embed_name
        setattr(self, embed_name, MLP([1, freq_dim, freq_dim], torch.nn.SiLU))
        self.lift_pp = torch.nn.Linear(1 + freq_dim, 2 * self.width)
        self.lift_pp_act = act() if activate else torch.nn.Identity()

    def set_norm(self, norm):
        """
        Dataset statistics needed to decode the bounded design and for the plate-mode features.
        """
        set_physical_feature_normalization(self, norm)
        self.config_mean.copy_(torch.tensor([norm[f"{n}_mean"] for n in FIELDS[:INV_CONF_DIM]]))
        self.config_std.copy_(torch.tensor([norm[f"{n}_std"] for n in FIELDS[:INV_CONF_DIM]]))
        self.b12_mean.copy_(torch.tensor([norm[f"b12_{b[0]}_mean"] for b in BOUNDED]))
        self.b12_std.copy_(torch.tensor([norm[f"b12_{b[0]}_std"] for b in BOUNDED]))

    # -- subclass hooks
    def _lift_forward(self, configuration, frequency):
        raise NotImplementedError

    def _lift_inverse(self, spectrum, frequency):
        lifted = self.lift_pp_act(self.lift_pp(torch.cat((spectrum[..., None], getattr(self, self._embed_name)(frequency)), dim=-1)))
        return lifted.transpose(1, 2)

    # -- pipeline
    def _frequency_batch(self, b):
        return self.frequency_norm[None, :, None].expand(b, -1, -1)

    def design_to_configuration(self, flat):
        physical = decode_bounded_torch(flat, self.num_res, self.b12_mean, self.b12_std)
        return (physical[..., :INV_CONF_DIM] - self.config_mean) / self.config_std

    def _pool(self, latent):
        binned = self.readout_bin_proj(self.readout_pool(latent)).flatten(1)
        return torch.cat((latent.mean(-1), latent.amax(-1), binned), dim=-1)

    def _through_blocks(self, v0):
        return torch.cat(self.blocks(*v0.chunk(2, dim=1)), dim=1)

    def predict_erp(self, flat_design):
        """
        Normalised design (B, D) -> normalised ERP (B, F).
        """
        v0 = self._lift_forward(self.design_to_configuration(flat_design), self._frequency_batch(flat_design.shape[0]))
        return self.project_q(self._through_blocks(v0).transpose(1, 2)).squeeze(-1)

    def point_estimate(self, spectrum):
        """
        Spectrum (B, F) -> design (B, D): lift, inverse coupling blocks, pooled readout.
        """
        u0 = self._lift_inverse(spectrum, self._frequency_batch(spectrum.shape[0]))
        v1, v2 = self.blocks.inverse(*u0.chunk(2, dim=1))
        return self.project_qp(self._pool(torch.cat((v1, v2), dim=1)))

    @torch.no_grad()
    def sample(self, spectrum, num_samples=8):
        """
        (B, S, D) designs: point estimate -> VAE posterior samples.
        """
        mu, log_var = self.vae.encode(self.point_estimate(spectrum))
        b = spectrum.shape[0]
        z = mu[:, None] + torch.exp(0.5 * log_var)[:, None] * torch.randn(b, num_samples, mu.shape[-1], device=mu.device)
        return self.vae.decode(z)

    # -- stage losses (Long et al., arXiv:2402.11722): MSE on normalised quantities
    def stage1_loss(self, spectrum, design, cycle_weight=0.1, align_weight=0.1):
        b = spectrum.shape[0]
        frequency = self._frequency_batch(b)
        v0 = self._lift_forward(self.design_to_configuration(design), frequency)
        vk = self._through_blocks(v0)
        predicted = self.project_q(vk.transpose(1, 2)).squeeze(-1)
        u0 = self._lift_inverse(spectrum, frequency)
        v1, v2 = self.blocks.inverse(*u0.chunk(2, dim=1))
        estimate = self.project_qp(self._pool(torch.cat((v1, v2), dim=1)))
        loss = (torch.nn.functional.mse_loss(predicted, spectrum) + torch.nn.functional.mse_loss(estimate, design)
                + torch.nn.functional.mse_loss(self.project_qp(self._pool(v0)), design) + torch.nn.functional.mse_loss(self.project_q(u0.transpose(1, 2)).squeeze(-1), spectrum))
        if cycle_weight > 0:
            loss = loss + cycle_weight * torch.nn.functional.mse_loss(self.point_estimate(predicted), design)
        if align_weight > 0:
            loss = loss + align_weight * ((u0 - vk) ** 2).mean() / (vk.detach() ** 2).mean().clamp_min(1e-8)
        return loss

    def stage2_loss(self, design, beta, encoder_input=None):
        mu, log_var = self.vae.encode(design if encoder_input is None else encoder_input)
        recon = self.vae.decode(self.vae.reparameterize(mu, log_var))
        return torch.nn.functional.mse_loss(recon, design) + beta * kkl(mu, log_var)

    def stage3_loss(self, spectrum, design, beta, inverse_weight=1.0, cycle_weight=0.1, align_weight=0.1):
        frequency = self._frequency_batch(spectrum.shape[0])
        v0 = self._lift_forward(self.design_to_configuration(design), frequency)
        vk = self._through_blocks(v0)
        predicted = self.project_q(vk.transpose(1, 2)).squeeze(-1)
        u0 = self._lift_inverse(spectrum, frequency)
        v1, v2 = self.blocks.inverse(*u0.chunk(2, dim=1))
        estimate = self.project_qp(self._pool(torch.cat((v1, v2), dim=1)))
        mu, log_var = self.vae.encode(estimate)
        recon = self.vae.decode(self.vae.reparameterize(mu, log_var))
        loss = (torch.nn.functional.mse_loss(predicted, spectrum) + torch.nn.functional.mse_loss(self.project_qp(self._pool(v0)), design)
                + torch.nn.functional.mse_loss(self.project_q(u0.transpose(1, 2)).squeeze(-1), spectrum)
                + torch.nn.functional.mse_loss(recon, design) + beta * kkl(mu, log_var) + inverse_weight * torch.nn.functional.mse_loss(estimate, design))
        if cycle_weight > 0:
            loss = loss + cycle_weight * torch.nn.functional.mse_loss(self.point_estimate(predicted), design)
        if align_weight > 0:
            loss = loss + align_weight * ((u0 - vk) ** 2).mean() / (vk.detach() ** 2).mean().clamp_min(1e-8)
        return loss


def kkl(mu, log_var):
    return kl_standard_normal(mu, log_var).mean()


class FourierGate(torch.nn.Module):
    """
    Gate L of iFNO: kernel-3 convolution + spectral convolution (zero-padded FFT), activation.
    """

    def __init__(self, width, modes, padding=8):
        super().__init__()
        self.spectral, self.local, self.norm = SpectralConv1d(width, width, modes, padding), torch.nn.Conv1d(width, width, 3, padding=1), torch.nn.GroupNorm(1, width)

    def forward(self, x):
        return torch.nn.functional.gelu(self.norm(self.local(x) + self.spectral(x)))


class IFNO(InvertibleOperator):
    def __init__(self, num_res, width=24, num_blocks=4, modes=32, config_hidden=64, query_dim=32):
        self._init_base(num_res, width, torch.nn.GELU)
        self.configuration_encoder = SortedResonatorEncoder(num_res, config_hidden, 2 * width, conf_dim=INV_CONF_DIM)
        self.resonance_query = ResonanceQueryEncoder(query_dim, query_dim, query_dim, conf_dim=INV_CONF_DIM)
        self.lift_p = torch.nn.Linear(2 * width + query_dim + 1, 2 * width)
        self._standard_inverse_lift(query_dim, torch.nn.GELU, activate=False)
        self.blocks = CouplingStack(lambda: FourierGate(width, modes), num_blocks)

    def _lift_forward(self, configuration, frequency):
        context = self.configuration_encoder(configuration)[:, None].expand(-1, frequency.shape[1], -1)
        return self.lift_p(torch.cat((context, self.resonance_query(configuration, frequency), frequency), dim=-1)).transpose(1, 2)


class DCOGate(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.residual_block, self.refine = ResidualMLPBlock(width, torch.nn.SiLU), FrequencyRefinement1d(width)

    def forward(self, x):
        return self.refine(self.residual_block(x.transpose(1, 2)).transpose(1, 2))


class IDCO(InvertibleOperator):
    def __init__(self, num_res, width=42, num_blocks=4, branch_dim=72, trunk_dim=28, query_dim=32):
        self._init_base(num_res, width, torch.nn.SiLU)
        self.branch = SortedResonatorEncoder(num_res, branch_dim, branch_dim, conf_dim=INV_CONF_DIM)
        self.trunk = MLP([1, trunk_dim, trunk_dim], torch.nn.SiLU)
        self.resonance_query = ResonanceQueryEncoder(query_dim, query_dim, query_dim, conf_dim=INV_CONF_DIM)
        self.lift_p = torch.nn.Linear(branch_dim + trunk_dim + query_dim, 2 * width)
        self._standard_inverse_lift(trunk_dim, torch.nn.SiLU, embed_name="trunk_inverse")
        self.blocks = CouplingStack(lambda: DCOGate(width), num_blocks)

    def _lift_forward(self, configuration, frequency):
        branch = self.branch(configuration)[:, None].expand(-1, frequency.shape[1], -1)
        h = torch.cat((branch, self.trunk(frequency), self.resonance_query(configuration, frequency)), dim=-1)
        return torch.nn.functional.silu(self.lift_p(h)).transpose(1, 2)


class DNOGate(torch.nn.Module):
    """
    Gate of iDNO: FiLM residual block conditioned on a learned embedding of the frequency, then refinement.
    """

    def __init__(self, width, frequency_norm, condition_dim=16):
        super().__init__()
        self.register_buffer("f", frequency_norm.clone()[:, None])
        self.condition = MLP([1, condition_dim, condition_dim], torch.nn.SiLU)
        self.block, self.refine = FiLMResidualBlock(width, condition_dim, torch.nn.SiLU), FrequencyRefinement1d(width)

    def forward(self, x):
        h = self.block(x.transpose(1, 2), self.condition(self.f)[None])
        return self.refine(h.transpose(1, 2))


class IDNO(InvertibleOperator):
    def __init__(self, num_res, width=40, num_blocks=4, context_dim=64, config_hidden=80, frequency_dim=24, query_dim=24):
        self._init_base(num_res, width, torch.nn.SiLU)
        self.configuration_encoder = SortedResonatorEncoder(num_res, config_hidden, context_dim, conf_dim=INV_CONF_DIM)
        self.frequency_encoder = MLP([1, frequency_dim, frequency_dim], torch.nn.SiLU)
        self.resonance_query = ResonanceQueryEncoder(query_dim, query_dim, query_dim, conf_dim=INV_CONF_DIM)
        self.lift_p = torch.nn.Linear(context_dim + frequency_dim + query_dim, 2 * width)
        self._standard_inverse_lift(frequency_dim, torch.nn.SiLU)
        self.blocks = CouplingStack(lambda: DNOGate(width, self.frequency_norm), num_blocks)

    def _lift_forward(self, configuration, frequency):
        context = self.configuration_encoder(configuration)[:, None].expand(-1, frequency.shape[1], -1)
        h = torch.cat((context, self.frequency_encoder(frequency), self.resonance_query(configuration, frequency)), dim=-1)
        return torch.nn.functional.silu(self.lift_p(h)).transpose(1, 2)


class GNOGate(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.pointwise, self.refine = MLP([width, width, width], torch.nn.SiLU), FrequencyRefinement1d(width)

    def forward(self, x):
        return self.refine(self.pointwise(x.transpose(1, 2)).transpose(1, 2))


class IGNO(InvertibleOperator):
    uses_modal_features = True
    uses_detuning = True

    def __init__(self, num_res, width=42, num_blocks=4, depth=3, frequency_dim=28, harmonics=4, dropout=0.1):
        self._init_base(num_res, width, torch.nn.SiLU)
        self.harmonics = harmonics
        self.node_lift = MLP([feature_dim(harmonics, INV_CONF_DIM), width, width], torch.nn.SiLU)
        self.graph_layers = torch.nn.ModuleList([GraphMessageLayer(width, dropout, INV_CONF_DIM) for _ in range(depth)])
        self.query_kernel = MLP([width + INV_CONF_DIM + frequency_dim + 4, width, width, width], torch.nn.SiLU)
        self.attention_score = MLP([width, width // 2, 1], torch.nn.SiLU)
        self.attention_dropout, self.lift_p = torch.nn.Dropout(dropout), torch.nn.Linear(width, 2 * width)
        self._standard_inverse_lift(frequency_dim, torch.nn.SiLU, embed_name="frequency_encoder")
        self.blocks = CouplingStack(lambda: GNOGate(width), num_blocks)

    def _lift_forward(self, configuration, frequency):
        h = self.node_lift(modal_features(self, configuration, self.harmonics))
        for layer in self.graph_layers:
            h = layer(h, configuration)
        b, f, _ = frequency.shape
        n = configuration.shape[1]
        query = frequency[:, :, None].expand(b, f, n, 1)
        delta = resonance_detuning(self, query, configuration[:, None, :, 2:3].expand(b, f, n, 1))
        pair = torch.cat((h[:, None].expand(b, f, n, -1), configuration[:, None].expand(b, f, n, -1),
                          self.frequency_encoder(frequency)[:, :, None].expand(b, f, n, -1), query, delta, delta.abs(), delta.square()), dim=-1)
        kernel = self.query_kernel(pair)
        weights = self.attention_dropout(torch.softmax(self.attention_score(kernel), dim=2))
        return torch.nn.functional.silu(self.lift_p((weights * kernel).sum(2))).transpose(1, 2)


class LNOGate(torch.nn.Module):
    """
    Gate of iLNO: fixed learned pole features of the frequency + pointwise transform + refinement.
    """

    def __init__(self, width, frequency_norm, num_poles=8):
        super().__init__()
        self.register_buffer("f", frequency_norm.clone())
        self.omega = torch.nn.Parameter(torch.linspace(float(frequency_norm.min()), float(frequency_norm.max()), num_poles))
        self.log_sigma = torch.nn.Parameter(torch.full((num_poles,), -2.0))
        self.r_re, self.r_im = torch.nn.Parameter(0.1 * torch.randn(num_poles)), torch.nn.Parameter(0.1 * torch.randn(num_poles))
        self.pointwise, self.pole_proj, self.refine = torch.nn.Conv1d(width, width, 1), torch.nn.Linear(2 * num_poles, width), FrequencyRefinement1d(width)

    def forward(self, x):
        feats = pole_features(self.f, torch.nn.functional.softplus(self.log_sigma) + 1e-3, self.omega, self.r_re, self.r_im)
        return self.refine(torch.nn.functional.silu(self.pointwise(x) + self.pole_proj(feats).T[None]))


class ILNO(InvertibleOperator):
    def __init__(self, num_res, width=44, num_blocks=4, num_poles=12, gate_poles=8, context_dim=48, config_hidden=80, pole_hidden=48, query_dim=24):
        self._init_base(num_res, width, torch.nn.SiLU)
        self.num_poles = num_poles
        self.configuration_encoder = SortedResonatorEncoder(num_res, config_hidden, context_dim, conf_dim=INV_CONF_DIM)
        self.pole_predictor = MLP([context_dim, pole_hidden, pole_hidden, 4 * num_poles], torch.nn.SiLU)
        self.resonance_query = ResonanceQueryEncoder(query_dim, query_dim, query_dim, conf_dim=INV_CONF_DIM)
        self.lift_p = torch.nn.Linear(2 * num_poles + query_dim + 1, 2 * width)
        self._standard_inverse_lift(query_dim, torch.nn.SiLU)
        self.blocks = CouplingStack(lambda: LNOGate(width, self.frequency_norm, gate_poles), num_blocks)

    def _lift_forward(self, configuration, frequency):
        raw = self.pole_predictor(self.configuration_encoder(configuration)).view(-1, 1, self.num_poles, 4)
        poles = pole_features(frequency[..., 0], torch.nn.functional.softplus(raw[..., 0]) + 1e-3, torch.nn.functional.softplus(raw[..., 1]), raw[..., 2], raw[..., 3])
        return torch.nn.functional.silu(self.lift_p(torch.cat((poles, self.resonance_query(configuration, frequency), frequency), dim=-1))).transpose(1, 2)


class ISTO(InvertibleOperator):
    uses_modal_features = True

    def __init__(self, num_res, width=28, num_blocks=4, token_width=64, heads=4, encoder_depth=2, ff_dim=96, harmonics=4, frequency_dim=24):
        self._init_base(num_res, width, torch.nn.GELU)
        self.harmonics = harmonics
        self.node_lift = MLP([feature_dim(harmonics, INV_CONF_DIM), token_width, token_width], torch.nn.GELU)
        layer = torch.nn.TransformerEncoderLayer(token_width, heads, ff_dim, 0.0, activation=torch.nn.GELU(), batch_first=True)
        self.token_encoder = torch.nn.TransformerEncoder(layer, encoder_depth)
        self.frequency_query = MLP([1, token_width, token_width], torch.nn.GELU)
        self.cross_attention = DetuningCrossAttention(token_width, heads, 0.0, torch.nn.GELU, INV_CONF_DIM)
        self.cross_norm, self.lift_p = torch.nn.LayerNorm(token_width), torch.nn.Linear(token_width, 2 * width)
        self._standard_inverse_lift(frequency_dim, torch.nn.GELU)
        self.blocks = CouplingStack(lambda: FrequencyMixer(width), num_blocks)

    def _lift_forward(self, configuration, frequency):
        tokens = self.token_encoder(self.node_lift(modal_features(self, configuration, self.harmonics)))
        query = self.frequency_query(frequency)
        h = self.cross_norm(query + self.cross_attention(query, tokens, configuration, frequency))
        return torch.nn.functional.gelu(self.lift_p(h)).transpose(1, 2)


# --- invertible DeepONet (Kaltenbach et al.): ERP = sum_q b_q(a) psi_q(f) + psi_0(f), b = RealNVP(a) ---
class AffineCoupling(torch.nn.Module):
    def __init__(self, dim, mask, hidden, depth=2):
        super().__init__()
        self.register_buffer("mask", mask)
        layers, width = [], dim
        for _ in range(depth):
            layers += [torch.nn.Linear(width, hidden), torch.nn.SiLU()]
            width = hidden
        layers.append(torch.nn.Linear(width, 2 * dim))
        self.net = torch.nn.Sequential(*layers)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def _st(self, x):
        log_s, t = self.net(x).chunk(2, dim=-1)
        return torch.tanh(log_s), t

    def forward(self, x):
        xm = x * self.mask
        log_s, t = self._st(xm)
        return xm + (1 - self.mask) * (x * torch.exp(log_s) + t)

    def inverse(self, y):
        ym = y * self.mask
        log_s, t = self._st(ym)
        return ym + (1 - self.mask) * ((y - t) * torch.exp(-log_s))


class RealNVP(torch.nn.Module):
    def __init__(self, dim, num_layers=10, hidden=256):
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        layers = []
        for i in range(num_layers):
            mask = torch.zeros(dim)
            if dim <= 16:
                mask[i % 2::2] = 1.0
            else:
                mask[torch.randperm(dim, generator=gen)[: dim // 2]] = 1.0
            layers.append(AffineCoupling(dim, mask, hidden))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, y):
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y


class FourierTrunk(torch.nn.Module):
    """
    Trunk of iDON: MLP of [f, sin(k pi f), cos(k pi f)], k = 1..num_fourier, for the frequency f in [-1, 1].
    """

    def __init__(self, num_fourier, hidden, out_dim):
        super().__init__()
        self.register_buffer("freqs", torch.arange(1, num_fourier + 1, dtype=torch.float32) * math.pi)
        self.net = MLP([1 + 2 * num_fourier, hidden, hidden, hidden, hidden, out_dim], torch.nn.SiLU).net

    def forward(self, f):
        arg = f * self.freqs[None]
        return self.net(torch.cat((f, torch.sin(arg), torch.cos(arg)), dim=-1))


class IDON(torch.nn.Module):
    """Branch = RealNVP on the design padded to Q basis functions (Q = 64: padded latent, Q = 8: no padding for 2 resonators),
    trunk = MLP of the frequency. The inverse is a closed-form projection onto the orthonormalised basis followed by RealNVP^-1."""

    def __init__(self, num_res, q=64, num_layers=6, hidden=56, trunk_hidden=120, num_fourier=32):
        super().__init__()
        self.num_res, self.design_dim = num_res, N_DESIGN * num_res
        q = max(q, self.design_dim)
        self.q = q
        self.pad = q - self.design_dim
        self.branch = RealNVP(q, num_layers, hidden)
        self.trunk = FourierTrunk(num_fourier, trunk_hidden, q + 1)
        self.register_buffer("f_norm", torch.linspace(-1.0, 1.0, n_freq))
        self.register_buffer("noise_var", torch.tensor(1.0))

    def set_norm(self, norm):
        pass

    def basis(self):
        out = self.trunk(self.f_norm[:, None])
        qmat, _ = torch.linalg.qr(out[:, 1:])
        return qmat * math.sqrt(n_freq), out[:, 0]

    def coefficients(self, a):
        return self.branch(torch.nn.functional.pad(a, (0, self.pad)))

    def predict_erp(self, a):
        psi, psi0 = self.basis()
        return self.coefficients(a) @ psi.T + psi0

    def least_squares(self, spectrum, psi=None, psi0=None):
        if psi is None:
            psi, psi0 = self.basis()
        return (spectrum - psi0) @ psi / n_freq

    def invert(self, b):
        x = self.branch.inverse(b)
        return x[..., : self.design_dim], x[..., self.design_dim:]

    def point_estimate(self, spectrum):
        return self.invert(self.least_squares(spectrum))[0]

    @torch.no_grad()
    def sample(self, spectrum, num_samples=8):
        b_star = self.least_squares(spectrum)
        offsets = torch.randn(b_star.shape[0], num_samples, self.q, device=b_star.device) * torch.sqrt(self.noise_var / n_freq)
        offsets[:, 0] = 0.0
        return self.invert(b_star[:, None] + offsets)[0]

    def training_loss(self, spectrum, design, inverse_weight=1.0, latent_weight=0.1, slope_weight=0.0, peak_weight=0.0):
        psi, psi0 = self.basis()
        pred = self.coefficients(design) @ psi.T + psi0
        forward = erp_spectrum_loss(pred[..., None], spectrum[..., None], slope_weight, peak_weight)
        a_hat, z_hat = self.invert(self.least_squares(spectrum, psi, psi0))
        loss = forward + inverse_weight * torch.nn.functional.smooth_l1_loss(a_hat, design)
        if z_hat.numel() > 0:   # no padded latent dimensions for Q = 8 with two resonators
            loss = loss + latent_weight * torch.nn.functional.smooth_l1_loss(z_hat, torch.zeros_like(z_hat))
        return loss


INVERTIBLE_MODELS = {"iDON": IDON, "iDON_Q8": functools.partial(IDON, q=8, num_layers=8, hidden=64, trunk_hidden=168), "iDNO": IDNO, "iDCO": IDCO, "iFNO": IFNO, "iLNO": ILNO, "iGNO": IGNO, "iSTO": ISTO}
LEGACY_IDON_CONFIG = dict(num_layers=10, hidden=256, trunk_hidden=256, num_fourier=32)   # iDON checkpoints saved without a model_config
MODEL_CONFIG_NAMES = {"modal_harmonics": "harmonics"}   # names in a checkpoint's model_config -> constructor arguments


def is_idon(name):
    return name.startswith("iDON")


def build_invertible_model(name, num_res, norm, config=None):
    """Invertible model; `config` = hyper-parameters stored with a checkpoint (model_config), the defaults otherwise.
    The resolved hyper-parameters are kept in model.model_config and saved with the weights."""
    factory = INVERTIBLE_MODELS[name]
    parameters = inspect.signature(factory).parameters
    resolved = {k: v.default for k, v in parameters.items() if k != "num_res" and v.default is not inspect.Parameter.empty}
    resolved.update({MODEL_CONFIG_NAMES.get(k, k): v for k, v in (config or {}).items() if MODEL_CONFIG_NAMES.get(k, k) in resolved})
    model = factory(num_res=num_res, **resolved)
    model.model_config = resolved
    if not is_idon(name):
        enable_physical_features(model)
        model.set_norm(norm)
    return model.to(device)


#%% 8. Training, validation, prediction and plots
#%% Losses and metrics
def spectrum_peak_mask(spectrum, window=7):
    """
    (B, F, 1) mask: local maxima of the spectrum (non-maximum suppression over `window` bins) and its global maximum.
    """
    x = spectrum.detach().transpose(1, 2)
    ninf = float("-inf")
    left, right = torch.nn.functional.pad(x, (1, 0), value=ninf)[..., :-1], torch.nn.functional.pad(x, (0, 1), value=ninf)[..., 1:]
    candidate = (x > left) & (x > right)
    pooled = torch.nn.functional.max_pool1d(torch.where(candidate, x, torch.full_like(x, ninf)), window, 1, window // 2)
    mask = (candidate & (x >= pooled)).transpose(1, 2)
    return mask.scatter(1, spectrum.argmax(dim=1, keepdim=True), True)


def erp_spectrum_loss(prediction, target, slope_weight=0.5, peak_weight=0.05):
    """
    MSE + slope (first difference) MSE + squared error summed over the true peaks; inputs (B, F, 1).
    """
    loss = torch.nn.functional.mse_loss(prediction, target)
    if slope_weight > 0:
        loss = loss + slope_weight * torch.nn.functional.mse_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1])
    if peak_weight > 0:
        mask = spectrum_peak_mask(target).to(prediction.dtype)
        loss = loss + peak_weight * (((prediction - target) ** 2 * mask).sum(1).squeeze(-1)).mean()
    return loss


def spectrum_metrics(pred, true):
    """
    Error measures (dB) of predicted vs true ERP spectra, arrays (n, F).
    """
    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    err = pred - true
    mask = spectrum_peak_mask(torch.from_numpy(true)[..., None]).squeeze(-1).numpy()
    top = true.argmax(1)
    rows = np.arange(len(true))
    per_spectrum = [np.corrcoef(p, t)[0, 1] for p, t in zip(pred, true)]
    return {"rmse_db": float(np.sqrt((err**2).mean())), "mae_db": float(np.abs(err).mean()),
            "r2": float(1 - (err**2).sum() / ((true - true.mean()) ** 2).sum()), "pearson_r": float(np.corrcoef(pred.ravel(), true.ravel())[0, 1]),
            "pearson_r_per_spectrum": float(np.nanmean(per_spectrum)), "rmse_at_peaks_db": float(np.sqrt((err[mask] ** 2).mean())),
            "rmse_off_peaks_db": float(np.sqrt((err[~mask] ** 2).mean())), "error_at_highest_peak_db": float(np.abs(err[rows, top]).mean())}


#%% File helpers
def kind_dir(kind):
    return FORWARD_DIR if kind == "forward" else INVERTIBLE_DIR


def model_path(kind, tag, name):
    return kind_dir(kind) / "models" / tag / f"{name}.pth"


def plot_dir(kind, tag, name):
    path = kind_dir(kind) / "plots" / tag / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def group_folder(kind, names):
    """
    Folder for the plots of several models: ALL_MODELS for every model, otherwise COMPARE_<names>.
    """
    every = FORWARD_MODELS if kind == "forward" else INVERTIBLE_MODELS
    return "ALL_MODELS" if set(names) == set(every) else "COMPARE_" + "-".join(names)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


#%% Forward model training
def train_forward(name, data, epochs=epochs_forward, batch_size=batch_size):
    """
    AdamW + cosine schedule, loss = MSE + slope + peak term, best-validation weights kept; resumable every 5 epochs.
    """
    loaders = data.loaders("forward", batch_size)
    model = build_forward_model(name, data.num_res, data.norm)
    lr = FORWARD_MODELS[name][1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_forward)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min_forward)
    path = model_path("forward", data.tag, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    resume = path.with_suffix(".resume.pt")
    history, best_val, best_state, start = {"train": [], "val": []}, math.inf, None, 0
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["epochs"] == epochs:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history, best_val, best_state, start = state["history"], state["best_val"], state["best_state"], state["epoch"]
            print(f"Resuming {name} at epoch {start + 1}/{epochs}")
    print(f"\n=== Training {name} ({parameter_count(model):,} parameters, {len(loaders['train'].dataset)} training configurations) ===")
    start_time = time.time()
    for epoch in range(start, epochs):
        model.train()
        total = count = 0
        for configuration, frequency, target in loaders["train"]:
            configuration, frequency, target = configuration.to(device), frequency.to(device), target.to(device)
            if name == "NN":  # random resonator order each batch
                order = torch.argsort(torch.rand(configuration.shape[:2], device=device), dim=1)
                configuration = torch.gather(configuration, 1, order[..., None].expand(-1, -1, CONF_DIM))
            optimizer.zero_grad(set_to_none=True)
            loss = erp_spectrum_loss(model(configuration, frequency), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_forward)
            optimizer.step()
            total, count = total + loss.item() * len(configuration), count + len(configuration)
        scheduler.step()
        model.eval()
        val_total = val_count = 0
        with torch.no_grad():
            for configuration, frequency, target in loaders["val"]:
                configuration, frequency, target = configuration.to(device), frequency.to(device), target.to(device)
                val_total += erp_spectrum_loss(model(configuration, frequency), target).item() * len(configuration)
                val_count += len(configuration)
        history["train"].append(total / count)
        history["val"].append(val_total / val_count)
        if history["val"][-1] < best_val:
            best_val, best_state = history["val"][-1], copy.deepcopy(model.state_dict())
        print(f"{name} epoch {epoch + 1:4d}/{epochs} | train {history['train'][-1]:.5f} | val {history['val'][-1]:.5f}")
        if (epoch + 1) % resume_every == 0 and epoch + 1 < epochs:
            torch.save(dict(epoch=epoch + 1, epochs=epochs, model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), history=history, best_val=best_val, best_state=best_state), resume)
    elapsed = time.time() - start_time
    print(f"Training complete in {int(elapsed // 3600)}h {int(elapsed % 3600 // 60)}m {int(elapsed % 60)}s.")
    model.load_state_dict(best_state)
    torch.save(dict(model_state_dict=model.state_dict(), norm=data.norm, name=name, tag=data.tag, num_res=data.num_res, history=history,
                    training=dict(epochs=epochs, batch_size=batch_size, lr=lr, loss="MSE + 0.5 slope MSE + 0.05 peak squared error")), path)
    resume.unlink(missing_ok=True)
    print(f"Saved {path}")
    return model


#%% Invertible model training

def run_stage(model, loaders, epochs, lr, parameters, loss_fn, state_module, label):
    """
    Adam + cosine, gradient clipping, best-validation weights, early stop. loss_fn(spectrum, design, epoch, final).
    """
    parameters = list(parameters)
    history = {"train": [], "val": []}
    if epochs <= 0:
        return history
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.01 * lr)
    best_val, best_state = math.inf, None
    for epoch in range(epochs):
        model.train()
        total = count = 0
        for spectrum, design in loaders["train"]:
            spectrum, design = spectrum.to(device), design.to(device).flatten(1)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(spectrum, design, epoch, False)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()
            total, count = total + loss.item() * len(spectrum), count + len(spectrum)
        scheduler.step()
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            torch.manual_seed(20251001)  # fixed VAE noise in validation
            val_total = val_count = 0
            for spectrum, design in loaders["val"]:
                spectrum, design = spectrum.to(device), design.to(device).flatten(1)
                value = loss_fn(spectrum, design, epoch, True)
                if torch.isfinite(value):
                    val_total, val_count = val_total + value.item() * len(spectrum), val_count + len(spectrum)
        history["train"].append(total / max(count, 1))
        history["val"].append(val_total / max(val_count, 1))
        if history["val"][-1] < best_val:
            best_val, best_state = history["val"][-1], copy.deepcopy(state_module.state_dict())
        print(f"{label} epoch {epoch + 1:3d}/{epochs} | train {history['train'][-1]:.4f} | val {history['val'][-1]:.4f}")
        val = history["val"]
        if (epoch + 1 >= max(20, epochs // 2) and epoch + 1 < epochs and len(val) > EARLY_STOP_PATIENCE
                and min(val[-EARLY_STOP_PATIENCE:]) >= min(val[:-EARLY_STOP_PATIENCE])):
            print(f"{label}: no validation improvement for {EARLY_STOP_PATIENCE} epochs, stopping at epoch {epoch + 1}")
            break
    if best_state is not None:
        state_module.load_state_dict(best_state)
    return history


def train_invertible(name, data, epochs, batch_size=batch_size):
    """iFNO family: stage 1 (invertible core + lifts + readouts), stage 2 (VAE on stage-1 estimates), stage 3 (joint).
    iDON: forward loss (MSE + slope + peak term from 80 % of the epochs) + inverse-consistency loss.
    `epochs` is (stage1, stage2, stage3) for the family and an int for iDON."""
    loaders = data.loaders("inverse", batch_size)
    model = build_invertible_model(name, data.num_res, data.norm)
    path = model_path("invertible", data.tag, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Training {name} ({parameter_count(model):,} parameters, {len(loaders['train'].dataset)} training configurations) ===")
    start_time = time.time()
    if is_idon(name):
        peak_start = int(round(IDON_PEAK_START * epochs))

        def loss_fn(spectrum, design, epoch, final):
            w = 1.0 if final else min(1.0, (epoch + 1) / IDON_INVERSE_WARMUP)
            peak = IDON_PEAK if final or epoch >= peak_start else 0.0
            return model.training_loss(spectrum, design, inverse_weight=w, slope_weight=IDON_SLOPE, peak_weight=peak)

        history = run_stage(model, loaders, epochs, 5e-4, model.parameters(), loss_fn, model, name)
        model.eval()
        with torch.no_grad():  # noise variance of the coefficient posterior = training residual
            err = n = 0
            for i, (spectrum, design) in enumerate(loaders["train"]):
                if i >= 200:
                    break
                spectrum, design = spectrum.to(device), design.to(device).flatten(1)
                err, n = err + ((model.predict_erp(design) - spectrum) ** 2).sum().item(), n + spectrum.numel()
            model.noise_var.fill_(err / n)
    else:
        e1, e2, e3 = epochs
        non_vae = [p for key, p in model.named_parameters() if not key.startswith("vae.")]
        h1 = run_stage(model, loaders, e1, STAGE_LR[0], non_vae,
                       lambda s, d, e, f: model.stage1_loss(s, d, CYCLE_WEIGHT, ALIGN_WEIGHT), model, f"{name} stage 1")

        def stage2(spectrum, design, epoch, final):
            beta = KL_TARGET if final else KL_TARGET * min(1.0, epoch / KL_WARMUP)
            with torch.no_grad():
                estimate = model.point_estimate(spectrum)
            return model.stage2_loss(design, beta, estimate)

        h2 = run_stage(model, loaders, e2, STAGE_LR[1], model.vae.parameters(), stage2, model.vae, f"{name} stage 2")
        h3 = run_stage(model, loaders, e3, STAGE_LR[2], model.parameters(),
                       lambda s, d, e, f: model.stage3_loss(s, d, KL_TARGET, 1.0, CYCLE_WEIGHT, ALIGN_WEIGHT), model, f"{name} stage 3")
        history = {"stage1": h1, "stage2": h2, "stage3": h3}
    elapsed = time.time() - start_time
    print(f"Training complete in {int(elapsed // 3600)}h {int(elapsed % 3600 // 60)}m {int(elapsed % 60)}s.")
    torch.save(dict(model_state_dict=model.state_dict(), norm=data.norm, name=name, tag=data.tag, num_res=data.num_res, history=history,
                    model_config=model.model_config, training=dict(epochs=epochs, batch_size=batch_size)), path)
    print(f"Saved {path}")
    return model


#%% Loading of trained models
def checkpoint_norm(ckpt):
    """
    Normalisation statistics of a checkpoint ('norm', or 'norm_params' in checkpoints saved by earlier versions).
    """
    return ckpt["norm"] if "norm" in ckpt else {k: v for k, v in ckpt["norm_params"].items() if k != "feature_names"}


def checkpoint_num_res(ckpt):
    return int(ckpt["num_res"] if "num_res" in ckpt else ckpt["model_config"]["num_res"])


def load_model(kind, tag, name):
    path = model_path(kind, tag, name)
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_res, norm = checkpoint_num_res(ckpt), checkpoint_norm(ckpt)
    if kind == "forward":
        model = build_forward_model(name, num_res, norm)
    else:
        config = ckpt.get("model_config") or (LEGACY_IDON_CONFIG if is_idon(name) else None)
        model = build_invertible_model(name, num_res, norm, config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), ckpt


def trained_models(kind, tag):
    names = FORWARD_MODELS if kind == "forward" else INVERTIBLE_MODELS
    return [n for n in names if model_path(kind, tag, n).exists()]


#%% Validation (test split) and plots
@torch.no_grad()
def predict_test(kind, model, data, loaders):
    """
    True and predicted ERP (dB) of the test split.
    """
    preds, trues = [], []
    for batch in loaders["test"]:
        if kind == "forward":
            configuration, frequency, target = (t.to(device) for t in batch)
            preds.append(model(configuration, frequency).squeeze(-1).cpu())
            trues.append(target.squeeze(-1).cpu())
        else:
            spectrum, design = (t.to(device) for t in batch)
            preds.append(model.predict_erp(design.flatten(1)).cpu())
            trues.append(spectrum.cpu())
    return denormalize_erp(torch.cat(trues), data.norm), denormalize_erp(torch.cat(preds), data.norm)


def plot_loss(history, name, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if "train" in history:
        ax.plot(history["train"], label="Training")
        ax.plot(history["val"], label="Validation")
    else:  # three stages one after the other
        offset = 0
        for color, (stage, label) in zip(("C0", "C1", "C2"), (("stage1", "Stage 1"), ("stage2", "Stage 2 (VAE)"), ("stage3", "Stage 3"))):
            n = len(history[stage]["train"])
            ax.plot(range(offset, offset + n), history[stage]["train"], color + "--", lw=1)
            ax.plot(range(offset, offset + n), history[stage]["val"], color, label=label)
            offset += n
    ax.set(xlabel="Epoch", ylabel="Loss", yscale="log", title=f"{name}: training history")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_forward_results(out, name, true, pred, frequencies):
    """
    Parity plot, six test spectra (best to worst), error versus frequency.
    """
    rng = np.random.default_rng(SEED)
    pick = rng.choice(true.size, min(30000, true.size), replace=False)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(true.ravel()[pick], pred.ravel()[pick], s=2, alpha=0.3)
    lim = [true.min(), true.max()]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel=f"True {ERP_LABEL}", ylabel=f"Predicted {ERP_LABEL}", title=f"{name}: prediction vs truth", aspect="equal")
    fig.tight_layout()
    fig.savefig(out / "prediction_vs_truth.png")
    plt.close(fig)

    rmse = np.sqrt(((pred - true) ** 2).mean(1))
    order = np.argsort(rmse)
    chosen = order[np.linspace(0, len(order) - 1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, i, label in zip(axes.ravel(), chosen, ("best", "", "", "", "", "worst")):
        ax.plot(frequencies, true[i], "k", lw=1.6, label="Solver")
        ax.plot(frequencies, pred[i], "C3", lw=1.2, label=name)
        ax.set(title=f"RMSE {rmse[i]:.2f} dB {label}".strip())
    axes[1, 0].set_xlabel(FREQ_LABEL)
    axes[0, 0].set_ylabel(ERP_LABEL)
    axes[1, 0].set_ylabel(ERP_LABEL)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "test_examples.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frequencies, np.sqrt(((pred - true) ** 2).mean(0)))
    ax.set(xlabel=FREQ_LABEL, ylabel="RMSE (dB)", title=f"{name}: error versus frequency")
    fig.tight_layout()
    fig.savefig(out / "error_vs_frequency.png")
    plt.close(fig)


def _place_labels(ax, labels, markers, w=0.24, h=0.075):
    """Write each label next to its marker on a side that overlaps neither other labels, markers nor the frame.
    labels: list of (x, y, text, colour); markers: all marker positions (x, y).
    Returns (text artists, True if every label found a free spot)."""
    r = 0.035  # marker radius (data units; the plate axes are equal-aspect)
    boxes = [(mx - r, my - r, mx + r, my + r) for mx, my in markers]
    xl, xr = ax.get_xlim()
    yl, yr = ax.get_ylim()
    sides = [(0.03, 0.025, "left"), (0.03, -0.025 - h, "left"), (-0.03, 0.025, "right"), (-0.03, -0.025 - h, "right"),
             (0.0, 0.05, "center"), (0.0, -0.05 - h, "center"), (0.0, 0.12, "center"), (0.0, -0.12 - h, "center")]
    texts, ok = [], True
    for x, y, text, color in labels:
        chosen = None
        for dx, dy, ha in sides:
            x0 = x + dx - (w if ha == "right" else w / 2 if ha == "center" else 0)
            box = (x0, y + dy, x0 + w, y + dy + h)
            inside = xl <= box[0] and box[2] <= xr and yl <= box[1] and box[3] <= yr
            free = all(box[2] <= o[0] or o[2] <= box[0] or box[3] <= o[1] or o[3] <= box[1] for o in boxes)
            if inside and free:
                chosen = (ha, box)
                break
        ok = ok and chosen is not None
        ha, box = chosen or ("left", (x + 0.03, y + 0.025, x + 0.03 + w, y + 0.025 + h))
        boxes.append(box)
        texts.append(ax.annotate(text, xy=(x, y), xytext=(box[0] if ha == "left" else box[2] if ha == "right" else (box[0] + box[2]) / 2, box[1]),
                                 textcoords="data", fontsize=8, color=color, ha=ha, va="bottom", zorder=6,
                                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                                 arrowprops=dict(arrowstyle="-", color=color, lw=0.6, shrinkA=0, shrinkB=4)))
    return texts, ok


def draw_plate(ax, true_config=None, proposals=(), legend=True):
    """Plate outline with the force position, the true resonators (circles) and proposed resonators (triangles).
    proposals: list of (label, (R, 6) configuration, colour, emphasised). Resonators are labelled with their f_t; when
    the labels would overlap (or several proposal sets are shown) they are numbered instead / left out."""
    ax.set_xlim(-0.05 * Lx, 1.05 * Lx)
    ax.set_ylim(-0.05 * Ly, 1.05 * Ly)
    ax.add_patch(plt.Rectangle((0, 0), Lx, Ly, fill=False, edgecolor="black", lw=1.5))
    ax.scatter([xf], [yf], marker="*", s=180, color="cyan", edgecolors="black", linewidths=0.8, zorder=3, label="Force $F_0$")
    markers, sets = [(xf, yf)], []  # sets: (handle, base label, configuration, colour) of the labelled resonator sets
    if true_config is not None:
        c = np.asarray(true_config, float)
        handle = ax.scatter(c[:, 3], c[:, 4], marker="o", s=90, color="crimson", edgecolors="black", linewidths=0.8, zorder=4, label="True resonator")
        markers += list(zip(c[:, 3], c[:, 4]))
        sets.append((handle, "True resonator", c, "crimson"))
    strong = [pr for pr in proposals if pr[3]]
    for label, cfg, color, emphasised in proposals:
        c = np.asarray(cfg, float)
        handle = ax.scatter(c[:, 3], c[:, 4], marker="^", s=90 if emphasised else 28, color=color, alpha=1.0 if emphasised else 0.35,
                            edgecolors="black" if emphasised else "none", linewidths=0.8, zorder=5 if emphasised else 2, label=label)
        markers += list(zip(c[:, 3], c[:, 4]))
        if emphasised and len(strong) == 1:  # one proposal set: label it; several sets would only clutter the plate
            sets.append((handle, label, c, color))
    labels = [(x, y, f"{ft:.1f} Hz", color) for _, _, c, color in sets for ft, x, y in c[:, [2, 3, 4]]]
    texts, ok = _place_labels(ax, labels, markers)
    if not ok:  # crowded: number the resonators and give the tuning frequencies in the legend
        for t in texts:
            t.remove()
        _place_labels(ax, [(x, y, str(i + 1), color) for _, _, c, color in sets for i, (x, y) in enumerate(c[:, [3, 4]])], markers, w=0.07)
        for handle, label, c, _ in sets:
            handle.set_label(f"{label} (1 to {len(c)}: " + ", ".join(f"{ft:.1f}" for ft in c[:, 2]) + " Hz)")
    ax.set(aspect="equal", xlabel="$x$ (m)", ylabel="$y$ (m)", title="Resonator layout on plate")
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.62), ncol=1 if not ok else 2, fontsize=8)


def plot_spectrum_and_plate(path, true, curves, true_config, title, proposals=(), faint=()):
    """Repo-style figure: spectrum (true in black, predictions dashed, red dotted lines at the resonator tuning
    frequencies) on the left, plate with force and resonator positions on the right.
    curves: list of (label, spectrum, colour); faint: extra thin spectra (other proposals)."""
    fig, (ax, ax_plate) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, y in enumerate(faint):
        ax.plot(freqs, y, color="C0", alpha=0.2, lw=0.9, label="Other proposals" if i == 0 else None)
    ax.plot(freqs, true, lw=2.2, color="black", label="Ground truth (solver)")
    for label, y, color in curves:
        ax.plot(freqs, y, "--", lw=1.7, color=color, label=label)
    for i, ft in enumerate(np.asarray(true_config, float)[:, 2]):
        ax.axvline(ft, color="red", ls=":", lw=1.3, alpha=0.85, zorder=0, label="Resonator $f_t$" if i == 0 else None)
    first = np.asarray(curves[0][1], float) - np.asarray(true, float)
    ax.set(xlabel=FREQ_LABEL, ylabel=ERP_LABEL, title=f"{title}\nMSE $= {np.mean(first**2):.3f}$ dB$^2$, MAE $= {np.mean(np.abs(first)):.3f}$ dB")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    draw_plate(ax_plate, true_config, proposals)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


#%% Inverse validation (invertible models)
@torch.no_grad()
def inverse_designs(model, data, spectrum, num_samples):
    """
    Designs for target spectra (n, F) normalised: samples (n, S, R, 6), point estimate (n, R, 6), own-forward pick (n,).
    """
    R = data.num_res
    spectrum = spectrum.to(device)
    samples = model.sample(spectrum, num_samples)
    n = len(spectrum)
    own = model.predict_erp(samples.reshape(n * num_samples, -1)).reshape(n, num_samples, -1)
    pick = ((own - spectrum[:, None]) ** 2).mean(-1).argmin(1).cpu().numpy()
    return (decode_bounded(samples.cpu().numpy(), R, data.norm), decode_bounded(model.point_estimate(spectrum).cpu().numpy(), R, data.norm), pick)


def evaluate_inverse(model, name, data, loaders, out, num_targets=100, num_samples=8):
    """
    Designs for held-out target spectra, re-simulated with the solver.
    """
    R = data.num_res
    spectrum, design = next(iter(DataLoader(loaders["test"].dataset, batch_size=num_targets)))
    n = len(spectrum)
    true = denormalize_erp(spectrum, data.norm)
    samples, point, pick = inverse_designs(model, data, spectrum, num_samples)
    print(f"{name}: re-simulating {n * num_samples} proposed designs with the solver")
    solved = solve_configurations(samples.reshape(-1, R, CONF_DIM)).reshape(n, num_samples, -1)
    solved_point = solve_configurations(point)
    rows = np.arange(n)
    oracle = ((solved - true[:, None]) ** 2).mean(-1).argmin(1)
    metrics = {"first_sample": spectrum_metrics(solved[:, 0], true), "own_forward_pick": spectrum_metrics(solved[rows, pick], true),
               "point_estimate": spectrum_metrics(solved_point, true), "best_of_samples_oracle": spectrum_metrics(solved[rows, oracle], true)}
    chosen = sort_by_ft(samples[rows, pick])
    truth = sort_by_ft(decode_bounded(design.reshape(n, -1).numpy(), R, data.norm))
    design_columns = [(FIELDS.index(f), f) for f in DESIGN_FIELDS]
    metrics["design_recovery_pearson_r"] = {f: float(np.corrcoef(chosen[..., i].ravel(), truth[..., i].ravel())[0, 1]) for i, f in design_columns}
    metrics["num_targets"], metrics["num_samples"] = int(n), int(num_samples)

    for r in range(min(5, n)):  # target spectrum + solver ERP of the selected proposal, true vs proposed configurations
        others = [k for k in range(num_samples) if k != pick[r]]
        plot_spectrum_and_plate(out / f"inverse_target_{r + 1:02d}.png", true[r], [("Selected proposal (solver)", solved[r, pick[r]], "C2")],
                                truth[r], f"{name}: inverse design for target {r + 1}",
                                proposals=[("Other proposals", samples[r, k], "C0", False) for k in others[:1]] +
                                          [("", samples[r, k], "C0", False) for k in others[1:]] +
                                          [("Selected proposal", samples[r, pick[r]], "C2", True)],
                                faint=[solved[r, k] for k in others])

    fig, axes = plt.subplots(1, 1 + N_DESIGN, figsize=(4.4 * (1 + N_DESIGN), 4))
    for ax, (i, f) in zip(axes[1:], design_columns):
        ax.scatter(truth[..., i].ravel(), chosen[..., i].ravel(), s=8, alpha=0.4)
        lim = [truth[..., i].min(), truth[..., i].max()]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(xlabel=f"True {f}", ylabel=f"Proposed {f}", title=f"r = {metrics['design_recovery_pearson_r'][f]:.3f}")
    axes[0].scatter(true.ravel(), solved[rows, pick].ravel(), s=2, alpha=0.3)
    axes[0].plot([true.min(), true.max()], [true.min(), true.max()], "k--", lw=1)
    axes[0].set(xlabel=f"Target {ERP_LABEL}", ylabel="Solver ERP of proposal")
    fig.tight_layout()
    fig.savefig(out / "inverse_recovery.png")
    plt.close(fig)
    return metrics


def evaluate_model(kind, tag, name, data=None, batch_size=batch_size):
    """
    Test-split validation of a trained model; writes metrics.json and the plots into <kind>/plots/<dataset>/<MODEL>/.
    """
    model, ckpt = load_model(kind, tag, name)
    if model is None:
        print(f"{name}: no trained model for '{tag}' -- train it first.")
        return None
    data = (data or ERPData(tag)).with_norm(checkpoint_norm(ckpt))
    out = plot_dir(kind, tag, name)
    loaders = data.loaders("forward" if kind == "forward" else "inverse", batch_size)
    true, pred = predict_test(kind, model, data, loaders)
    metrics = {"model": name, "dataset": tag, "parameters": parameter_count(model), "forward": spectrum_metrics(pred, true)}
    print(f"{name} forward: RMSE {metrics['forward']['rmse_db']:.3f} dB | R2 {metrics['forward']['r2']:.4f} | "
          f"Pearson {metrics['forward']['pearson_r']:.4f} | RMSE at peaks {metrics['forward']['rmse_at_peaks_db']:.3f} dB")
    plot_loss(ckpt["history"], name, out / "loss_curve.png")
    plot_forward_results(out, name, true, pred, data.freqs)
    test_configs = data.config[data.split["test"]]
    for i in range(min(5, len(true))):  # spectrum + plate with force and resonators, as in the repository
        plot_spectrum_and_plate(out / f"erp_spectrum_test_config_{i + 1:02d}.png", true[i], [(name, pred[i], "#C44E52")],
                                test_configs[i], f"{name}: test configuration {i + 1}")
    if kind == "invertible":
        metrics["inverse"] = evaluate_inverse(model, name, data, loaders, out)
        m = metrics["inverse"]["own_forward_pick"]
        print(f"{name} inverse (solver-checked): RMSE {m['rmse_db']:.3f} dB | Pearson {m['pearson_r']:.4f} | "
              f"design recovery r(f_t) {metrics['inverse']['design_recovery_pearson_r']['f_t']:.3f}")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def compare_models(kind, tag, names, data):
    """
    Table, bar charts, combined loss curves and true vs predicted spectra of all models -> <kind>/plots/<dataset>/ALL_MODELS/.
    """
    rows = []
    for name in names:
        file = plot_dir(kind, tag, name) / "metrics.json"
        if file.exists():
            rows.append(json.loads(file.read_text()))
    if not rows:
        return
    out = plot_dir(kind, tag, group_folder(kind, names))
    header = f"{'Model':8s} {'Params':>9s} {'RMSE':>7s} {'R2':>7s} {'Pearson':>8s} {'PeakRMSE':>9s}" + (f" {'InvRMSE':>8s} {'InvPearson':>10s} {'r(f_t)':>7s}" if kind == "invertible" else "")
    lines = [f"{kind} models on {tag} (test split, dB)", header]
    for r in rows:
        f = r["forward"]
        line = f"{r['model']:8s} {r['parameters']:9,d} {f['rmse_db']:7.3f} {f['r2']:7.4f} {f['pearson_r']:8.4f} {f['rmse_at_peaks_db']:9.3f}"
        if kind == "invertible":
            i = r["inverse"]
            line += f" {i['own_forward_pick']['rmse_db']:8.3f} {i['own_forward_pick']['pearson_r']:10.4f} {i['design_recovery_pearson_r']['f_t']:7.3f}"
        lines.append(line)
    (out / "comparison.txt").write_text("\n".join(lines) + "\n")
    (out / "comparison.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))
    panels = [("rmse_db", "RMSE (dB)"), ("rmse_at_peaks_db", "RMSE at peaks (dB)"), ("pearson_r", "Pearson correlation")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (key, label) in zip(axes, panels):
        ax.bar([r["model"] for r in rows], [r["forward"][key] for r in rows], color="C0")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out / "comparison_forward.png")
    plt.close(fig)
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for i, name in enumerate(names):
        _, ckpt = load_model(kind, tag, name)
        if ckpt is None:
            continue
        h = ckpt["history"]
        for ax, key in zip(axes, ("train", "val")):
            ax.plot(h[key] if key in h else sum((h[st][key] for st in ("stage1", "stage2", "stage3")), []), color=colors[i % 10], label=name)
    for ax, title in zip(axes, ("Training loss", "Validation loss")):
        ax.set(xlabel="Epoch", yscale="log", title=title)
    axes[0].set_ylabel("Loss")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1)
    fig.tight_layout()
    fig.savefig(out / "loss_curves_all_models.png")
    plt.close(fig)

    # the same test configurations for every model: solver ERP vs the prediction of each model
    ids = data.split["test"][:6]
    true = data.erp[ids]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    for ax, t in zip(axes.ravel(), true):
        ax.plot(freqs, t, "k", lw=2, label="Solver (true)")
    for i, name in enumerate(names):
        model, ckpt = load_model(kind, tag, name)
        if model is None:
            continue
        norm = checkpoint_norm(ckpt)
        with torch.no_grad():
            if kind == "forward":
                configuration = torch.from_numpy(normalize_config(data.config[ids], norm)).to(device)
                frequency = torch.from_numpy(((data.freqs - norm["freq_mean"]) / norm["freq_std"]).astype(np.float32))[None, :, None].expand(len(ids), -1, -1).to(device)
                pred = model(configuration, frequency).squeeze(-1).cpu()
            else:
                design = torch.from_numpy(encode_bounded(sort_by_ft(data.config[ids]), norm)).flatten(1).to(device)
                pred = model.predict_erp(design).cpu()
        for ax, p in zip(axes.ravel(), denormalize_erp(pred, norm)):
            ax.plot(freqs, p, color=colors[i % 10], lw=1.0, alpha=0.85, label=name)
    for k, ax in enumerate(axes.ravel()):
        ax.set_title(f"Test configuration {k + 1}")
    axes[1, 0].set_xlabel(FREQ_LABEL)
    axes[0, 0].set_ylabel(ERP_LABEL)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "test_examples_all_models.png")
    plt.close(fig)


#%% Prediction for a configuration
def configuration_from_rows(rows):
    """
    [[m, f_t, x, y, zeta], ...] -> (R, 6) [m, k, f_t, x, y, zeta]; zeta defaults to c = 1 N s/m when a row has four numbers.
    """
    out = []
    for row in rows:
        m, f_t = np.clip(row[0], m_min, m_max), np.clip(row[1], f_min, f_max)
        k = m * (2 * np.pi * f_t) ** 2
        zeta = row[4] if len(row) > 4 else 1.0 / (2 * np.sqrt(k * m))
        out.append([m, k, f_t, np.clip(row[2], 0, Lx), np.clip(row[3], 0, Ly), np.clip(zeta, zeta_min, zeta_max)])
    return np.array(out)


@torch.no_grad()
def predict_configuration(kind, tag, names, data, configuration, direction="forward", num_samples=8):
    """Forward: predicted ERP of the configuration. Inverse (invertible): designs proposed for the configuration's ERP,
    shown next to the true configuration on the plate."""
    true = compute_erp_spectrum(to_resonators(configuration))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = plot_dir(kind, tag, names[0] if len(names) == 1 else group_folder(kind, names))
    report = [f"Configuration [m, k, f_t, x, y, zeta]:\n{np.array2string(configuration, precision=4)}"]
    colors = [f"C{k}" for k in (1, 2, 4, 5, 6, 7, 8, 9)]  # not red (true resonators) and not blue (other proposals)
    curves, proposals, faint = [], [], []
    for i, name in enumerate(names):
        model, ckpt = load_model(kind, tag, name)
        if model is None:
            print(f"{name}: not trained for '{tag}', skipped")
            continue
        data_m = data.with_norm(checkpoint_norm(ckpt))   # statistics the model was trained with
        target = torch.from_numpy(normalize_erp(true, data_m.norm))[None]
        color = colors[i % len(colors)]
        if direction == "forward":
            if kind == "forward":
                cfg = torch.from_numpy(normalize_config(configuration, data_m.norm))[None].to(device)
                freq = torch.from_numpy(data_m.freq_norm)[None, :, None].to(device)
                pred = model(cfg, freq).squeeze(-1).cpu()
            else:
                design = torch.from_numpy(encode_bounded(sort_by_ft(configuration), data_m.norm)).flatten()[None].to(device)
                pred = model.predict_erp(design).cpu()
            pred = denormalize_erp(pred, data_m.norm)[0]
            rmse = np.sqrt(np.mean((pred - true) ** 2))
            curves.append((f"{name} (RMSE {rmse:.2f} dB)", pred, color))
            report.append(f"{name}: RMSE {rmse:.3f} dB")
        else:
            samples, point, pick = inverse_designs(model, data_m, target, num_samples)
            solved = solve_configurations(samples[0])
            best = solved[pick[0]]
            rmse = np.sqrt(np.mean((best - true) ** 2))
            curves.append((f"{name} proposal, solver (RMSE {rmse:.2f} dB)", best, color))
            proposals.append((f"{name} proposal", sort_by_ft(samples[0][pick[0]]), color, True))
            if len(names) == 1:
                faint = [solved[k] for k in range(num_samples) if k != pick[0]]
                proposals = [("Other proposals", samples[0][k], "C0", False) for k in range(num_samples) if k != pick[0]][:1] + \
                            [("", samples[0][k], "C0", False) for k in range(num_samples) if k != pick[0]][1:] + proposals
            report.append(f"{name}: proposed design [m, k, f_t, x, y, zeta]:\n{np.array2string(sort_by_ft(samples[0][pick[0]]), precision=4)}\n"
                          f"   solver RMSE {rmse:.3f} dB")
    if curves:
        title = "Prediction for the given configuration" if direction == "forward" else "Inverse design for the target ERP"
        plot_spectrum_and_plate(out / f"{direction}_prediction_{stamp}.png", true, curves, configuration, title, proposals, faint)
    (out / f"{direction}_prediction_{stamp}.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"Saved {out / f'{direction}_prediction_{stamp}.png'}")


#%% 9. Main
def ask(prompt, default, cast=str):
    while True:
        text = input(f"{prompt} [{default}]: ").strip()
        if not text:
            return default
        try:
            return cast(text)
        except ValueError:
            print("  Invalid input.")


def choose(title, options):
    """
    Numbered menu; returns the index of the chosen option.
    """
    print(f"\n{title}")
    for i, option in enumerate(options, 1):
        print(f"  {i}) {option}")
    while True:
        text = input(f"Choice [1-{len(options)}]: ").strip()
        if text.isdigit() and 1 <= int(text) <= len(options):
            return int(text) - 1
        print("  Invalid choice.")


def choose_models(title, names, labels):
    """
    Numbered menu where one number, several numbers (1,4,6) or the 'All models' entry can be chosen.
    """
    print(f"\n{title}")
    for i, label in enumerate(labels, 1):
        print(f"  {i}) {label}")
    print(f"  {len(names) + 1}) All models")
    while True:
        picks = [t for t in input(f"Choice: one or several numbers separated by commas, e.g. 1,4,6 [1-{len(names) + 1}]: ").replace(" ", "").split(",") if t]
        if picks and all(t.isdigit() and 1 <= int(t) <= len(names) + 1 for t in picks):
            numbers = sorted({int(t) for t in picks})
            return list(names) if len(names) + 1 in numbers else [names[i - 1] for i in numbers]
        print("  Invalid choice.")


def yes(prompt, default=False):
    text = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not text else text.startswith("y")


def prompt_configuration(data):
    """
    Resonator rows 'm, f_t, x, y[, zeta]'; Enter on the first row selects a random test configuration.
    """
    print(f"\nConfiguration of {data.num_res} resonators. One line per resonator: m (kg, {m_min}-{m_max}), f_t (Hz, {f_min:g}-{f_max:g}), "
          f"x (m, 0-{Lx}), y (m, 0-{Ly}), zeta (damping ratio, {zeta_min:g}-{zeta_max:g}, optional: default c = 1 N s/m). "
          "Press Enter for a random test configuration.")
    rows = []
    for i in range(data.num_res):
        while True:
            text = input(f"  Resonator {i + 1}: ").strip()
            if not text and i == 0:
                return data.config[np.random.default_rng().choice(data.split["test"])]
            try:
                values = [float(v) for v in text.replace(",", " ").split()]
                assert len(values) in (4, 5)
                rows.append(values)
                break
            except (ValueError, AssertionError):
                print("  Enter four or five numbers: m, f_t, x, y [, zeta]")
    return configuration_from_rows(rows)


def choose_dataset():
    datasets = list_datasets()
    if not datasets:
        print("No dataset found in dataset/<name>/ -- generate one first.")
        return None
    index = choose("Which dataset?", [f"{d['tag']}  ({d['n']} configurations, {d['num_res']} resonators, {d['nx_modes']}x{d['ny_modes']} plate modes)"
                                       for d in datasets])
    data = ERPData(datasets[index]["tag"])
    if not (dataset_folder(data.tag) / "summary.txt").exists():
        dataset_statistics(data)
    return data


def run_models(kind, data):
    names = list(FORWARD_MODELS if kind == "forward" else INVERTIBLE_MODELS)
    labels = [f"{n}{'  [trained]' if model_path(kind, data.tag, n).exists() else ''}" for n in names]
    selected = choose_models(f"Which {kind} models? ({data.tag})", names, labels)
    action = choose("What do you want to do?", ["Train (then validate on the test split)", "Evaluate trained models (validation, plots)", "Predict for a configuration"])

    if action == 0:
        if kind == "forward":
            epochs = ask("Epochs", epochs_forward, int)
        else:
            idon_epochs = ask("Epochs for iDON", epochs_idon, int) if any(is_idon(n) for n in selected) else 0
            stage_text = ask("Epochs for stage 1/2/3 of the other models", "/".join(map(str, epochs_stages))) if any(not is_idon(n) for n in selected) else "/".join(map(str, epochs_stages))
            stages = tuple(int(v) for v in stage_text.split("/"))
        batch = ask("Batch size", batch_size, int)
        retrain = True
        if any(model_path(kind, data.tag, n).exists() for n in selected):
            retrain = yes("Retrain models that already exist?", False)
        for name in selected:
            if model_path(kind, data.tag, name).exists() and not retrain:
                print(f"{name}: already trained, skipped")
            elif kind == "forward":
                train_forward(name, data, epochs, batch)
            else:
                train_invertible(name, data, idon_epochs if is_idon(name) else stages, batch)
            evaluate_model(kind, data.tag, name, data, batch)
    elif action == 1:
        for name in selected:
            evaluate_model(kind, data.tag, name, data)
    else:
        direction = "forward"
        if kind == "invertible":
            direction = ("forward", "inverse")[choose("Direction?", ["Forward: design -> ERP", "Inverse: target ERP (of a given design) -> design"])]
        predict_configuration(kind, data.tag, selected, data, prompt_configuration(data), direction)
    if len(selected) > 1 and action != 2:
        compare_models(kind, data.tag, selected, data)


def main():
    make_folders()
    print("ERP neural operators: forward and invertible models for a plate with tuned mass resonators")
    if yes("\nRegenerate / create a dataset?", False):
        n = ask("Number of configurations", 100000, int)
        num_res = ask("Number of resonators", 2, int)
        modes = (6, 3) if choose("Plate modes?", ["6 x 3 = 18 modes", "15 x 10 = 150 modes"]) == 0 else (15, 10)
        tag = generate_dataset(n, num_res, *modes)
        dataset_statistics(ERPData(tag))
    while True:
        workflow = choose("Which models?", ["Forward models (design -> ERP)", "Invertible models (iDON, iFNO family)", "Quit"])
        if workflow == 2:
            break
        data = choose_dataset()
        if data is not None:
            run_models("forward" if workflow == 0 else "invertible", data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
