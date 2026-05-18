
# ============================================================
#  IMPORTS
# ============================================================
import os, sys, time, json, argparse, warnings, base64, re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
#  GLOBAL CONFIG
# ============================================================
CONFIG = {
    "vae_epochs": 50, "teacher_epochs": 50, "student_epochs": 50,
    "batch_size": 64, "lr": 1e-3, "latent_dim": 32,
    "vae_hidden": [128, 64],
    "teacher_hidden": [256, 128, 64],
    "student_hidden": [64, 32],
    "kd_temperature": 2.0, "kd_alpha": 0.5, "beta": 1.0,
    "train_ratio": 0.70,
    "n_background": 200, "top_k_features": 15,
    "early_stop_patience": 10,
}

DATASET_CONFIGS = {
    "cicids2017": {**CONFIG,
        "vae_epochs": 100, "teacher_epochs": 100, "student_epochs": 100,
        "latent_dim": 64, "vae_hidden": [256, 128],
        "teacher_hidden": [512, 256, 128], "student_hidden": [128, 64],
        "n_background": 300, "train_ratio": 0.70, "early_stop_patience": 10},
    "nsl-kdd": {**CONFIG,
        "vae_epochs": 100, "teacher_epochs": 100, "student_epochs": 100,
        "latent_dim": 32, "train_ratio": 0.70},
    "synthetic": {**CONFIG,
        "latent_dim": 16, "vae_hidden": [64, 32],
        "teacher_hidden": [128, 64, 32], "student_hidden": [32, 16],
        "n_background": 100, "train_ratio": 0.70},
}

sep = "=" * 60

# Hardcoded Groq credentials/config (no .env required).
HARDCODED_GROQ_API_KEY = "PUT_UR_KEY"
HARDCODED_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def load_local_env(env_file=".env"):
    """
    Lightweight .env loader with no extra dependency.
    Existing environment variables are not overwritten.
    """
    path = Path(env_file)
    if not path.exists():
        return

    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1

    if loaded:
        print(f"  [ENV] Loaded {loaded} variable(s) from {env_file}")


# ============================================================
#  PREPROCESSING
# ============================================================
def _drop_constant_cols(df):
    return df.loc[:, df.nunique() > 1]


def _encode_and_scale(df, target_col, drop_cols=None):
    """
    Encode categoricals, drop constants, standard-scale features.
    Returns: X (float32 ndarray), y (int64 ndarray), feature_names,
             class_names, encoders dict, scaler, label_encoder
    NOTE: This is the ONE place we fit the scaler. Do NOT scale again downstream.
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    if target_col not in df.columns:
        def _norm_col(name):
            return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())

        target_lookup = {_norm_col(c): c for c in df.columns}
        resolved_target = target_lookup.get(_norm_col(target_col))
        if resolved_target is None:
            raise KeyError(
                f"Target column '{target_col}' not found. Available columns: {list(df.columns[:10])}"
            )
        target_col = resolved_target

    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df = df[df[target_col].notna()].reset_index(drop=True)

    y_raw = df[target_col].astype(str).str.strip()
    X_df  = _drop_constant_cols(df.drop(columns=[target_col]))

    encoders = {}
    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
        encoders[col] = le

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values).astype(np.float32)

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y_raw).astype(np.int64)

    return X, y, list(X_df.columns), list(label_enc.classes_), encoders, scaler, label_enc


# ============================================================
#  SMOTE (SAFE WRAPPER)
# ============================================================
def apply_smote_safe(X_train, y_train):
    """
    Apply SMOTE after split + scaling.
    Clamps k_neighbors to min(5, min_class_size - 1).
    Skips silently if any class still has only 1 sample.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("  [SMOTE] imbalanced-learn not installed. Run: pip install imbalanced-learn")
        return X_train, y_train

    unique, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()

    if min_count < 2:
        print(f"  [SMOTE] Skipped — class(es) with only 1 sample found. "
              f"Using class weights instead.")
        return X_train, y_train

    k = min(5, int(min_count) - 1)
    print(f"  [SMOTE] Applying SMOTE (k_neighbors={k}) …")
    before = dict(zip(*np.unique(y_train, return_counts=True)))

    smote = SMOTE(random_state=42, k_neighbors=k)
    try:
        X_res, y_res = smote.fit_resample(X_train, y_train)
        after = dict(zip(*np.unique(y_res, return_counts=True)))
        print(f"  [SMOTE] Before: {before}")
        print(f"  [SMOTE] After : {after}")
        return X_res.astype(np.float32), y_res.astype(np.int64)
    except Exception as e:
        print(f"  [SMOTE] Failed ({e}). Proceeding without SMOTE.")
        return X_train, y_train


# ============================================================
#  VAE
# ============================================================
class _MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        dims = [input_dim] + hidden_dims
        self.net = nn.Sequential(
            *[_MLPBlock(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.fc_mu     = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        rdims = [latent_dim] + list(reversed(hidden_dims))
        layers = [_MLPBlock(rdims[i], rdims[i+1]) for i in range(len(rdims)-1)]
        layers.append(nn.Linear(rdims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), latent_dim=32, beta=1.0):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dims, input_dim)
        self.beta = beta
        self.latent_dim = latent_dim

    def reparameterise(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar, z

    def loss(self, x, x_recon, mu, logvar):
        recon = F.mse_loss(x_recon, x, reduction="mean")
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta * kl, recon, kl

    @torch.no_grad()
    def encode(self, x_tensor):
        self.eval()
        mu, _ = self.encoder(x_tensor)
        return mu


# ============================================================
#  TEACHER / STUDENT
# ============================================================
class TeacherModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128, 64)):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = [_MLPBlock(dims[i], dims[i+1], dropout=0.3)
                  for i in range(len(dims)-1)]
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class StudentModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(64, 32)):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = [_MLPBlock(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def distillation_loss(s_logits, t_logits, labels, T=2.0, alpha=0.5,
                      class_weights=None):
    hard = F.cross_entropy(s_logits, labels, weight=class_weights)
    soft = F.kl_div(
        F.log_softmax(s_logits / T, dim=1),
        F.softmax(t_logits / T, dim=1).detach(),
        reduction="batchmean"
    ) * (T ** 2)
    return (1 - alpha) * hard + alpha * soft


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
#  TRAINER
# ============================================================
class LENSXAITrainer:
    def __init__(self, input_dim, num_classes, cfg=None, device=None,
                 class_weights=None):
        cfg = cfg or CONFIG
        self.cfg = cfg
        self.device = device or DEVICE
        self.num_classes = num_classes

        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                self.class_weights = class_weights.to(self.device)
            else:
                self.class_weights = torch.FloatTensor(class_weights).to(self.device)
        else:
            self.class_weights = None

        self.vae     = VAE(input_dim, cfg["vae_hidden"],
                           cfg["latent_dim"], cfg["beta"]).to(self.device)
        self.teacher = TeacherModel(cfg["latent_dim"], num_classes,
                                    cfg["teacher_hidden"]).to(self.device)
        self.student = StudentModel(cfg["latent_dim"], num_classes,
                                    cfg["student_hidden"]).to(self.device)

        self.opt_vae     = torch.optim.Adam(self.vae.parameters(),     lr=cfg["lr"])
        self.opt_teacher = torch.optim.Adam(self.teacher.parameters(), lr=cfg["lr"])
        self.opt_student = torch.optim.Adam(self.student.parameters(), lr=cfg["lr"])

        self.history = {"vae": [], "teacher": [], "student": []}

    # ── data helpers ──
    def _loader(self, X, y=None, shuffle=True):
        Xt = torch.FloatTensor(X).to(self.device)
        ds = (TensorDataset(Xt, torch.LongTensor(y).to(self.device))
              if y is not None else TensorDataset(Xt))
        return DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=shuffle)

    def _to_latent(self, X):
        self.vae.eval()
        chunks = []
        with torch.no_grad():
            for (b,) in self._loader(X, shuffle=False):
                chunks.append(self.vae.encode(b).cpu())
        return torch.cat(chunks).numpy()

    def _log(self, label, ep, total, loss):
        if ep % max(1, total // 10) == 0:
            print(f"  [{label:<8}] epoch {ep:>4}/{total}  loss={loss:.4f}")

    # ── VAE ──
    def train_vae(self, X_train, epochs=None, verbose=True):
        epochs = epochs or self.cfg["vae_epochs"]
        loader = self._loader(X_train, shuffle=True)
        self.vae.train()
        epoch_iter = tqdm(range(1, epochs + 1), desc="VAE", leave=True)
        for ep in epoch_iter:
            tot = 0.0
            batch_iter = tqdm(loader, desc=f"VAE epoch {ep}/{epochs}", leave=False)
            for (b,) in batch_iter:
                self.opt_vae.zero_grad()
                xr, mu, lv, _ = self.vae(b)
                loss, _, _ = self.vae.loss(b, xr, mu, lv)
                loss.backward()
                self.opt_vae.step()
                tot += loss.item()
            avg = tot / len(loader)
            self.history["vae"].append(avg)
            epoch_iter.set_postfix(loss=f"{avg:.4f}")
            if verbose:
                self._log("VAE", ep, epochs, avg)

    # ── Teacher ──
    def train_teacher(self, Z, y, Z_val=None, y_val=None,
                      epochs=None, verbose=True):
        epochs   = epochs or self.cfg["teacher_epochs"]
        patience = self.cfg.get("early_stop_patience", 10)
        loader   = self._loader(Z, y, shuffle=True)
        best_loss, no_improve, best_state = float("inf"), 0, None

        epoch_iter = tqdm(range(1, epochs + 1), desc="Teacher", leave=True)
        for ep in epoch_iter:
            self.teacher.train()
            tot = 0.0
            batch_iter = tqdm(loader, desc=f"Teacher epoch {ep}/{epochs}", leave=False)
            for zb, yb in batch_iter:
                self.opt_teacher.zero_grad()
                loss = F.cross_entropy(self.teacher(zb), yb,
                                       weight=self.class_weights)
                loss.backward()
                self.opt_teacher.step()
                tot += loss.item()
            avg = tot / len(loader)
            self.history["teacher"].append(avg)
            epoch_iter.set_postfix(loss=f"{avg:.4f}")
            if verbose:
                self._log("Teacher", ep, epochs, avg)

            check = (self._eval_loss(Z_val, y_val, "teacher")
                     if Z_val is not None else avg)
            if check < best_loss - 1e-4:
                best_loss, no_improve = check, 0
                best_state = {k: v.clone()
                              for k, v in self.teacher.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [Teacher] Early stop at epoch {ep}")
                    break
        if best_state:
            self.teacher.load_state_dict(best_state)

    # ── Student ──
    def train_student(self, Z, y, Z_val=None, y_val=None,
                      epochs=None, verbose=True):
        epochs   = epochs or self.cfg["student_epochs"]
        patience = self.cfg.get("early_stop_patience", 10)
        loader   = self._loader(Z, y, shuffle=True)
        T, alpha = self.cfg["kd_temperature"], self.cfg["kd_alpha"]
        best_loss, no_improve, best_state = float("inf"), 0, None

        self.teacher.eval()
        epoch_iter = tqdm(range(1, epochs + 1), desc="Student", leave=True)
        for ep in epoch_iter:
            self.student.train()
            tot = 0.0
            batch_iter = tqdm(loader, desc=f"Student epoch {ep}/{epochs}", leave=False)
            for zb, yb in batch_iter:
                self.opt_student.zero_grad()
                with torch.no_grad():
                    t_log = self.teacher(zb)
                s_log = self.student(zb)
                loss  = distillation_loss(s_log, t_log, yb, T, alpha,
                                          self.class_weights)
                loss.backward()
                self.opt_student.step()
                tot += loss.item()
            avg = tot / len(loader)
            self.history["student"].append(avg)
            epoch_iter.set_postfix(loss=f"{avg:.4f}")
            if verbose:
                self._log("Student", ep, epochs, avg)

            check = (self._eval_loss(Z_val, y_val, "student")
                     if Z_val is not None else avg)
            if check < best_loss - 1e-4:
                best_loss, no_improve = check, 0
                best_state = {k: v.clone()
                              for k, v in self.student.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [Student] Early stop at epoch {ep}")
                    break
        if best_state:
            self.student.load_state_dict(best_state)

    @torch.no_grad()
    def _eval_loss(self, Z, y, model_name):
        clf = self.teacher if model_name == "teacher" else self.student
        clf.eval()
        Zt = torch.FloatTensor(Z).to(self.device)
        yt = torch.LongTensor(y).to(self.device)
        return F.cross_entropy(clf(Zt), yt,
                               weight=self.class_weights).item()

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        print(f"\n{sep}\nSTEP 1 — Training VAE\n{sep}")
        self.train_vae(X_train, verbose=verbose)

        print(f"\n{sep}\nSTEP 2 — Latent representations\n{sep}")
        Z = self._to_latent(X_train)
        print(f"  Latent shape: {Z.shape}")
        Z_val_latent = self._to_latent(X_val) if X_val is not None else None

        print(f"\n{sep}\nSTEP 3 — Training Teacher\n{sep}")
        self.train_teacher(Z, y_train, Z_val=Z_val_latent,
                           y_val=y_val, verbose=verbose)

        print(f"\n{sep}\nSTEP 4 — Training Student (KD)\n{sep}")
        self.train_student(Z, y_train, Z_val=Z_val_latent,
                           y_val=y_val, verbose=verbose)
        return Z

    @torch.no_grad()
    def predict(self, X, model="teacher"):
        Z  = self._to_latent(X)
        Zt = torch.FloatTensor(Z).to(self.device)
        clf = self.teacher if model == "teacher" else self.student
        clf.eval()
        return clf(Zt).argmax(dim=1).cpu().numpy(), Z

    def evaluate(self, X_test, y_test, class_names=None, verbose=True):
        results = {}
        for name in ("teacher", "student"):
            t0 = time.perf_counter()
            preds, _ = self.predict(X_test, model=name)
            ms = ((time.perf_counter() - t0) * 1000
                  / max(1, len(X_test) // self.cfg["batch_size"]))
            results[name] = dict(
                accuracy  = accuracy_score(y_test, preds),
                precision = precision_score(y_test, preds, average="weighted",
                                            zero_division=0),
                recall    = recall_score(y_test, preds, average="weighted",
                                         zero_division=0),
                f1        = f1_score(y_test, preds, average="weighted",
                                     zero_division=0),
                preds     = preds,
                params    = count_params(self.teacher if name == "teacher"
                                         else self.student),
                ms_per_batch = ms,
            )
            if verbose:
                r = results[name]
                print(f"\n  ▶ {name.upper()} MODEL")
                print(f"    Accuracy   : {r['accuracy']*100:.2f}%")
                print(f"    Precision  : {r['precision']*100:.2f}%")
                print(f"    Recall     : {r['recall']*100:.2f}%")
                print(f"    F1-Score   : {r['f1']*100:.2f}%")
                print(f"    Parameters : {r['params']:,}")
                print(f"    Infer time : {r['ms_per_batch']:.2f} ms/batch")
                print(f"\n  Per-class report ({name.upper()}):")
                print(classification_report(
                    y_test, preds,
                    target_names=class_names if class_names else None,
                    zero_division=0, digits=4))
        return results

    def save(self, path):
        torch.save({"vae": self.vae.state_dict(),
                    "teacher": self.teacher.state_dict(),
                    "student": self.student.state_dict()}, path)
        print(f"  Saved → {path}")


# ============================================================
#  XAI — VARIABLE ATTRIBUTION EXPLAINER
# ============================================================
class VariableAttributionExplainer:
    def __init__(self, trainer, feature_names=None):
        self.trainer = trainer
        self.feature_names = feature_names

    def _proba(self, X, model, target_class):
        Xt  = torch.FloatTensor(X).to(self.trainer.device)
        clf = (self.trainer.teacher if model == "teacher"
               else self.trainer.student)
        clf.eval()
        self.trainer.vae.eval()
        with torch.no_grad():
            probs = torch.softmax(
                clf(self.trainer.vae.encode(Xt)), dim=1)
        return probs[:, target_class].mean().item()

    def explain(self, x_test, X_background, target_class=1,
                model="teacher", n_bg=None):
        n_bg = n_bg or self.trainer.cfg["n_background"]
        bg   = X_background[:n_bg]
        p    = x_test.shape[1]
        v0   = self._proba(bg, model, target_class)

        # Marginal importance pass (ordering)
        marg = np.zeros(p)
        for j in range(p):
            bg_mod = bg.copy()
            bg_mod[:, j] = x_test[0, j]
            marg[j] = abs(self._proba(bg_mod, model, target_class) - v0)

        order = np.argsort(marg)[::-1]

        # Shapley-style additive pass
        contributions = np.zeros(p)
        x_cur = bg.copy()
        prev  = v0
        for j in order:
            x_cur[:, j] = x_test[0, j]
            curr = self._proba(x_cur, model, target_class)
            contributions[j] = curr - prev
            prev = curr

        return {
            "baseline": v0,
            "contributions": contributions,
            "order": order,
            "prediction": prev,
            "local_accuracy": abs((v0 + contributions.sum()) - prev) < 1e-6,
        }

    def build_explanation_context(self, result, feature_names, class_names,
                                  target_class, model_name, dataset_name,
                                  eval_results, top_k=15):
        order  = result["order"][:top_k]
        values = result["contributions"][order]
        names  = ([feature_names[i] for i in order]
                  if feature_names else [f"feat_{i}" for i in order])
        positive = [(n, float(v)) for n, v in zip(names, values) if v > 0]
        negative = [(n, float(v)) for n, v in zip(names, values) if v < 0]
        target_name = (class_names[target_class]
                       if target_class < len(class_names)
                       else str(target_class))
        r = eval_results.get(model_name, {})
        return {
            "dataset": dataset_name,
            "model": model_name,
            "target_class": target_name,
            "baseline": round(float(result["baseline"]), 4),
            "prediction": round(float(result["prediction"]), 4),
            "local_accuracy_ok": bool(result["local_accuracy"]),
            "top_positive_features": positive[:5],
            "top_negative_features": negative[:5],
            "model_accuracy": round(float(r.get("accuracy", 0)) * 100, 2),
            "model_f1": round(float(r.get("f1", 0)) * 100, 2),
            "model_params": r.get("params", 0),
        }


# ============================================================
#  FREE LLM EXPLAINER
# ============================================================
class LLMExplainer:
    """
    XAI narrative explanation using a free LLM.
    Supports: gemini | groq | ollama | anthropic (paid)
    Falls back to rule-based text if no provider/key is available.
    """

    SYSTEM_PROMPT = (
        "You are a cybersecurity AI assistant explaining network intrusion "
        "detection model decisions to security analysts. Be concise and use "
        "security terminology. Explain what features mean in a network security context."
    )

    def __init__(self, provider="gemini", api_key=None,
                 ollama_model="llama3.2", groq_model="llama-3.1-8b-instant"):
        self.provider     = provider.lower()
        self.ollama_model = ollama_model
        self.groq_model   = groq_model
        self.available    = False
        self.supports_images = False
        self._client      = None

        if self.provider == "gemini":
            self._init_gemini(api_key)
        elif self.provider == "groq":
            self._init_groq(api_key)
        elif self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "anthropic":
            self._init_anthropic(api_key)
        elif self.provider == "none":
            print("  [LLM] Provider set to 'none' — rule-based fallback only.")
        else:
            print(f"  [LLM] Unknown provider '{provider}'. "
                  "Choose: gemini | groq | ollama | anthropic | none")

    # ── provider initialisers ──
    def _init_gemini(self, api_key):
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            print("  [LLM] Gemini: set GEMINI_API_KEY or pass api_key=. "
                  "Falling back to rule-based.")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._client   = genai.GenerativeModel("gemini-1.5-flash")
            self.available = True
            print("  [LLM] Gemini Flash initialised ✓")
        except ImportError:
            print("  [LLM] Run: pip install google-generativeai")
        except Exception as e:
            print(f"  [LLM] Gemini init failed: {e}")

    def _init_groq(self, api_key):
        key = api_key or HARDCODED_GROQ_API_KEY
        if not key:
            print("  [LLM] Groq: set HARDCODED_GROQ_API_KEY or pass api_key=. "
                  "Falling back to rule-based.")
            return
        try:
            from groq import Groq
            self._client   = Groq(api_key=key)
            self.available = True
            model_name = self.groq_model.lower()
            self.supports_images = any(
                token in model_name for token in ("vision", "scout", "maverick", "multimodal")
            )
            print(f"  [LLM] Groq ({self.groq_model}) initialised ✓")
            if self.supports_images:
                print("  [LLM] Groq vision mode enabled for image inputs ✓")
        except ImportError:
            print("  [LLM] Run: pip install groq")
        except Exception as e:
            print(f"  [LLM] Groq init failed: {e}")

    def _init_ollama(self):
        try:
            import ollama
            # Quick ping to check model is available
            ollama.show(self.ollama_model)
            self._client   = ollama
            self.available = True
            print(f"  [LLM] Ollama ({self.ollama_model}) initialised ✓")
        except ImportError:
            print("  [LLM] Run: pip install ollama  (and install ollama app)")
        except Exception as e:
            print(f"  [LLM] Ollama init failed: {e}. "
                  f"Make sure ollama is running and model pulled: "
                  f"ollama pull {self.ollama_model}")

    def _init_anthropic(self, api_key):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            print("  [LLM] Anthropic: set ANTHROPIC_API_KEY. "
                  "Falling back to rule-based.")
            return
        try:
            import anthropic
            self._client   = anthropic.Anthropic(api_key=key)
            self.available = True
            print("  [LLM] Anthropic Claude initialised ✓")
        except ImportError:
            print("  [LLM] Run: pip install anthropic")
        except Exception as e:
            print(f"  [LLM] Anthropic init failed: {e}")

    # ── prompt builder ──
    def _build_prompt(self, ctx):
        pos = ", ".join(f"{n} (+{v:.4f})"
                        for n, v in ctx["top_positive_features"]) or "none"
        neg = ", ".join(f"{n} ({v:.4f})"
                        for n, v in ctx["top_negative_features"]) or "none"
        return (
            f"Dataset: {ctx['dataset']}\n"
            f"Model: {ctx['model']} "
            f"(Acc={ctx['model_accuracy']}%, F1={ctx['model_f1']}%, "
            f"Params={ctx['model_params']:,})\n"
            f"Predicted as: {ctx['target_class']}\n"
            f"Baseline prob: {ctx['baseline']}\n"
            f"Predicted prob: {ctx['prediction']}\n"
            f"Attribution valid: {ctx['local_accuracy_ok']}\n\n"
            f"Features INCREASING probability of {ctx['target_class']}: {pos}\n"
            f"Features DECREASING probability: {neg}\n\n"
            "Write 3-4 sentences for a security analyst: "
            "(1) what this prediction means, "
            "(2) which features drove it and why they matter in network security, "
            "(3) whether to trust the prediction."
        )

    def _image_to_data_url(self, image_path):
        if not image_path:
            return None

        path = Path(image_path)
        if not path.exists():
            return None

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    # ── call router ──
    def _call_llm(self, prompt, max_tokens=400, image_path=None):
        if self.provider == "gemini":
            resp = self._client.generate_content(
                self.SYSTEM_PROMPT + "\n\n" + prompt,
                generation_config={"max_output_tokens": max_tokens},
            )
            return resp.text.strip()

        elif self.provider == "groq":
            user_content = [{"type": "text", "text": prompt}]
            if image_path:
                if self.supports_images:
                    image_url = self._image_to_data_url(image_path)
                    if image_url:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        })
                else:
                    print("  [LLM] Groq model is not vision-capable; sending text only.")

            resp = self._client.chat.completions.create(
                model=self.groq_model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
            )
            return resp.choices[0].message.content.strip()

        elif self.provider == "ollama":
            resp = self._client.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp["message"]["content"].strip()

        elif self.provider == "anthropic":
            resp = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()

        raise RuntimeError(f"Unknown provider: {self.provider}")

    # ── fallback ──
    def _fallback(self, ctx):
        pos   = ", ".join(n for n, _ in ctx["top_positive_features"][:3]) or "none"
        neg   = ", ".join(n for n, _ in ctx["top_negative_features"][:3]) or "none"
        trust = "high" if ctx["model_accuracy"] >= 85 else "moderate"
        direction = ("increased" if ctx["prediction"] > ctx["baseline"]
                     else "decreased")
        return (
            f"The {ctx['model']} model classified this traffic as "
            f"'{ctx['target_class']}' with probability {ctx['prediction']:.4f} "
            f"(baseline: {ctx['baseline']:.4f}). "
            f"Prediction probability {direction} from baseline. "
            f"Key features driving classification: {pos}. "
            f"Features against: {neg}. "
            f"Model confidence is {trust} "
            f"(Acc: {ctx['model_accuracy']}%, F1: {ctx['model_f1']}%)."
        )

    def _cmp_fallback(self, t, s):
        return (
            f"Both teacher ({t['model_accuracy']}% acc) and "
            f"student ({s['model_accuracy']}% acc) agreed on "
            f"'{t['target_class']}'. Student ({s['model_params']:,} params vs "
            f"teacher {t['model_params']:,}) achieved comparable attributions "
            "via knowledge distillation."
        )

    # ── public API ──
    def explain(self, ctx, image_path=None):
        if not self.available:
            return self._fallback(ctx)
        try:
            return self._call_llm(self._build_prompt(ctx), image_path=image_path)
        except Exception as e:
            print(f"  [LLM] Call failed ({e}) — fallback.")
            return self._fallback(ctx)

    def explain_both_models(self, teacher_ctx, student_ctx, image_path=None):
        t_exp = self.explain(teacher_ctx, image_path=image_path)
        s_exp = self.explain(student_ctx, image_path=image_path)

        cmp_exp = None
        if self.available:
            cmp_prompt = (
                f"Teacher:\n{t_exp}\n\n"
                f"Student ({student_ctx['model_params']:,} params):\n{s_exp}\n\n"
                "In 2-3 sentences compare how teacher and student arrived at "
                "their decisions. Note feature differences and whether the "
                "student reasons like the teacher."
            )
            try:
                cmp_exp = self._call_llm(cmp_prompt, max_tokens=250,
                                         image_path=image_path)
            except Exception as e:
                print(f"  [LLM] Compare failed: {e}")
                cmp_exp = self._cmp_fallback(teacher_ctx, student_ctx)
        else:
            cmp_exp = self._cmp_fallback(teacher_ctx, student_ctx)

        return {
            "teacher_explanation":  t_exp,
            "student_explanation":  s_exp,
            "comparative_analysis": cmp_exp,
        }


# ============================================================
#  VISUALISATION
# ============================================================
def plot_confusion_matrix(y_true, y_pred, classes=None,
                          title="Confusion Matrix", ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    # seaborn >= 0.12 does not accept None for xticklabels/yticklabels;
    # use "auto" (shows integer indices) when no class names are provided.
    tick_labels = classes if classes is not None else "auto"
    annot = cm.size <= 400   # skip per-cell annotations for very large matrices
    sns.heatmap(cm, annot=annot, fmt="d", cmap="Blues", ax=ax,
                xticklabels=tick_labels, yticklabels=tick_labels,
                annot_kws={"size": 7})
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual",    fontsize=9)
    ax.set_title(title,        fontsize=10)
    ax.tick_params(axis="both", labelsize=6)
    return ax


def plot_training_history(history, title="Training History"):
    keys  = {k: v for k, v in history.items() if v}
    ncols = len(keys)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    colors = {"vae": "#3498db", "teacher": "#e67e22", "student": "#2ecc71"}
    for ax, (k, vals) in zip(axes, keys.items()):
        ax.plot(vals, color=colors.get(k, "grey"), lw=1.5)
        ax.set_title(f"{k.title()} Loss", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_metrics_comparison(results, title="Model Comparison"):
    metrics = ["accuracy", "precision", "recall", "f1"]
    x     = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = {"teacher": "#e67e22", "student": "#2ecc71"}
    for i, (name, res) in enumerate(results.items()):
        vals = [res[m] * 100 for m in metrics]
        ax.bar(x + i * width, vals, width, label=name.capitalize(),
               color=colors.get(name, "grey"), edgecolor="white")
        for xi, v in zip(x + i * width, vals):
            ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_class_distribution(y, class_names, title="Class Distribution",
                             save_path=None):
    counts = np.bincount(y, minlength=len(class_names))
    order  = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, len(class_names) * 0.6)))
    colors = ["#2980b9" if c == max(counts) else "#27ae60" for c in counts[order]]
    bars   = ax.barh([class_names[i] for i in order], counts[order],
                     color=colors, edgecolor="white")
    for bar, cnt in zip(bars, counts[order]):
        ax.text(bar.get_width() + max(counts) * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Count")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(counts) * 1.15)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_xai_with_explanation(attr_t, attr_s, llm_exp, feature_names,
                               dataset_name, top_k, save_path=None):
    fig = plt.figure(figsize=(24, 14))
    gs  = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.5, wspace=0.35)
    for col, (mname, attr) in enumerate([("teacher", attr_t),
                                          ("student",  attr_s)]):
        ax_bar = fig.add_subplot(gs[0, col])
        order  = attr["order"][:top_k]
        values = attr["contributions"][order]
        names  = ([feature_names[i] for i in order]
                  if feature_names else [f"feat_{i}" for i in order])
        colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in values]
        ax_bar.barh(range(len(order)), values,
                    left=[attr["baseline"]] * len(order),
                    color=colors, edgecolor="white", linewidth=0.5)
        ax_bar.axvline(attr["baseline"],   color="grey",  lw=1,   ls="--")
        ax_bar.axvline(attr["prediction"], color="black", lw=1.5, ls="-")
        ax_bar.set_yticks(range(len(order)))
        ax_bar.set_yticklabels(names, fontsize=8)
        ax_bar.set_xlabel("Contribution", fontsize=9)
        ax_bar.set_title(
            f"{dataset_name} — {mname.title()} Attribution\n"
            f"Baseline={attr['baseline']:.4f} → Prediction={attr['prediction']:.4f}",
            fontsize=10)
        ax_bar.legend(handles=[
            mpatches.Patch(color="#27ae60", label="Positive"),
            mpatches.Patch(color="#e74c3c", label="Negative")],
            fontsize=7, loc="lower right")
        ax_bar.grid(axis="x", alpha=0.3)

        ax_txt = fig.add_subplot(gs[1, col])
        ax_txt.axis("off")
        text = llm_exp.get(f"{mname}_explanation", "No explanation.")
        ax_txt.text(0, 1.0, f"LLM Explanation — {mname.title()} Model:",
                    transform=ax_txt.transAxes, fontsize=9,
                    fontweight="bold", va="top")
        ax_txt.text(0, 0.85, text,
                    transform=ax_txt.transAxes, fontsize=8, va="top",
                    wrap=True,
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="#f0f4f8", edgecolor="#aac"))

    fig.suptitle(f"{dataset_name} — XAI Attribution with LLM Explanations",
                 fontsize=13, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ============================================================
#  CICIDS2017 LOADER
# ============================================================
CICIDS_LABEL_MAP = {
    "BENIGN": "Benign", "Benign": "Benign",
    "FTP-Patator": "FTP-Patator", "SSH-Patator": "SSH-Patator",
    "DoS slowloris": "DoS-Slowloris",   "DoS Slowloris": "DoS-Slowloris",
    "DoS Slowhttptest": "DoS-Slowhttptest",
    "DoS slowhttptest": "DoS-Slowhttptest",
    "DoS Hulk": "DoS-Hulk", "DoS GoldenEye": "DoS-GoldenEye",
    "Heartbleed": "Heartbleed",
    "Web Attack \u2013 Brute Force": "Web-BruteForce",
    "Web Attack \x96 Brute Force":  "Web-BruteForce",
    "Web Attack \u2013 XSS": "Web-XSS",
    "Web Attack \x96 XSS":  "Web-XSS",
    "Web Attack \u2013 Sql Injection": "Web-SQLi",
    "Web Attack \x96 Sql Injection":  "Web-SQLi",
    "Infiltration": "Infiltration", "Bot": "Bot",
    "PortScan": "PortScan", "DDoS": "DDoS",
}

CICIDS_DROP_COLS = [
    "Flow ID", "Source IP", "Destination IP",
    "Source Port", "Destination Port", "Protocol", "Timestamp",
]


def load_cicids2017(data_dir="data/cicids2017", sample_n=None,
                    multiclass=True):
    """
    Load CICIDS2017. Finds all CSV files in data_dir automatically.
    Download: kaggle.com/datasets/chethuhn/network-intrusion-dataset
    """
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files in: {data_dir}\n"
            "Download from kaggle.com/datasets/chethuhn/network-intrusion-dataset")

    print(f"[CICIDS2017] Found {len(csv_files)} CSV file(s):")
    frames = []
    for fpath in csv_files:
        for enc in ("utf-8", "latin-1"):
            try:
                df_part = pd.read_csv(fpath, low_memory=False,
                                      encoding=enc, on_bad_lines="skip")
                frames.append(df_part)
                print(f"  {fpath.name}: {len(df_part):,} rows")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"  Warning {fpath.name}: {e}")
                break

    if not frames:
        raise ValueError("All CSV files failed to load.")

    df = pd.concat(frames, ignore_index=True)
    print(f"[CICIDS2017] Total: {len(df):,} rows")
    df.columns = df.columns.str.strip()

    def _norm_col(name):
        return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())

    label_col = next(
        (c for c in df.columns if _norm_col(c) == "label"), None)
    if label_col is None:
        raise ValueError(f"Label not found. Cols: {list(df.columns[:10])}")

    if label_col != "Label":
        df = df.rename(columns={label_col: "Label"})
    label_col = "Label"

    df[label_col] = (df[label_col].astype(str).str.strip()
                     .map(lambda v: CICIDS_LABEL_MAP.get(v, v)))
    df = df[df[label_col].notna()].reset_index(drop=True)

    if not multiclass:
        df[label_col] = df[label_col].apply(
            lambda v: "Benign" if v == "Benign" else "Attack")

    df = df.drop(columns=[c for c in CICIDS_DROP_COLS if c in df.columns],
                 errors="ignore")
    df = (df.replace([np.inf, -np.inf], np.nan)
            .dropna(axis=1, thresh=int(0.5 * len(df))))

    if sample_n and len(df) > sample_n:
        total_rows = len(df)
        sampled_parts = []
        for _, group in df.groupby(label_col, sort=False):
            n_take = min(len(group), max(1, int(sample_n * len(group) / total_rows)))
            sampled_parts.append(group.sample(n=n_take, random_state=42))
        df = pd.concat(sampled_parts, ignore_index=True)
        print(f"[CICIDS2017] Sampled to {len(df):,} rows")

    X, y, feat, cls, *_ = _encode_and_scale(df, label_col)
    print(f"[CICIDS2017]  Samples={len(X):,}  Features={X.shape[1]}"
          f"  Classes={cls}")

    counts = np.bincount(y, minlength=len(cls))
    print("[CICIDS2017] Class distribution:")
    for c, n in zip(cls, counts):
        print(f"  {c:<30} {n:>8,}  ({100*n/len(y):.2f}%)")

    return X, y, feat, cls


def load_synthetic_demo(dataset="cicids2017", n=5000, seed=42):
    rng = np.random.default_rng(seed)
    templates = {
        "cicids2017": {
            "classes": ["Benign","DDoS","PortScan","Bot","DoS-Hulk",
                        "DoS-GoldenEye","DoS-Slowloris","DoS-Slowhttptest",
                        "FTP-Patator","SSH-Patator","Web-BruteForce",
                        "Web-XSS","Web-SQLi","Infiltration","Heartbleed"],
            "dist": [0.80,0.05,0.04,0.02,0.02,0.01,0.01,0.01,0.01,0.01,
                     0.005,0.005,0.005,0.001,0.001],
            "n_feat": 78, "prefix": "cic_"},
        "nsl-kdd": {
            "classes": ["Normal","DoS","Probe","R2L","U2R"],
            "dist": [0.52,0.36,0.095,0.02,0.005],
            "n_feat": 41, "prefix": "kdd_"},
    }
    key = dataset.lower().replace("_", "-")
    t   = templates.get(key, templates["cicids2017"])
    d   = np.array(t["dist"]); d /= d.sum()
    y   = rng.choice(len(t["classes"]), size=n, p=d)
    X   = rng.standard_normal((n, t["n_feat"])).astype(np.float32)
    for c in range(len(t["classes"])):
        X[y == c] += c * 0.3
    feat = [f"{t['prefix']}feat_{i}" for i in range(t["n_feat"])]
    print(f"[Synthetic-{dataset}]  n={n}  features={t['n_feat']}"
          f"  classes={t['classes']}")
    return X, y.astype(np.int64), feat, t["classes"]


def load_or_synthetic(name, data_dir, **kwargs):
    try:
        key = name.lower().replace(" ", "-").replace("_", "-")
        if key in ("cicids2017", "cicids"):
            return load_cicids2017(
                data_dir=data_dir,
                **{k: v for k, v in kwargs.items()
                   if k in ("sample_n", "multiclass")})
        raise FileNotFoundError(f"Loader not found for {name}")
    except FileNotFoundError as e:
        print(f"\n  [WARNING] {e}\n  → Falling back to synthetic data.\n")
        return load_synthetic_demo(name)


# ============================================================
#  MAIN PIPELINE
# ============================================================
def run_lens_xai(X, y, feature_names, class_names,
                 dataset_name="Dataset", cfg=None,
                 save_plots=True, output_dir="lens_xai_output",
                 verbose=True, llm_explainer=None,
                 use_smote=False):

    cfg = cfg or CONFIG
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Remove singleton classes ──
    unique_labels, counts = np.unique(y, return_counts=True)
    too_rare = unique_labels[counts < 2]
    if len(too_rare) > 0:
        removed = [class_names[c] for c in too_rare
                   if c < len(class_names)]
        print(f"\n[WARNING] Removing classes with < 2 samples: {removed}")
        mask = ~np.isin(y, too_rare)
        X, y = X[mask], y[mask]

    # ── 2. Re-index labels (fill any gaps after removal) ──
    le_final = LabelEncoder()
    y = le_final.fit_transform(y)
    current_class_names = [class_names[i] for i in le_final.classes_]
    num_classes = len(le_final.classes_)

    print(f"\n{sep}\nDATASET SUMMARY\n{sep}")
    print(f"  Samples     : {len(X):,}")
    print(f"  Features    : {X.shape[1]}")
    print(f"  Classes     : {num_classes}  →  {current_class_names}")
    print(f"  Device      : {DEVICE}")

    # ── 3. Stratified 70 / 15 val / 15 test split ──
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)

    # Some classes can become singleton after the first split on tiny sampled sets.
    # Stratified second split requires at least 2 samples per class in y_tr.
    _, tr_counts = np.unique(y_tr, return_counts=True)
    if tr_counts.min() < 2:
        print("  [SPLIT] Rare class in train fold (<2 samples) after first split; "
              "using non-stratified train/val split.")
        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr, y_tr, test_size=0.15, random_state=42, stratify=None)
    else:
        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr)

    # ── 4. Optional SMOTE (applied only on train, after split & scale) ──
    if use_smote:
        X_tr2, y_tr2 = apply_smote_safe(X_tr2, y_tr2)

    # ── 5. Compute class weights ──
    cw = compute_class_weight(class_weight="balanced",
                              classes=np.unique(y_tr2),
                              y=y_tr2)
    full_cw = torch.FloatTensor(cw).to(DEVICE)

    print(f"\n  Train: {len(X_tr2):,} | Val: {len(X_val):,} "
          f"| Test: {len(X_te):,}")
    print(f"  Class weights: { {n: round(w,3) for n, w in zip(current_class_names, cw)} }")

    # ── 6. Train ──
    trainer = LENSXAITrainer(
        input_dim=X.shape[1],
        num_classes=num_classes,
        cfg=cfg, device=DEVICE,
        class_weights=full_cw)

    trainer.fit(X_tr2, y_tr2, X_val=X_val, y_val=y_val, verbose=verbose)

    # ── 7. Evaluate ──
    print(f"\n{sep}\nEVALUATION\n{sep}")
    results = trainer.evaluate(X_te, y_te,
                               class_names=current_class_names,
                               verbose=True)

    # ── 8. Plots ──
    if save_plots:
        fig_h = plot_training_history(
            trainer.history,
            title=f"{dataset_name} — Training History")
        fig_h.savefig(
            os.path.join(output_dir, f"{dataset_name}_training.png"),
            dpi=120, bbox_inches="tight")
        plt.close(fig_h)

        show_classes = (current_class_names
                        if len(current_class_names) <= 12 else [])
        fig_cm, axes = plt.subplots(1, 2, figsize=(18, 7))
        for ax, name in zip(axes, ("teacher", "student")):
            plot_confusion_matrix(
                y_te, results[name]["preds"],
                classes=show_classes,
                title=f"{dataset_name} — {name.title()}", ax=ax)
        fig_cm.tight_layout()
        fig_cm.savefig(
            os.path.join(output_dir, f"{dataset_name}_confusion.png"),
            dpi=120, bbox_inches="tight")
        plt.close(fig_cm)

        fig_b = plot_metrics_comparison(
            results,
            title=f"{dataset_name} — Teacher vs Student")
        fig_b.savefig(
            os.path.join(output_dir, f"{dataset_name}_metrics.png"),
            dpi=120, bbox_inches="tight")
        plt.close(fig_b)

        fig_d = plot_class_distribution(
            y_te, current_class_names,
            title=f"{dataset_name} — Test Set Distribution",
            save_path=os.path.join(output_dir,
                                   f"{dataset_name}_class_dist.png"))
        plt.close(fig_d)

    # ── 9. XAI Attribution ──
    print(f"\n{sep}\nSTEP 5 — Variable Attribution Explainability\n{sep}")
    explainer = VariableAttributionExplainer(trainer, feature_names)
    x_sample  = X_te[0:1]
    bg        = X_tr2[:cfg["n_background"]]
    tc_idx    = 1 if len(np.unique(y)) > 1 else 0

    attrs = {}
    for mname in ("teacher", "student"):
        print(f"  Computing {mname} attributions … ", end="", flush=True)
        t0   = time.perf_counter()
        attr = explainer.explain(x_sample, bg, target_class=tc_idx,
                                 model=mname, n_bg=cfg["n_background"])
        attrs[mname] = attr
        print(f"{time.perf_counter()-t0:.1f}s")
        print(f"    baseline={attr['baseline']:.4f}  "
              f"prediction={attr['prediction']:.4f}  "
              f"local-accuracy-OK={attr['local_accuracy']}")

    # ── 10. XAI plot (saved before LLM so the image can be attached) ──
    xai_image_path = os.path.join(output_dir,
                                 f"{dataset_name}_attributions_llm.png")
    fig_xai = None
    if save_plots or llm_explainer is not None:
        fig_xai = plot_xai_with_explanation(
            attrs["teacher"], attrs["student"],
            {"teacher_explanation": "", "student_explanation": ""},
            feature_names, dataset_name,
            top_k=cfg["top_k_features"],
            save_path=xai_image_path)
        plt.close(fig_xai)

    # ── 11. LLM Explanation ──
    print(f"\n{sep}\nSTEP 6 — LLM Narrative Explanation\n{sep}")
    if llm_explainer is not None:
        t_ctx = explainer.build_explanation_context(
            attrs["teacher"], feature_names, current_class_names,
            tc_idx, "teacher", dataset_name, results,
            top_k=cfg["top_k_features"])
        s_ctx = explainer.build_explanation_context(
            attrs["student"], feature_names, current_class_names,
            tc_idx, "student", dataset_name, results,
            top_k=cfg["top_k_features"])
        print("  Generating LLM explanations with the XAI image …")
        llm_explanations = llm_explainer.explain_both_models(
            t_ctx, s_ctx, image_path=xai_image_path)
        print(f"\n  ── Teacher ──\n  {llm_explanations['teacher_explanation']}\n")
        print(f"  ── Student ──\n  {llm_explanations['student_explanation']}\n")
        print(f"  ── Comparison ──\n  {llm_explanations['comparative_analysis']}\n")
    else:
        llm_explanations = {
            "teacher_explanation":  "LLM disabled.",
            "student_explanation":  "LLM disabled.",
            "comparative_analysis": "LLM disabled.",
        }

    # ── 12. Save model & results ──
    trainer.save(os.path.join(output_dir, f"{dataset_name}_weights.pt"))

    lines = [f"LENS-XAI + LLM Results — {dataset_name}", "=" * 55]
    for name, res in results.items():
        lines += [
            f"\n{name.upper()} MODEL",
            f"  Accuracy   : {res['accuracy']*100:.2f}%",
            f"  Precision  : {res['precision']*100:.2f}%",
            f"  Recall     : {res['recall']*100:.2f}%",
            f"  F1-Score   : {res['f1']*100:.2f}%",
            f"  Parameters : {res['params']:,}",
            f"  Infer time : {res['ms_per_batch']:.2f} ms/batch",
        ]
    lines += [
        f"\nClasses: {current_class_names}",
        "\n" + "=" * 55,
        "LLM EXPLANATIONS", "=" * 55,
        "\n[Teacher]",    llm_explanations.get("teacher_explanation",  "N/A"),
        "\n[Student]",    llm_explanations.get("student_explanation",  "N/A"),
        "\n[Comparison]", llm_explanations.get("comparative_analysis", "N/A"),
    ]
    txt = "\n".join(lines)
    print("\n" + txt)

    with open(os.path.join(output_dir, f"{dataset_name}_results.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(output_dir,
                           f"{dataset_name}_llm_explanations.json"), "w", encoding="utf-8") as f:
        json.dump(llm_explanations, f, indent=2)

    print(f"\n  All outputs saved to: {output_dir}/")
    return {
        "trainer": trainer, "results": results,
        "explainer": explainer, "attrs": attrs,
        "llm_explanations": llm_explanations,
    }


# ============================================================
#  CLI
# ============================================================
def _parse_args():
    p = argparse.ArgumentParser(
        description="LENS-XAI + Free LLM with CICIDS2017 [COLAB-READY]")
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", "cicids2017", "nsl-kdd"])
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir",  default="lens_xai_output")
    p.add_argument("--sample_n", type=int, default=None)
    p.add_argument("--binary",   action="store_true",
                   help="Binary classification (Benign vs Attack)")
    p.add_argument("--quiet",    action="store_true")
    p.add_argument("--smote",    action="store_true",
                   help="Apply SMOTE on training set after split")
    p.add_argument("--llm",      default="groq",
                   choices=["gemini", "groq", "ollama", "anthropic", "none"],
                   help="Free LLM provider for XAI explanations")
    p.add_argument("--api_key",  default=None,
                   help="API key (overrides env var)")
    p.add_argument("--ollama_model", default="llama3.2",
                   help="Ollama model name (default: llama3.2)")
    p.add_argument("--groq_model",   default=HARDCODED_GROQ_MODEL,
                   help="Groq model name (multimodal recommended)")
    return p.parse_args()


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    args = _parse_args()

    # ── LLM setup ──
    if args.llm == "none":
        llm = None
    else:
        llm = LLMExplainer(
            provider=args.llm,
            api_key=(args.api_key or HARDCODED_GROQ_API_KEY),
            ollama_model=args.ollama_model,
            groq_model=args.groq_model,
        )

    # ── Load data ──
    if args.dataset == "synthetic":
        X, y, feat, cls = load_synthetic_demo(n=5000)
        ds_name = "synthetic"
    else:
        X, y, feat, cls = load_or_synthetic(
            args.dataset, data_dir=args.data_dir,
            multiclass=not args.binary, sample_n=args.sample_n)
        ds_name = args.dataset

    cfg = DATASET_CONFIGS.get(ds_name, CONFIG)

    # ── Run ──
    run_lens_xai(
        X, y, feat, cls,
        dataset_name=ds_name,
        cfg=cfg,
        save_plots=True,
        output_dir=os.path.join(args.out_dir, ds_name),
        verbose=not args.quiet,
        llm_explainer=llm,
        use_smote=args.smote,
    )

    print("\n  LENS-XAI + LLM complete ✓")
