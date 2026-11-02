"""Appearance-based gaze CNN (PyTorch): eye crops + face crop + landmark features -> screen point.

Architecture (iTracker-style, sized for a webcam + one laptop):

    left eye  36x60 ─┐ shared conv tower ─> 128 ─┐
    right eye 36x60 ─┘                    ─> 128 ─┤
    face      64x64 ── small conv tower   ─>  64 ─┼─ FC 128 ─ FC 2  -> screen fraction (x, y)
    11 landmark features (head pose, head position, iris offsets) ─> 32 ─┘

The landmark features are the same ones the sklearn models use (ml/features.py),
so the CNN starts from everything they know and only has to learn what the
pixels add (eyelid shape, iris detail that MediaPipe's iris points blur).
Both eyes go through the SAME conv weights (twice the data per filter); their
outputs are concatenated in a fixed order, so the FC layers still know which is which.

Evaluation follows ml/training.py exactly: whole sessions held out (whole
participants with group_by="person"; 20 s time blocks with < 3 sessions), never random frames. Early stopping uses the LAST 10%
of each TRAINING session, so the held-out session is never looked at while training.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from doomscroll.ml.eye_crops import CROP_SPEC, extract_crops
from doomscroll.ml.features import FEATURE_DIM, FEATURE_NAMES, extract_features
from doomscroll.ml.model import CLAMP_FRAC, AffineCorrection

DEFAULT_CNN_PATH = Path(__file__).resolve().parents[3] / "models" / "gaze_cnn.pt"


# --------------------------------------------------------------------------- model

def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class EyeTower(nn.Module):
    """36x60 -> 128."""

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.body = nn.Sequential(
            _conv(1, w), _conv(w, w), nn.MaxPool2d(2),            # 18x30
            _conv(w, 2 * w), _conv(2 * w, 2 * w), nn.MaxPool2d(2),  # 9x15
            _conv(2 * w, 4 * w), _conv(4 * w, 4 * w), nn.MaxPool2d(2),  # 4x7
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(4 * w * 4 * 7, 128), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fc(self.body(x))


class FaceTower(nn.Module):
    """64x64 -> 64."""

    def __init__(self, width: int = 16):
        super().__init__()
        w = width
        self.body = nn.Sequential(
            _conv(1, w), nn.MaxPool2d(2),          # 32
            _conv(w, 2 * w), nn.MaxPool2d(2),      # 16
            _conv(2 * w, 4 * w), nn.MaxPool2d(2),  # 8
            _conv(4 * w, 4 * w), nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(4 * w * 16, 64), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fc(self.body(x))


class GazeNet(nn.Module):
    def __init__(self, use_face: bool = True, use_aux: bool = True, eye_width: int = 32):
        super().__init__()
        self.use_face, self.use_aux = use_face, use_aux
        self.eye = EyeTower(eye_width)
        self.face = FaceTower() if use_face else None
        self.aux = nn.Sequential(nn.Linear(FEATURE_DIM, 32), nn.ReLU(inplace=True)) if use_aux else None
        d = 2 * 128 + (64 if use_face else 0) + (32 if use_aux else 0)
        self.head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 2))
        with torch.no_grad():  # start predicting the screen centre, not a random corner
            self.head[-1].weight.mul_(0.1)
            self.head[-1].bias.fill_(0.5)

    def forward(self, eyes, face, aux):
        """eyes (B,2,H,W), face (B,1,S,S), aux (B,FEATURE_DIM); all float, already normalized."""
        parts = [self.eye(eyes[:, 0:1]), self.eye(eyes[:, 1:2])]
        if self.use_face:
            parts.append(self.face(face))
        if self.use_aux:
            parts.append(self.aux(aux))
        return self.head(torch.cat(parts, dim=1))


# ------------------------------------------------------------------ tensors & augment

def _img(u8: torch.Tensor) -> torch.Tensor:
    """uint8 -> float, roughly zero-mean unit-ish range (crops are hist-equalized)."""
    return (u8.float() / 255.0 - 0.5) / 0.29


def _augment_imgs(x: torch.Tensor, max_shift_px: float, max_rot_deg: float, max_scale: float) -> torch.Tensor:
    """x (B,C,H,W) float. Small random affine per sample (mimics landmark jitter, so the
    net doesn't trust exact iris-in-crop placement more than it should at runtime)
    + brightness/contrast + noise."""
    b, _, h, w = x.shape
    dev = x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    sc = 1 + (torch.rand(b, device=dev) * 2 - 1) * max_scale
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_shift_px * 2 / w
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_shift_px * 2 / h
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin * h / w, tx], 1), torch.stack([sin * w / h, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    contrast = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * 0.3
    bright = (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * 0.3
    return x * contrast + bright + torch.randn_like(x) * 0.05


@dataclass
class Arrays:
    """Pooled training rows (CPU numpy). Images stay uint8 until a batch is drawn."""

    eyes: np.ndarray       # (N,2,H,W) uint8
    face: np.ndarray       # (N,S,S) uint8
    aux: np.ndarray        # (N,FEATURE_DIM) float32, raw (unnormalized)
    y: np.ndarray          # (N,2) screen fractions
    screen_wh: np.ndarray  # (N,2)
    groups: np.ndarray     # (N,) test-fold group (session id or session#block)
    session: np.ndarray    # (N,) session id
    t_ms: np.ndarray       # (N,)
    participant: np.ndarray  # (N,) who was recorded
    cv_kind: str = ""

    def take(self, idx: np.ndarray) -> "Arrays":
        return Arrays(self.eyes[idx], self.face[idx], self.aux[idx], self.y[idx], self.screen_wh[idx],
                      self.groups[idx], self.session[idx], self.t_ms[idx], self.participant[idx], self.cv_kind)

    def __len__(self):
        return len(self.y)


def pool_crops(sessions, lag_ms=0.0, window_ms=600.0, min_travel_px=40.0, group_by: str = "session") -> Arrays:
    """group_by: "session" (default) or "person" (hold out every session of one participant at a time)."""
    from doomscroll.ml.training import BLOCK_S, MIN_SESSIONS_FOR_SESSION_CV

    parts = []
    for s in sessions:
        k, y = s.labeled_index(lag_ms, window_ms, min_travel_px)
        if len(k):
            parts.append((s, k, y))
    if not parts:
        raise ValueError("No labeled frames (no crop datasets, no face found, or cursor never moved).")
    if group_by not in ("session", "person"):
        raise ValueError(f"group_by must be 'session' or 'person', not {group_by!r}")
    by_session = len(parts) >= MIN_SESSIONS_FOR_SESSION_CV
    eyes, face, aux, ys, wh, groups, sess, ts, who = [], [], [], [], [], [], [], [], []
    for s, k, y in parts:
        t = s.t_ms[s.rows[k]]
        eyes.append(s.eyes[k]); face.append(s.face[k]); aux.append(s.X[k].astype(np.float32)); ys.append(y)
        wh.append(np.tile(s.screen_wh, (len(k), 1)))
        sess.append(np.array([s.session_id] * len(k)))
        who.append(np.array([s.participant or s.session_id] * len(k)))
        if group_by == "person":
            groups.append(who[-1])
        else:
            groups.append(sess[-1] if by_session else np.array([f"{s.session_id}#{int(x // (BLOCK_S * 1000))}" for x in t]))
        ts.append(t)
    if group_by == "person":
        people = sorted({p for s, _, _ in parts for p in [s.participant or s.session_id]})
        if len(people) < 2:
            raise ValueError(f"--cv person needs at least 2 participants; found {people}. "
                             "Check the 'participant' field in each session's meta.json (then build_crops.py --force).")
        kind = f"leave-person-out ({len(people)} people: {', '.join(people)})"
    elif by_session:
        kind = "leave-sessions-out"
    else:
        kind = f"{BLOCK_S:.0f}s time blocks (only {len(parts)} session(s); collect more for a stricter test)"
    return Arrays(np.concatenate(eyes), np.concatenate(face), np.concatenate(aux), np.concatenate(ys).astype(np.float32),
                  np.concatenate(wh).astype(np.float32), np.concatenate(groups), np.concatenate(sess),
                  np.concatenate(ts), np.concatenate(who), kind)


def tail_split(data: Arrays, frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Inner validation = the last `frac` (by time) of every session in `data`.
    Contiguous, so it is not a near-duplicate of the training frames."""
    val = np.zeros(len(data), bool)
    for sid in np.unique(data.session):
        m = np.flatnonzero(data.session == sid)
        cut = np.quantile(data.t_ms[m], 1 - frac)
        val[m[data.t_ms[m] >= cut]] = True
    return np.flatnonzero(~val), np.flatnonzero(val)


# ------------------------------------------------------------------------ training

@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_face: bool = True
    use_aux: bool = True
    eye_width: int = 32
    augment: bool = True
    max_shift_px: float = 2.0
    max_rot_deg: float = 4.0
    max_scale: float = 0.05
    aux_noise: float = 0.05     # in standardized units
    train_stride: int = 1       # use every n-th training frame (near-duplicates anyway)
    patience: int = 6           # stop if inner-val error hasn't improved for this many epochs
    seed: int = 0
    device: str = "auto"

    def to_dict(self):
        return dict(self.__dict__)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, aux: np.ndarray) -> "Normalizer":
        return cls(aux.mean(0).astype(np.float32), (aux.std(0) + 1e-6).astype(np.float32))

    def __call__(self, aux: np.ndarray) -> np.ndarray:
        return ((aux - self.mean) / self.std).astype(np.float32)


def _batches(n: int, bs: int, shuffle: bool, rng: np.random.Generator):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def _to_device(data: Arrays, idx: np.ndarray, norm: Normalizer, dev: torch.device):
    eyes = _img(torch.from_numpy(data.eyes[idx]).to(dev))
    face = _img(torch.from_numpy(data.face[idx]).to(dev)).unsqueeze(1)
    aux = torch.from_numpy(norm(data.aux[idx])).to(dev)
    return eyes, face, aux


@torch.no_grad()
def predict_arrays(net: GazeNet, data: Arrays, norm: Normalizer, dev: torch.device, bs: int = 512) -> np.ndarray:
    net.eval()
    out = []
    for idx in _batches(len(data), bs, False, np.random.default_rng(0)):
        out.append(net(*_to_device(data, idx, norm, dev)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2), np.float32)


def px_errors(pred: np.ndarray, data: Arrays) -> np.ndarray:
    return np.linalg.norm((pred - data.y) * data.screen_wh, axis=1)


def train_net(train: Arrays, val: Arrays | None, cfg: TrainConfig, log=print) -> tuple[GazeNet, Normalizer, dict]:
    """Train one network. If `val` is given, keep the epoch with the lowest val px error."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = pick_device(cfg.device)
    if cfg.train_stride > 1:
        train = train.take(np.arange(0, len(train), cfg.train_stride))
    norm = Normalizer.fit(train.aux)
    net = GazeNet(cfg.use_face, cfg.use_aux, cfg.eye_width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(train) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(steps, 1), pct_start=0.15)
    px_scale = torch.tensor([1000.0], device=dev)  # loss in units of ~1000 px -> Huber knee ~= 100 px

    best = {"val_px": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(cfg.epochs):
        net.train()
        t0, tot, cnt = time.time(), 0.0, 0
        for idx in _batches(len(train), cfg.batch_size, True, rng):
            eyes, face, aux = _to_device(train, idx, norm, dev)
            if cfg.augment:
                eyes = _augment_imgs(eyes, cfg.max_shift_px, cfg.max_rot_deg, cfg.max_scale)
                face = _augment_imgs(face, cfg.max_shift_px, cfg.max_rot_deg, cfg.max_scale)
                aux = aux + torch.randn_like(aux) * cfg.aux_noise
            y = torch.from_numpy(train.y[idx]).to(dev)
            wh = torch.from_numpy(train.screen_wh[idx]).to(dev)
            pred = net(eyes, face, aux)
            loss = F.smooth_l1_loss((pred - y) * wh / px_scale, torch.zeros_like(y), beta=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
            cnt += len(idx)
        row = {"epoch": epoch + 1, "train_loss": tot / max(cnt, 1), "secs": time.time() - t0}
        if val is not None and len(val):
            row["val_px"] = float(px_errors(predict_arrays(net, val, norm, dev), val).mean())
            if row["val_px"] < best["val_px"]:
                best = {"val_px": row["val_px"], "epoch": epoch + 1,
                        "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}}
        history.append(row)
        log(f"    epoch {row['epoch']:3d}  loss {row['train_loss']:.4f}"
            + (f"  val {row['val_px']:6.0f}px" if "val_px" in row else "") + f"  ({row['secs']:.0f}s)")
        if val is not None and epoch + 1 - best["epoch"] >= cfg.patience:
            log(f"    early stop (best epoch {best['epoch']})")
            break
    if best["state"] is not None:
        net.load_state_dict(best["state"])
    return net, norm, {"history": history, "best_epoch": best["epoch"], "best_val_px": best["val_px"]}


def cross_validate(data: Arrays, cfg: TrainConfig, max_folds: int = 5, log=print) -> np.ndarray:
    """Out-of-fold predictions for every row, held out by group (see module doc)."""
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(data.groups))
    if n_groups < 2:
        raise ValueError("Need at least 2 groups (sessions or 20 s blocks) to cross-validate; record more data.")
    pred = np.zeros_like(data.y)
    dev = pick_device(cfg.device)
    folds = GroupKFold(n_splits=min(max_folds, n_groups)).split(data.y, data.y, data.groups)
    for f, (tr, te) in enumerate(folds):
        if data.cv_kind.startswith("leave-person"):
            held = ", ".join(sorted(set(data.participant[te])))
        elif data.cv_kind.startswith("leave"):
            held = ", ".join(sorted(set(data.session[te])))
        else:
            held = f"{len(set(data.groups[te]))} blocks"
        log(f"  fold {f + 1}: test = {held}  ({len(tr)} train / {len(te)} test frames)")
        train = data.take(tr)
        itr, iva = tail_split(train)
        net, norm, _ = train_net(train.take(itr), train.take(iva), cfg, log)
        pred[te] = predict_arrays(net, data.take(te), norm, dev)
        log(f"  fold {f + 1}: held-out mean error {px_errors(pred[te], data.take(te)).mean():.0f}px")
    return pred


def baseline_oof(data: Arrays, names=("ridge", "hgb"), max_folds: int = 5) -> dict[str, np.ndarray]:
    """The sklearn landmark models on EXACTLY the same rows and folds, for a fair comparison."""
    from sklearn.model_selection import GroupKFold

    from doomscroll.ml.training import CANDIDATES

    out = {}
    n_groups = len(np.unique(data.groups))
    for name in names:
        pred = np.zeros_like(data.y)
        for tr, te in GroupKFold(n_splits=min(max_folds, n_groups)).split(data.y, data.y, data.groups):
            m = CANDIDATES[name]()
            m.fit(data.aux[tr], data.y[tr])
            pred[te] = m.predict(data.aux[te])
        out[name] = pred
    return out


# --------------------------------------------------------------------- saved model

class CnnGazeModel:
    """Live-inference wrapper with the same shape as ml.model.TrainedGazeModel:
    predict_px(points_px, pose, gray, screen_wh). Runs on CPU by default (one frame
    at a time is faster on CPU than paying the MPS/GPU transfer cost)."""

    def __init__(self, net: GazeNet, norm: Normalizer, metadata: dict, correction: AffineCorrection | None = None,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.norm = norm
        self.metadata = metadata
        self.correction = correction

    @torch.no_grad()
    def predict_frac(self, points_px: np.ndarray, pose, gray: np.ndarray) -> np.ndarray:
        feats = extract_features(points_px, pose, gray).astype(np.float32)[None]
        eyes, face = extract_crops(gray, points_px)
        e = _img(torch.from_numpy(eyes[None]).to(self.device))
        f = _img(torch.from_numpy(face[None, None]).to(self.device))
        a = torch.from_numpy(self.norm(feats)).to(self.device)
        frac = self.net(e, f, a)[0].cpu().numpy().astype(np.float64)
        if self.correction is not None:
            frac = self.correction.apply(frac)
        return np.clip(frac, *CLAMP_FRAC)

    predict_frac_frame = predict_frac  # same name as TrainedGazeModel's (see ml.model.load_gaze_model)

    def predict_px(self, points_px, pose, gray, screen_wh: tuple[int, int]) -> np.ndarray:
        return self.predict_frac(points_px, pose, gray) * np.array(screen_wh, dtype=np.float64)

    def save(self, path: Path | str = DEFAULT_CNN_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "net_kwargs": {"use_face": self.net.use_face, "use_aux": self.net.use_aux,
                           "eye_width": self.net.eye.body[0][0].out_channels},
            "aux_mean": self.norm.mean, "aux_std": self.norm.std,
            "feature_names": list(FEATURE_NAMES), "crop_spec": CROP_SPEC, "metadata": self.metadata,
        }, path)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CNN_PATH, correction_path: Path | str | None = None,
             device: str = "cpu") -> "CnnGazeModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No CNN model at {path}. Run scripts/train_cnn.py first.")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(ck["feature_names"]) != FEATURE_NAMES or ck["crop_spec"] != CROP_SPEC:
            raise ValueError("CNN was trained with different features/crops than the current code; retrain.")
        net = GazeNet(**ck["net_kwargs"])
        net.load_state_dict(ck["state_dict"])
        correction = None
        if correction_path is not None and Path(correction_path).exists():
            correction = AffineCorrection.load(Path(correction_path))
        return cls(net, Normalizer(np.asarray(ck["aux_mean"]), np.asarray(ck["aux_std"])), ck["metadata"],
                   correction, device)
