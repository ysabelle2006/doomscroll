# End-to-end tests building synthetic sessions, crop datasets, and training the CNN.
from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from doomscroll.gaze.head_pose import HeadPose
from doomscroll.ml.crop_dataset import CropSession, build_crop_session
from doomscroll.ml.eye_crops import EYE_CORNERS, EYE_H, EYE_W, FACE_SIZE, crop_eye, extract_crops
from doomscroll.ml.features import FEATURE_DIM, FrameResult
from doomscroll.recording.session import SessionMeta, SessionPaths

# Shared screen/frame size and synthetic eye-corner positions.
SCREEN = (1000, 600)
FW, FH = 320, 240
EYES = (((80.0, 110.0), (130.0, 110.0)), ((190.0, 110.0), (240.0, 110.0)))


# Builds a synthetic 478-point face landmark array with eyes at fixed positions.
def fake_points(rng=None) -> np.ndarray:
    pts = np.zeros((478, 2))
    pts[:] = (160.0, 130.0)
    pts[0], pts[1] = (60.0, 50.0), (260.0, 220.0)
    (a0, a1), (b0, b1) = EYE_CORNERS
    pts[a0], pts[a1] = EYES[0]
    pts[b0], pts[b1] = EYES[1]
    if rng is not None:
        pts += rng.normal(0, 0.5, pts.shape)
    return pts


# Draws a synthetic face frame with iris position offset by fx/fy.
def draw_frame(fx: float, fy: float) -> np.ndarray:
    img = np.full((FH, FW), 170, np.uint8)
    cv2.ellipse(img, (160, 130), (95, 85), 0, 0, 360, 140, -1)
    for (l, r) in EYES:
        cx, cy = (l[0] + r[0]) / 2, l[1]
        cv2.ellipse(img, (int(cx), int(cy)), (25, 10), 0, 0, 360, 235, -1)
        ix = int(round(l[0] + (0.25 + 0.5 * fx) * (r[0] - l[0])))
        iy = int(round(cy + (fy - 0.5) * 10))
        cv2.circle(img, (ix, iy), 6, 30, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# Generates a smooth oscillating synthetic cursor path.
def cursor_path(n: int, phase: float):
    t = np.arange(n) * 33.3
    fx = 0.5 + 0.45 * np.sin(t / 900.0 + phase)
    fy = 0.5 + 0.45 * np.sin(t / 1300.0 + 2 * phase)
    return t, fx, fy


# Writes a synthetic recorded session (video + frames CSV + meta) to disk.
def write_session(root, sid: str, n: int, phase: float) -> SessionPaths:
    paths = SessionPaths(root / sid)
    paths.create()
    t, fx, fy = cursor_path(n, phase)
    vw = cv2.VideoWriter(str(paths.webcam_video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (FW, FH))
    with open(paths.frames_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "t_ms", "cursor_x", "cursor_y"])
        for i in range(n):
            vw.write(draw_frame(fx[i], fy[i]))
            w.writerow([i, f"{t[i]:.3f}", f"{fx[i] * SCREEN[0]:.1f}", f"{fy[i] * SCREEN[1]:.1f}"])
    vw.release()
    paths.write_meta(SessionMeta(sid, "tester", *SCREEN, n_frames=n))
    return paths


# Builds a fake frame-processing callable that occasionally reports a blink.
def make_process(seed=0):
    rng = np.random.default_rng(seed)
    pose = HeadPose(0.0, 0.0, 0.0, np.zeros((3, 1)), np.array([[0.0], [0.0], [600.0]]))
    calls = {"i": 0}

    def process(frame, ts):
        calls["i"] += 1
        if calls["i"] % 15 == 0:
            return FrameResult(None, "blink")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feats = rng.normal(size=FEATURE_DIM)
        return FrameResult(feats, "ok", fake_points(rng), pose, gray)

    return process


# Fixture building three synthetic recorded sessions as saved CropSessions.
@pytest.fixture(scope="module")
def crop_sessions(tmp_path_factory):
    root = tmp_path_factory.mktemp("rec")
    out = []
    for s in range(3):
        sp = write_session(root, f"s{s}", 300, phase=s * 1.7)
        cs = build_crop_session(sp.dir, make_process(s))
        cs.save(root / f"s{s}.npz")
        out.append(CropSession.load(root / f"s{s}.npz"))
    return out


# Checks eye/face crop shapes and that eye crops are roughly roll-invariant.
def test_crop_shapes_and_roll_invariance():
    frame = cv2.cvtColor(draw_frame(0.8, 0.5), cv2.COLOR_BGR2GRAY)
    eyes, face = extract_crops(frame, fake_points())
    assert eyes.shape == (2, EYE_H, EYE_W) and eyes.dtype == np.uint8
    assert face.shape == (FACE_SIZE, FACE_SIZE)
    M = cv2.getRotationMatrix2D((160, 120), 15, 1.0)
    tilted = cv2.warpAffine(frame, M, (FW, FH), borderMode=cv2.BORDER_REPLICATE)
    l, r = (np.hstack([np.array(p), 1.0]) @ M.T for p in EYES[0])
    a = crop_eye(frame, np.array(EYES[0][0]), np.array(EYES[0][1])).astype(float)
    b = crop_eye(tilted, l, r).astype(float)
    assert np.abs(a - b).mean() < 20
    assert np.argmin(a.mean(axis=0)) > EYE_W / 2


# Checks a crop dataset round-trips through save/load with consistent labels.
def test_crop_dataset_roundtrip_and_labels(crop_sessions):
    cs = crop_sessions[0]
    assert cs.summary()["blink"] > 0
    assert len(cs.rows) == len(cs.X) == len(cs.eyes) == len(cs.face)
    k, y = cs.labeled_index()
    assert len(k) > 0.5 * len(cs.rows) and y.min() >= 0 and y.max() <= 1


# Checks a stride setting skips the expected number of frames.
def test_stride_skips_frames(tmp_path):
    sp = write_session(tmp_path, "st", 60, 0.0)
    cs = build_crop_session(sp.dir, make_process(), stride=3)
    assert (cs.reasons == "skipped").sum() == 40
    assert np.all(cs.rows % 3 == 0)


# Checks the CNN trains to beat a ridge baseline and its saved model reproduces predictions.
def test_cnn_learns_from_pixels_and_roundtrips(crop_sessions, tmp_path):
    from doomscroll.ml.cnn import CnnGazeModel, TrainConfig, baseline_oof, cross_validate, pool_crops, px_errors, tail_split, train_net
    from doomscroll.ml.model import load_gaze_model

    data = pool_crops(crop_sessions)
    assert data.cv_kind == "leave-sessions-out"
    cfg = TrainConfig(epochs=8, batch_size=64, lr=2e-3, device="cpu", eye_width=8, patience=99)
    pred = cross_validate(data, cfg, log=lambda *_: None)
    cnn_err = px_errors(pred, data).mean()
    ridge_err = px_errors(baseline_oof(data, ["ridge"])["ridge"], data).mean()
    print(f"cnn {cnn_err:.0f}px vs ridge-on-useless-features {ridge_err:.0f}px")
    assert cnn_err < 0.5 * ridge_err

    itr, iva = tail_split(data)
    net, norm, info = train_net(data.take(itr), data.take(iva), cfg, log=lambda *_: None)
    model = CnnGazeModel(net, norm, {"kind": "cnn"})
    model.save(tmp_path / "m.pt")
    loaded = load_gaze_model(tmp_path / "m.pt", with_correction=False)
    frame = cv2.cvtColor(draw_frame(0.9, 0.5), cv2.COLOR_BGR2GRAY)
    pose = HeadPose(0.0, 0.0, 0.0, np.zeros((3, 1)), np.array([[0.0], [0.0], [600.0]]))
    p1 = model.predict_frac(fake_points(), pose, frame)
    p2 = loaded.predict_frac_frame(fake_points(), pose, frame)
    np.testing.assert_allclose(p1, p2, atol=1e-5)
    left = loaded.predict_frac_frame(fake_points(), pose, cv2.cvtColor(draw_frame(0.1, 0.5), cv2.COLOR_BGR2GRAY))
    assert p2[0] > left[0] + 0.2


# Checks loading a model with a changed crop spec raises an error.
def test_cnn_rejects_changed_crop_spec(tmp_path):
    from doomscroll.ml.cnn import CnnGazeModel, GazeNet, Normalizer

    m = CnnGazeModel(GazeNet(eye_width=8), Normalizer(np.zeros(FEATURE_DIM, np.float32), np.ones(FEATURE_DIM, np.float32)), {})
    m.save(tmp_path / "m.pt")
    ck = torch.load(tmp_path / "m.pt", weights_only=False)
    ck["crop_spec"] = {**ck["crop_spec"], "eye_hw": [24, 40]}
    torch.save(ck, tmp_path / "m.pt")
    with pytest.raises(ValueError, match="different features/crops"):
        CnnGazeModel.load(tmp_path / "m.pt")


# Checks leave-person-out cross-validation groups sessions by participant correctly.
def test_cnn_leave_person_out_groups(crop_sessions):
    from dataclasses import replace

    from doomscroll.ml.cnn import TrainConfig, cross_validate, pool_crops

    people = [replace(s, participant=p) for s, p in zip(crop_sessions, ["anni", "bo", "bo"])]
    data = pool_crops(people, group_by="person")
    assert data.cv_kind.startswith("leave-person-out (2 people")
    assert set(data.groups) == {"anni", "bo"}
    logs = []
    cfg = TrainConfig(epochs=1, batch_size=64, device="cpu", eye_width=8)
    cross_validate(data, cfg, log=logs.append)
    assert any("test = anni" in m for m in logs) and any("test = bo" in m for m in logs)
    with pytest.raises(ValueError, match="at least 2 participants"):
        pool_crops([replace(s, participant="anni") for s in crop_sessions], group_by="person")
