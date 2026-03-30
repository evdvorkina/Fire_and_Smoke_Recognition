import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
from collections import defaultdict
from pycocotools import mask as maskUtils
import zipfile
import io
from Model import compute_crop_y_original, make_3class_pred, infer_out_chunked, features_B_278_gpu_for_image_stride, apply_standardizer_gpu
import shutil
from tqdm.auto import tqdm
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
import torchvision.transforms as T
import torch.nn.functional as F
import torch
import torchvision.models as models
import torch.nn as nn
from dataclasses import dataclass
import concurrent.futures
from itertools import product
import math
import time
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Union
import h5py
import glob

# ============================================================
# Unified committee logic (K1..K18 compatible)
# ============================================================

def smoke_committee_pred(mode: str, p1: torch.Tensor, p2: torch.Tensor, p3b: torch.Tensor | None):
    """
    mode:
      - "v12"      : INS1 OR INS2
      - "c12"      : INS1 AND INS2
      - "v12_v3"   : (INS1 OR INS2 OR INS3B)
      - "v12_c3"   : (INS1 OR INS2) AND INS3B
    """
    if mode == "v12":
        return p1 | p2
    if mode == "c12":
        return p1 & p2
    if mode == "v12_v3":
        assert p3b is not None
        return p1 | p2 | p3b
    if mode == "v12_c3":
        assert p3b is not None
        return (p1 | p2) & p3b
    raise ValueError(f"Unknown smoke mode: {mode}")

def fire_committee_pred(mode: str, p4: torch.Tensor, p5: torch.Tensor):
    """
    mode:
      - "4"   : INS4 only
      - "5"   : INS5 only
      - "v45" : INS4 OR INS5
      - "c45" : INS4 AND INS5
    """
    if mode == "4":
        return p4
    if mode == "5":
        return p5
    if mode == "v45":
        return p4 | p5
    if mode == "c45":
        return p4 & p5
    raise ValueError(f"Unknown fire mode: {mode}")

def committee_from_cfg(outs: dict, thr: dict, cfg: dict):
    """
    outs: dict with tensors (N,1) float32/float16 etc on device
    thr: dict with float thresholds keys: ins1, ins2, ins3B, ins4, ins5
    cfg: dict like {"sm":"v12", "fi":"v45", "use_1":True, ...}

    Returns: smoke_pos(bool N), fire_pos(bool N)
    """
    def _get_out(key):
        if key not in outs:
            raise KeyError(f"outs missing '{key}'. Available: {list(outs.keys())}")
        x = outs[key]
        if x.ndim == 2: x = x[:, 0]
        return x

    # booleans (present or zeroed)
    p1 = (_get_out("ins1")  >= float(thr["ins1"])) if cfg.get("use_1", False) else torch.zeros((outs[next(iter(outs))].shape[0],), device=_get_out("ins1").device, dtype=torch.bool)
    p2 = (_get_out("ins2")  >= float(thr["ins2"])) if cfg.get("use_2", False) else torch.zeros_like(p1)
    p4 = (_get_out("ins4")  >= float(thr["ins4"])) if cfg.get("use_4", False) else torch.zeros_like(p1)
    p5 = (_get_out("ins5")  >= float(thr["ins5"])) if cfg.get("use_5", False) else torch.zeros_like(p1)

    p3b = None
    if cfg.get("use_3", False):
        p3b = (_get_out("ins3B") >= float(thr["ins3B"]))

    smoke_pos = smoke_committee_pred(cfg["sm"], p1, p2, p3b)
    fire_pos  = fire_committee_pred(cfg["fi"], p4, p5)
    return smoke_pos, fire_pos


def committee_from_cfg(outs: dict, thr: dict, cfg: dict):
    """
    outs: dict {key: tensor (N,1) or (N,)} on device
    thr: dict thresholds
    cfg: dict(sm=..., fi=..., use_1..use_5)
    """

    # pick device + N from ANY available out (no assumptions about ins1 presence)
    any_key = next(iter(outs.keys()))
    any_out = outs[any_key]
    if any_out.ndim == 2:
        N = any_out.shape[0]
        dev = any_out.device
    else:
        N = any_out.numel()
        dev = any_out.device

    def _get_1d(key):
        if key not in outs:
            raise KeyError(f"outs missing '{key}'. Available: {list(outs.keys())}")
        x = outs[key]
        if x.ndim == 2:
            x = x[:, 0]
        return x

    def _zeros():
        return torch.zeros((N,), device=dev, dtype=torch.bool)

    # base preds
    p1 = (_get_1d("ins1")  >= float(thr["ins1"]))  if cfg.get("use_1", False) else _zeros()
    p2 = (_get_1d("ins2")  >= float(thr["ins2"]))  if cfg.get("use_2", False) else _zeros()
    p4 = (_get_1d("ins4")  >= float(thr["ins4"]))  if cfg.get("use_4", False) else _zeros()
    p5 = (_get_1d("ins5")  >= float(thr["ins5"]))  if cfg.get("use_5", False) else _zeros()

    p3b = None
    if cfg.get("use_3", False):
        p3b = (_get_1d("ins3B") >= float(thr["ins3B"]))

    smoke_pos = smoke_committee_pred(cfg["sm"], p1, p2, p3b)
    fire_pos  = fire_committee_pred(cfg["fi"], p4, p5)
    return smoke_pos, fire_pos

COMMITTEES_CFG = {
    "K1":  dict(sm="v12",     fi="4",   use_1=True, use_2=False,use_3=False,use_4=True, use_5=False),
    "K2":  dict(sm="v12",     fi="4",   use_1=False,use_2=True, use_3=False,use_4=True, use_5=False),
    "K3":  dict(sm="v12",     fi="4",   use_1=True, use_2=True, use_3=False,use_4=True, use_5=False),
    "K4":  dict(sm="c12",     fi="4",   use_1=True, use_2=True, use_3=False,use_4=True, use_5=False),

    "K5":  dict(sm="v12",     fi="5",   use_1=True, use_2=False,use_3=False,use_4=False,use_5=True),
    "K6":  dict(sm="v12",     fi="5",   use_1=False,use_2=True, use_3=False,use_4=False,use_5=True),
    "K7":  dict(sm="v12",     fi="5",   use_1=True, use_2=True, use_3=False,use_4=False,use_5=True),
    "K8":  dict(sm="c12",     fi="5",   use_1=True, use_2=True, use_3=False,use_4=False,use_5=True),

    "K9":  dict(sm="v12_v3",  fi="4",   use_1=True, use_2=True, use_3=True, use_4=True, use_5=False),
    "K10": dict(sm="v12_c3",  fi="4",   use_1=True, use_2=True, use_3=True, use_4=True, use_5=False),

    "K11": dict(sm="v12_v3",  fi="5",   use_1=True, use_2=True, use_3=True, use_4=False,use_5=True),
    "K12": dict(sm="v12_c3",  fi="5",   use_1=True, use_2=True, use_3=True, use_4=False,use_5=True),

    "K13": dict(sm="v12",     fi="v45", use_1=True, use_2=False,use_3=False,use_4=True, use_5=True),
    "K14": dict(sm="v12",     fi="v45", use_1=False,use_2=True, use_3=False,use_4=True, use_5=True),
    "K15": dict(sm="v12",     fi="v45", use_1=True, use_2=True, use_3=False,use_4=True, use_5=True),
    "K16": dict(sm="c12",     fi="v45", use_1=True, use_2=True, use_3=False,use_4=True, use_5=True),

    "K17": dict(sm="v12_v3",  fi="v45", use_1=True, use_2=True, use_3=True, use_4=True, use_5=True),
    "K18": dict(sm="v12_c3",  fi="v45", use_1=True, use_2=True, use_3=True, use_4=True, use_5=True),
}

def _unwrap_out(x):
    """
    Accepts:
      - torch.Tensor
      - (meta, torch.Tensor) or (torch.Tensor, meta) tuple/list
      - dict with tensor under common keys
    Returns: torch.Tensor
    """
    if isinstance(x, torch.Tensor):
        return x

    if isinstance(x, (list, tuple)):
        # pick the first tensor inside (often the last element)
        for v in reversed(x):
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError("infer_out_chunked returned tuple/list without a Tensor inside")

    if isinstance(x, dict):
        for k in ["out", "outs", "logits", "y", "pred", "output"]:
            if k in x and isinstance(x[k], torch.Tensor):
                return x[k]
        # fallback: first tensor value
        for v in x.values():
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError("infer_out_chunked returned dict without a Tensor value inside")

    raise TypeError(f"Unsupported infer_out_chunked output type: {type(x)}")

# ============================================================
# COMMITTEE PIPELINE (from raw image -> full-size mask + timings)
# - stride=1 fixed
# - resize optional (off by default)
# - committee selectable (string id) + easy to extend
# ============================================================

# ----------------------------
# small utils
# ----------------------------
def _now():
    return time.perf_counter()

def _sync(device):
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()

def _to_u8_rgb(img):
    """Accepts np uint8 RGB or PIL. Returns np.uint8 RGB (H,W,3)."""
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def _resize_rgb(img_rgb_u8, target_wh):
    # target_wh: (W,H)
    return cv2.resize(img_rgb_u8, target_wh, interpolation=cv2.INTER_AREA)

def _slice_stats_from_278(X278):
    # Laws+Stats = first 182; Stats = 147:182 (35 dims)
    return X278[:, 147:182]

def _rasterize_mask_proc_from_points(y3_pts_u8, cy_np, cx_np, Hr, Wr):
    """
    y3_pts_u8: (N,) uint8 {0,1,2}
    Returns mask_proc (Hr,Wr) uint8 {0,1,2}
    """
    mask = np.zeros((Hr, Wr), dtype=np.uint8)
    mask[cy_np, cx_np] = y3_pts_u8
    return mask

def _restore_full_mask(mask_proc_u8, H0, W0, crop_y, was_resized, Hr, Wr):
    """
    mask_proc_u8: (Hr,Wr) in processed space (cropped + maybe resized)
    Returns mask_full_u8: (H0,W0) labels:
      0=non, 1=smoke, 2=fire, 3=sky
    """
    mask_full = np.zeros((H0, W0), dtype=np.uint8)
    # sky
    if crop_y > 0:
        mask_full[:crop_y, :] = 3

    if not was_resized:
        # direct paste back
        Hc = H0 - crop_y
        # safety clip if any mismatch
        h = min(Hc, mask_proc_u8.shape[0])
        w = min(W0, mask_proc_u8.shape[1])
        region = mask_full[crop_y:crop_y+h, :w]
        region = np.maximum(region, mask_proc_u8[:h, :w])
        mask_full[crop_y:crop_y+h, :w] = region
        return mask_full

    # resized: upsample back to cropped original size (H0-crop_y, W0)
    Hc = H0 - crop_y
    mask_unres = cv2.resize(mask_proc_u8, (W0, Hc), interpolation=cv2.INTER_NEAREST)
    mask_full[crop_y:, :] = np.maximum(mask_full[crop_y:, :], mask_unres.astype(np.uint8))
    return mask_full

# ============================================================
# Committee registry (extendable)
# Each committee takes "outs" dict and "thr" dict and returns:
#   smoke_pos (bool N), fire_pos (bool N)
# ============================================================

def committee_smoke_C1_voting(outs, thr):
    # INS1+INS2 voting (OR)
    o1 = outs["ins1"].squeeze(1)
    o2 = outs["ins2"].squeeze(1)
    smoke_pos = (o1 >= thr["ins1"]) | (o2 >= thr["ins2"])
    return smoke_pos

def committee_smoke_C2_confirmation(outs, thr):
    # INS1+INS2 confirmation (AND)
    o1 = outs["ins1"].squeeze(1)
    o2 = outs["ins2"].squeeze(1)
    smoke_pos = (o1 >= thr["ins1"]) & (o2 >= thr["ins2"])
    return smoke_pos

def committee_smoke_C4_voting_with_ins3B(outs, thr):
    # (INS1 OR INS2) voting; no confirmation by ins3B (pure voting among 3 not defined here)
    # You asked earlier: "INS3B fixes errors" -> simplest is OR of all three:
    o1 = outs["ins1"].squeeze(1)
    o2 = outs["ins2"].squeeze(1)
    o3 = outs["ins3B"].squeeze(1)
    smoke_pos = (o1 >= thr["ins1"]) | (o2 >= thr["ins2"]) | (o3 >= thr["ins3B"])
    return smoke_pos

def committee_smoke_C6_strict_confirm_by_ins3B(outs, thr):
    # INS1/INS2 voting -> then INS3B confirmation
    o1 = outs["ins1"].squeeze(1)
    o2 = outs["ins2"].squeeze(1)
    o3 = outs["ins3B"].squeeze(1)
    smoke12 = (o1 >= thr["ins1"]) | (o2 >= thr["ins2"])
    smoke_pos = smoke12 & (o3 >= thr["ins3B"])
    return smoke_pos

def committee_fire_F1_voting(outs, thr):
    # INS4 + INS5 voting (OR)
    o4 = outs["ins4"].squeeze(1)
    o5 = outs["ins5"].squeeze(1)
    fire_pos = (o4 >= thr["ins4"]) | (o5 >= thr["ins5"])
    return fire_pos

def committee_fire_F2_confirmation(outs, thr):
    # INS4 + INS5 confirmation (AND)
    o4 = outs["ins4"].squeeze(1)
    o5 = outs["ins5"].squeeze(1)
    fire_pos = (o4 >= thr["ins4"]) & (o5 >= thr["ins5"])
    return fire_pos

# The "full" committees: return both smoke_pos and fire_pos
def committee_K18(outs, thr):
    # smoke: (INS1 OR INS2) confirmed by INS3B; fire: (INS4 OR INS5)
    smoke_pos = committee_smoke_C6_strict_confirm_by_ins3B(outs, thr)
    fire_pos  = committee_fire_F1_voting(outs, thr)
    return smoke_pos, fire_pos

def committee_K15(outs, thr):
    # smoke voting (INS1 OR INS2), fire voting (INS4 OR INS5)
    o1 = outs["ins1"].squeeze(1); o2 = outs["ins2"].squeeze(1)
    smoke_pos = (o1 >= thr["ins1"]) | (o2 >= thr["ins2"])
    fire_pos  = committee_fire_F1_voting(outs, thr)
    return smoke_pos, fire_pos

def committee_K16(outs, thr):
    # smoke confirmation (INS1 AND INS2), fire voting (INS4 OR INS5)
    o1 = outs["ins1"].squeeze(1); o2 = outs["ins2"].squeeze(1)
    smoke_pos = (o1 >= thr["ins1"]) & (o2 >= thr["ins2"])
    fire_pos  = committee_fire_F1_voting(outs, thr)
    return smoke_pos, fire_pos

def _get_needed_models_for_committee(committee_id: str) -> set[str]:
    cfg = COMMITTEES_CFG[committee_id]
    need = set()
    if cfg.get("use_1", False): need.add("ins1")
    if cfg.get("use_2", False): need.add("ins2")
    if cfg.get("use_3", False): need.add("ins3B")
    if cfg.get("use_4", False): need.add("ins4")
    if cfg.get("use_5", False): need.add("ins5")
    return need

# ============================================================
# Main pipeline
# ============================================================

@torch.no_grad()
def run_committee_on_image(
    img_rgb_or_pil,
    models: dict,
    thr: dict,
    committee_id: str = "K18",
    device: str = "cuda",
    *,
    use_resize: bool = False,
    resize_to: tuple[int,int] = (224, 224),   # (W,H)
    stride: int = 1,                          # fixed by you, but kept for completeness
    window: int = 25,
    levels: int = 32,
    chunk_center_rows: int = 64,
    # normalization
    mu278: torch.Tensor = None,
    sigma278: torch.Tensor = None,
    out_dtype: torch.dtype = torch.float16,
    bs_infer: int = 32768,
    # output options
    return_outs: bool = False,
):
    """
    img -> crop -> (optional resize) -> features -> normalize -> infer models -> committee -> y3 -> mask_full (H0,W0)

    Required in `models` dict keys (for K18):
      "ins1","ins2","ins3B","ins4","ins5"
    Required thresholds in `thr`:
      thr["ins1"], thr["ins2"], thr["ins3B"], thr["ins4"], thr["ins5"]

    Returns:
      mask_full_u8: (H0,W0) labels {0,1,2,3}
      info: dict with timings + geometry + throughput
      optionally outs: dict of model outputs (N,1) torch float32 on GPU
    """
    assert committee_id in COMMITTEES_CFG, f"Unknown committee_id={committee_id}. Available: {list(COMMITTEES_CFG.keys())}"
    device = torch.device(device)

    # ---------- 0) load input ----------
    t0 = _now()
    img0 = _to_u8_rgb(img_rgb_or_pil)
    H0, W0 = img0.shape[:2]
    _sync(device)
    t_load = _now() - t0

    # ---------- 1) horizon crop ----------
    t1 = _now()
    crop_y = int(compute_crop_y_original(img0))
    crop_y = int(np.clip(crop_y, 0, H0))
    img_c = img0[crop_y:, :, :] if crop_y > 0 else img0
    Hc, Wc = img_c.shape[:2]
    _sync(device)
    t_crop = _now() - t1

    # ---------- 2) optional resize ----------
    t2 = _now()
    was_resized = False
    if use_resize and ((Hc != resize_to[1]) or (Wc != resize_to[0])):
        img_proc = _resize_rgb(img_c, resize_to)  # (H',W',3) u8
        was_resized = True
    else:
        img_proc = img_c
    Hr, Wr = img_proc.shape[:2]
    _sync(device)
    t_resize = _now() - t2

    # ---------- 3) features ----------
    t3 = _now()
    X278_np, cy_np, cx_np = features_B_278_gpu_for_image_stride(
        img_proc,
        window=window,
        levels=levels,
        device=str(device),
        chunk_center_rows=chunk_center_rows,
        stride=stride,
    )
    _sync(device)
    t_feat = _now() - t3

    N = int(X278_np.shape[0])
    if N == 0:
        # no samples -> return sky+non
        mask_full = np.zeros((H0, W0), dtype=np.uint8)
        if crop_y > 0:
            mask_full[:crop_y, :] = 3
        info = dict(
            committee_id=committee_id,
            H0=H0, W0=W0, crop_y=crop_y,
            Hr=Hr, Wr=Wr, was_resized=int(was_resized),
            N=0,
            timings=dict(load=t_load, crop=t_crop, resize=t_resize, feat=t_feat, norm=0.0, infer=0.0, committee=0.0, restore=0.0),
            px_per_sec=0.0,
        )
        return (mask_full, info, {}) if return_outs else (mask_full, info)

    # ---------- 4) to torch + normalize 278 ----------
    t4 = _now()
    X278 = torch.from_numpy(X278_np).to(device=device, dtype=torch.float16, non_blocking=True)

    if (mu278 is not None) and (sigma278 is not None):
        X278_n = apply_standardizer_gpu(X278, mu278.to(device), sigma278.to(device), out_dtype=out_dtype)
    else:
        # allow running without normalization (not recommended, but sometimes useful for debug)
        X278_n = X278.to(dtype=out_dtype)

    # slices
    X182_n = X278_n[:, :182]
    X35_n  = _slice_stats_from_278(X278_n)
    _sync(device)
    t_norm = _now() - t4

    # ---------- 5) inference ----------
    t5 = _now()
    outs = {}

    # infer only those models that the committee needs
    # (simple dependency check based on committee id; you can make this smarter later)
    need = _get_needed_models_for_committee(committee_id)
    for key in sorted(list(need)):
        m = models[key]

        if key in ["ins1"]:
            raw = infer_out_chunked(m, X182_n, bs=bs_infer)
            out = _unwrap_out(raw)
            outs[key] = out.to(torch.float32)

        elif key in ["ins2", "ins3B", "ins5"]:
            raw = infer_out_chunked(m, X278_n, bs=bs_infer)
            out = _unwrap_out(raw)
            outs[key] = out.to(torch.float32)

        elif key in ["ins4"]:
            raw = infer_out_chunked(m, X35_n, bs=bs_infer)
            out = _unwrap_out(raw)
            outs[key] = out.to(torch.float32)

        else:
            raise ValueError(f"Unknown model key '{key}' (add routing here).")
    _sync(device)
    t_infer = _now() - t5

    # ---------- 6) committee logic -> y3 on points ----------
    t6 = _now()
    cfg = COMMITTEES_CFG[committee_id]
    smoke_pos, fire_pos = committee_from_cfg(outs, thr, cfg)

    # fire priority (your function already exists)
    y3 = make_3class_pred(fire_pos, smoke_pos)  # torch int64 (N,)
    y3_u8 = y3.detach().to("cpu").numpy().astype(np.uint8)

    # rasterize in processed space
    mask_proc = _rasterize_mask_proc_from_points(y3_u8, cy_np, cx_np, Hr, Wr)
    _sync(device)
    t_committee = _now() - t6

    # ---------- 7) restore to full original size ----------
    t7 = _now()
    mask_full = _restore_full_mask(mask_proc, H0, W0, crop_y, was_resized, Hr, Wr)
    t_restore = _now() - t7

    # ---------- throughput ----------
    total_core = t_feat + t_norm + t_infer + t_committee
    px_per_sec = float(N / max(total_core, 1e-12))

    info = dict(
        committee_id=committee_id,
        H0=H0, W0=W0,
        crop_y=crop_y,
        Hr=Hr, Wr=Wr,
        was_resized=int(was_resized),
        N=N,
        timings=dict(
            load=float(t_load),
            crop=float(t_crop),
            resize=float(t_resize),
            feat=float(t_feat),
            norm=float(t_norm),
            infer=float(t_infer),
            committee=float(t_committee),
            restore=float(t_restore),
            core=float(total_core),
            total=float(t_load+t_crop+t_resize+total_core+t_restore),
        ),
        px_per_sec=px_per_sec,
        # quick ratios on centers (not image-level thresholds, just quick debug)
        pred_fire_ratio=float((mask_proc==2).mean()),
        pred_smoke_ratio=float((mask_proc==1).mean()),
        pred_hazard_ratio=float((mask_proc>0).mean()),
    )

    if return_outs:
        return mask_full, info, outs
    return mask_full, info


# ---- CALL (подставь пути) ----
# split_dir = os.path.join(ROOT, "test")
# visualize_committee_masks_one(h5_test_path, split_dir, SAVE_NPZ, img_idx=0)
def benchmark_committees_speed(
    image_paths: list[str],
    committee_ids: list[str],
    models: dict,
    thr: dict,
    *,
    device: str = "cuda",
    use_resize: bool = False,
    resize_to: tuple[int,int] = (224,224),
    mu278=None,
    sigma278=None,
    bs_infer: int = 32768,
    warmup: int = 1,          # прогон без записи, чтобы прогреть GPU/кеши
    repeats: int = 5,         # сколько раз повторять замер (берём median)
    out_xlsx: str = "/content/committee_speed.xlsx",
):
    rows = []

    # ---- warmup (optional) ----
    if warmup > 0 and len(image_paths) > 0 and len(committee_ids) > 0:
        img = Image.open(image_paths[0]).convert("RGB")
        for _ in range(warmup):
            _ = run_committee_on_image(
                img, models=models, thr=thr,
                committee_id=committee_ids[0],
                device=device, use_resize=use_resize, resize_to=resize_to,
                mu278=mu278, sigma278=sigma278,
                bs_infer=bs_infer,
            )

    for img_i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")

        for cid in committee_ids:
            pxs = []
            cores = []
            Ns = []
            totals = []

            for r in range(repeats):
                mask_full, info = run_committee_on_image(
                    img, models=models, thr=thr,
                    committee_id=cid,
                    device=device, use_resize=use_resize, resize_to=resize_to,
                    mu278=mu278, sigma278=sigma278,
                    bs_infer=bs_infer,
                )
                pxs.append(float(info.get("px_per_sec", 0.0)))
                cores.append(float(info["timings"].get("core", 0.0)))
                totals.append(float(info["timings"].get("total", 0.0)))
                Ns.append(int(info.get("N", 0)))

            # robust summary
            row = {
                "image_idx": img_i,
                "image_path": img_path,
                "committee_id": cid,
                "N_windows": int(np.median(Ns)),
                "px_per_sec_median": float(np.median(pxs)),
                "core_sec_median": float(np.median(cores)),
                "total_sec_median": float(np.median(totals)),
                "use_resize": int(bool(use_resize)),
                "resize_to": f"{resize_to[0]}x{resize_to[1]}",
                "device": str(device),
                "bs_infer": int(bs_infer),
                "repeats": int(repeats),
            }
            rows.append(row)
            print(f"[{img_i}] {os.path.basename(img_path)} | {cid}: px/s={row['px_per_sec_median']:.1f}  core={row['core_sec_median']:.3f}s  total={row['total_sec_median']:.3f}s")

    df = pd.DataFrame(rows)

    # ---- aggregate (mean over images per committee) ----
    agg = (
        df.groupby("committee_id", as_index=False)
          .agg(
              images=("image_idx", "nunique"),
              N_windows_med=("N_windows", "median"),
              px_per_sec_mean=("px_per_sec_median", "mean"),
              px_per_sec_med=("px_per_sec_median", "median"),
              core_sec_mean=("core_sec_median", "mean"),
              total_sec_mean=("total_sec_median", "mean"),
          )
          .sort_values("px_per_sec_mean", ascending=False)
    )

    # ---- save excel with 2 sheets ----
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        agg.to_excel(w, index=False, sheet_name="summary_by_committee")
        df.to_excel(w, index=False, sheet_name="per_image")

    print("Saved:", out_xlsx)
    return df, agg