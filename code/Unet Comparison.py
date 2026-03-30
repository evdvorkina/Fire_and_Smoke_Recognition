import os
import json
import numpy as np
import cv2
from pycocotools import mask as maskUtils
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch.nn as nn
import matplotlib as plt
from PIL import Image
from collections import defaultdict
from Model import compute_crop_y_original
import time
import glob

R = 12
PIXELS_INPUT = 224*224
PIXELS_INTERIOR = (224-2*R)*(224-2*R)

ROOT = "/content/fire-smoke-segmentation.v4i.coco-segmentation"

train_dir = os.path.join(ROOT, "train")
valid_dir = os.path.join(ROOT, "valid")
test_dir  = os.path.join(ROOT, "test")

crop_train = os.path.join(ROOT, "_cache", "crop_y_train_orig_K11_t0.06.json")
crop_valid = os.path.join(ROOT, "_cache", "crop_y_valid_orig_K11_t0.06.json")
crop_test  = os.path.join(ROOT, "_cache", "crop_y_test_orig_K11_t0.06.json")  # если есть; если нет — скажешь

SEED = 42
KEEP = 1/3


class CocoFireSmokeDataset(Dataset):
    def __init__(self, root_dir, resize=(224,224), keep_frac=1.0, seed=42):
        self.root_dir = root_dir
        self.resize = resize

        ann_path = os.path.join(root_dir, "_annotations.coco.json")
        self.coco = COCO(ann_path)

        img_ids = list(self.coco.imgs.keys())
        img_ids = sorted(img_ids)

        if keep_frac < 1.0:
            rng = np.random.RandomState(seed)
            k = int(np.ceil(len(img_ids) * keep_frac))
            img_ids = rng.choice(img_ids, size=k, replace=False)
            img_ids = sorted(img_ids)

        self.img_ids = img_ids

        # категории
        self.cat_fire = self.coco.getCatIds(catNms=["fire"])[0]
        self.cat_smoke = self.coco.getCatIds(catNms=["smoke"])[0]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.root_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]

        # --- создаём пустые маски ---
        fire_mask = np.zeros((H, W), dtype=np.uint8)
        smoke_mask = np.zeros((H, W), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            m = self.coco.annToMask(ann)

            if ann["category_id"] == self.cat_fire:
                fire_mask = np.maximum(fire_mask, m)
            elif ann["category_id"] == self.cat_smoke:
                smoke_mask = np.maximum(smoke_mask, m)

        # resize
        image = cv2.resize(image, self.resize)
        fire_mask = cv2.resize(fire_mask, self.resize, interpolation=cv2.INTER_NEAREST)
        smoke_mask = cv2.resize(smoke_mask, self.resize, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0

        mask = np.stack([fire_mask, smoke_mask], axis=0).astype(np.float32)

        image = torch.from_numpy(image.transpose(2,0,1))
        mask = torch.from_numpy(mask)

        return image, mask


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _select_subset(file_names, frac, rng):
    if frac >= 1.0:
        return file_names
    k = int(np.ceil(len(file_names) * frac))
    if k <= 0:
        return []
    idx = rng.choice(len(file_names), size=k, replace=False)
    return [file_names[i] for i in idx]

def load_coco_split(split_dir):
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
    return images, cats, anns_by_img

def _ann_to_binary_mask(ann, h, w):
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((h, w), dtype=np.uint8)

    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles) if isinstance(rles, list) else rles
        m = maskUtils.decode(rle)
        return m.astype(np.uint8)

    if isinstance(seg, dict):
        rle = seg
        if "size" not in rle:
            rle = dict(rle)
            rle["size"] = [h, w]
        if isinstance(rle.get("counts", None), list):
            rle = maskUtils.frPyObjects(rle, h, w)
        m = maskUtils.decode(rle)
        if m.ndim == 3:
            m = m[:, :, 0]
        return m.astype(np.uint8)

    return np.zeros((h, w), dtype=np.uint8)

def build_fire_smoke_masks(img_meta, anns, cats_map):
    h = int(img_meta["height"])
    w = int(img_meta["width"])
    fire = np.zeros((h, w), dtype=np.uint8)
    smoke = np.zeros((h, w), dtype=np.uint8)

    for ann in anns:
        cat_id = ann.get("category_id", None)
        if cat_id is None:
            continue

        cat = cats_map.get(cat_id)
        name = cat.get("name", "") if isinstance(cat, dict) else str(cat)
        name = name.lower()

        m = _ann_to_binary_mask(ann, h, w)
        if m is None:
            continue

        if "fire" in name:
            fire = np.maximum(fire, m)
        elif "smoke" in name:
            smoke = np.maximum(smoke, m)

    return fire, smoke

def get_img_path(split_dir, file_name):
    p1 = os.path.join(split_dir, file_name)
    if os.path.exists(p1): return p1
    p2 = os.path.join(split_dir, "images", file_name)
    if os.path.exists(p2): return p2
    raise FileNotFoundError(file_name)

def load_crop_map(crop_json_path):
    with open(crop_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["crop_y_by_file"]  # {file_name: crop_y}

def pad_to_224(img, fire, smoke, out_hw=(224,224)):
    Ht, Wt = out_hw
    H, W = img.shape[:2]
    pad_h = max(0, Ht - H)
    pad_w = max(0, Wt - W)
    if pad_h == 0 and pad_w == 0:
        return img, fire, smoke
    # pad bottom/right (как самый нейтральный вариант)
    img2   = np.pad(img,   ((0,pad_h),(0,pad_w),(0,0)), mode="constant", constant_values=0)
    fire2  = np.pad(fire,  ((0,pad_h),(0,pad_w)), mode="constant", constant_values=0)
    smoke2 = np.pad(smoke, ((0,pad_h),(0,pad_w)), mode="constant", constant_values=0)
    return img2, fire2, smoke2

class FireSmokeCocoCommitteeLike(Dataset):
    def __init__(self, split_dir, crop_json_path, keep_frac=1.0, seed=42,
                 enable_resize=True, resize_to=(224,224)):  # (W,H) как у тебя
        self.split_dir = split_dir
        self.crop_map = load_crop_map(crop_json_path)
        self.keep_frac = float(keep_frac)
        self.seed = int(seed)
        self.enable_resize = bool(enable_resize)
        self.resize_to = tuple(resize_to)

        images_meta, cats_map, anns_by_img = load_coco_split(split_dir)
        self.images_meta = images_meta
        self.cats_map = cats_map
        self.anns_by_img = anns_by_img

        inv_map = {im_meta["file_name"]: im_id for im_id, im_meta in images_meta.items()}

        # 1) file_names in SAME ORDER as committees
        file_names = [im["file_name"] for im in images_meta.values()]
        file_names = [fn for fn in file_names if fn in self.crop_map]

        # 2) same RNG type and subset logic
        rng = np.random.default_rng(self.seed)
        file_names = _select_subset(file_names, self.keep_frac, rng)

        # 3) metas in fixed order
        metas = []
        for fn in file_names:
            metas.append({
                "file_name": fn,
                "crop_y": int(self.crop_map[fn]),
                "img_id_original": inv_map[fn],
            })
        self.metas = metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        m = self.metas[idx]
        fn = m["file_name"]
        crop_y = int(m["crop_y"])
        img_id = m["img_id_original"]

        img_meta = self.images_meta[img_id]
        anns = self.anns_by_img.get(img_id, [])

        img_path = get_img_path(self.split_dir, fn)
        img_rgb = np.array(Image.open(img_path).convert("RGB"))

        fire, smoke = build_fire_smoke_masks(img_meta, anns, self.cats_map)

        # crop horizon BEFORE resize (как в комитетах)
        if crop_y > 0:
            img_rgb = img_rgb[crop_y:, :, :]
            fire = fire[crop_y:, :]
            smoke = smoke[crop_y:, :]

        # committee-style conditional downscale (only if bigger than 224)
        if self.enable_resize:
            Wt, Ht = self.resize_to
            H, W = img_rgb.shape[:2]
            if H > Ht or W > Wt:
                img_rgb = cv2.resize(img_rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)
                fire    = cv2.resize(fire.astype(np.uint8),  (Wt, Ht), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                smoke   = cv2.resize(smoke.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # pad up to 224x224 if smaller
        img_rgb, fire, smoke = pad_to_224(img_rgb, fire, smoke, out_hw=(224,224))

        img = (img_rgb.astype(np.float32) / 255.0)
        img = torch.from_numpy(img.transpose(2,0,1))  # C,H,W

        # 2 канала: [fire, smoke]
        mask = np.stack([fire, smoke], axis=0).astype(np.float32)
        mask = torch.from_numpy(mask)

        return img, mask

 
train_ds = FireSmokeCocoCommitteeLike(train_dir, crop_train, keep_frac=KEEP, seed=SEED)
val_ds   = FireSmokeCocoCommitteeLike(valid_dir, crop_valid, keep_frac=KEEP, seed=SEED)
# test — полностью
test_ds  = FireSmokeCocoCommitteeLike(test_dir,  crop_test,  keep_frac=1.0, seed=SEED)  # если crop_test есть

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

device = get_device()

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
    activation=None
).to(device)

bce  = nn.BCEWithLogitsLoss()
dice = smp.losses.DiceLoss(mode="multilabel")  # 2 независимых канала

def loss_fn(logits, targets):
    return bce(logits, targets) + dice(logits, targets)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def train_one_epoch():
    model.train()
    total = 0.0
    for x, y in tqdm(train_loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
    return total / max(1, len(train_loader))

@torch.no_grad()
def eval_one_epoch():
    model.eval()
    total = 0.0
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)
        total += float(loss.item())
    return total / max(1, len(val_loader))


def load_coco_split(split_dir):
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
    return images, cats, anns_by_img

def _ann_to_binary_mask(ann, h, w):
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles) if isinstance(rles, list) else rles
        m = maskUtils.decode(rle)
        return m.astype(np.uint8)
    if isinstance(seg, dict):
        rle = seg
        if "size" not in rle:
            rle = dict(rle); rle["size"] = [h, w]
        if isinstance(rle.get("counts", None), list):
            rle = maskUtils.frPyObjects(rle, h, w)
        m = maskUtils.decode(rle)
        if m.ndim == 3: m = m[:, :, 0]
        return m.astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)

def build_fire_smoke_masks(img_meta, anns, cats_map):
    h = int(img_meta["height"]); w = int(img_meta["width"])
    fire = np.zeros((h, w), dtype=np.uint8)
    smoke = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        cat_id = ann.get("category_id", None)
        if cat_id is None:
            continue
        name = str(cats_map.get(cat_id, "")).lower()
        m = _ann_to_binary_mask(ann, h, w)
        if "fire" in name:
            fire = np.maximum(fire, m)
        elif "smoke" in name:
            smoke = np.maximum(smoke, m)
    return fire, smoke

def get_img_path(split_dir, file_name):
    p1 = os.path.join(split_dir, file_name)
    if os.path.exists(p1): return p1
    p2 = os.path.join(split_dir, "images", file_name)
    if os.path.exists(p2): return p2
    raise FileNotFoundError(file_name)

def load_crop_map(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["crop_y_by_file"]

def preprocess_committee_like(img_rgb, fire, smoke, crop_y, resize_to=(224,224), enable_resize=True):
    # crop horizon
    if crop_y > 0:
        img_rgb = img_rgb[crop_y:, :, :]
        fire = fire[crop_y:, :]
        smoke = smoke[crop_y:, :]

    # conditional downscale only if bigger than 224
    if enable_resize:
        Wt, Ht = resize_to
        H, W = img_rgb.shape[:2]
        if H > Ht or W > Wt:
            img_rgb = cv2.resize(img_rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)
            fire    = cv2.resize(fire.astype(np.uint8),  (Wt, Ht), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            smoke   = cv2.resize(smoke.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    # pad up to 224
    img_rgb, fire, smoke = pad_to_224(img_rgb, fire, smoke, out_hw=(224,224))
    return img_rgb, fire, smoke

@torch.no_grad()
def binary_metrics_from_pred(pred_pos: torch.Tensor, true_pos: torch.Tensor):
    tp = (pred_pos & true_pos).sum().item()
    fp = (pred_pos & (~true_pos)).sum().item()
    fn = ((~pred_pos) & true_pos).sum().item()
    tn = ((~pred_pos) & (~true_pos)).sum().item()
    P = tp / (tp + fp + 1e-12)
    R = tp / (tp + fn + 1e-12)
    F1 = (2*P*R) / (P + R + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    bal_acc = 0.5*(R + tnr)
    return dict(P=P, R=R, F1=F1, acc=acc, bal_acc=bal_acc, tp=tp, fp=fp, fn=fn, tn=tn)

@torch.no_grad()
def make_3class_pred(fire_pos: torch.Tensor, smoke_pos: torch.Tensor):
    y3 = torch.zeros_like(fire_pos, dtype=torch.int64)
    y3[smoke_pos] = 1
    y3[fire_pos]  = 2
    return y3

@torch.no_grad()
def confusion_3x3(y_true3, y_pred3):
    cm = torch.zeros((3,3), dtype=torch.int64)
    for t in range(3):
        for p in range(3):
            cm[t,p] = ((y_true3==t) & (y_pred3==p)).sum()
    return cm

@torch.no_grad()
def prf_from_cm_3class(cm):
    out = {}
    cm = cm.to(torch.float64)
    for cls, name in [(1,"smoke"), (2,"fire"), (0,"non")]:
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        P = float(tp / (tp + fp + 1e-12))
        R = float(tp / (tp + fn + 1e-12))
        F1 = float((2*P*R) / (P+R+1e-12))
        out[f"{name}_P"] = P
        out[f"{name}_R"] = R
        out[f"{name}_F1"] = F1

    cls_set = [1,2]
    TP = sum(cm[c,c] for c in cls_set)
    FP = sum(cm[:,c].sum() - cm[c,c] for c in cls_set)
    FN = sum(cm[c,:].sum() - cm[c,c] for c in cls_set)
    Pm = float(TP / (TP + FP + 1e-12))
    Rm = float(TP / (TP + FN + 1e-12))
    F1m = float((2*Pm*Rm) / (Pm+Rm+1e-12))
    out["micro_P_fire_smoke"] = Pm
    out["micro_R_fire_smoke"] = Rm
    out["micro_F1_fire_smoke"] = F1m
    out["macro_P_fire_smoke"] = 0.5*(out["smoke_P"] + out["fire_P"])
    out["macro_R_fire_smoke"] = 0.5*(out["smoke_R"] + out["fire_R"])
    out["macro_F1_fire_smoke"] = 0.5*(out["smoke_F1"] + out["fire_F1"])
    return out

def image_detection_rate_from_images(pred_ratio_list, true_ratio_list, thresholds=(0.02,0.05,0.07,0.10)):
    rep = {}
    n = len(pred_ratio_list)
    rep["n_images"] = int(n)
    for thr in thresholds:
        p_hit = [pr >= thr for pr in pred_ratio_list]
        t_hit = [tr >= thr for tr in true_ratio_list]
        det_pred = sum(p_hit)
        det_true = sum(t_hit)
        det_tp   = sum((ph and th) for ph,th in zip(p_hit, t_hit))
        rep[f"img_det_pred@{thr:.4f}"] = det_pred / max(n,1)
        rep[f"img_true@{thr:.4f}"]     = det_true / max(n,1)
        rep[f"img_recall@{thr:.4f}"]   = det_tp / max(det_true,1)
    return rep

def mean_per_image_and_detection_from_lists(
    tp_list, fp_list, fn_list,
    pred_ratio_list, true_ratio_list,
    thresholds=(0.01,0.02,0.05,0.07,0.10),
):
    thr_list = list(thresholds)
    n_used = len(tp_list)
    if n_used == 0:
        rep = {"n_images_used": 0, "P_mean": 0.0, "R_mean": 0.0, "F1_mean": 0.0,
               "pred_ratio_mean": 0.0, "pred_ratio_p90": 0.0,
               "true_ratio_mean": 0.0, "true_ratio_p90": 0.0,
               "n_gt_pos_images": 0, "n_gt_neg_images": 0}
        for t in thr_list:
            rep[f"det_img@{t:.4f}"] = 0.0
            rep[f"recall_img@{t:.4f}"] = 0.0
            rep[f"fpr_img@{t:.4f}"] = 0.0
            rep[f"tp_img@{t:.4f}"] = 0
            rep[f"fp_img@{t:.4f}"] = 0
        return rep

    P_list, R_list, F1_list = [], [], []
    det_all = {t: 0 for t in thr_list}
    det_tp  = {t: 0 for t in thr_list}
    det_fp  = {t: 0 for t in thr_list}
    n_gt_pos = 0
    n_gt_neg = 0

    for tp, fp, fn, pr, tr in zip(tp_list, fp_list, fn_list, pred_ratio_list, true_ratio_list):
        P = tp / (tp + fp + 1e-12)
        R = tp / (tp + fn + 1e-12)
        F1 = (2*P*R) / (P + R + 1e-12)
        P_list.append(P); R_list.append(R); F1_list.append(F1)

        gt_pos_img = (tr > 0.0)
        if gt_pos_img: n_gt_pos += 1
        else:          n_gt_neg += 1

        for t in thr_list:
            if pr >= t:
                det_all[t] += 1
                if gt_pos_img: det_tp[t] += 1
                else:          det_fp[t] += 1

    rep = {
        "n_images_used": int(n_used),
        "P_mean": float(np.mean(P_list)),
        "R_mean": float(np.mean(R_list)),
        "F1_mean": float(np.mean(F1_list)),
        "pred_ratio_mean": float(np.mean(pred_ratio_list)),
        "pred_ratio_p90":  float(np.quantile(pred_ratio_list, 0.90)),
        "true_ratio_mean": float(np.mean(true_ratio_list)),
        "true_ratio_p90":  float(np.quantile(true_ratio_list, 0.90)),
        "n_gt_pos_images": int(n_gt_pos),
        "n_gt_neg_images": int(n_gt_neg),
    }
    for t in thr_list:
        rep[f"det_img@{t:.4f}"]    = det_all[t] / max(n_used, 1)
        rep[f"recall_img@{t:.4f}"] = det_tp[t]  / max(n_gt_pos, 1)
        rep[f"fpr_img@{t:.4f}"]    = det_fp[t]  / max(n_gt_neg, 1)
        rep[f"tp_img@{t:.4f}"]     = int(det_tp[t])
        rep[f"fp_img@{t:.4f}"]     = int(det_fp[t])
    return rep

EPOCHS = 40
best = 1e9

for ep in range(1, EPOCHS+1):
    tr = train_one_epoch()
    va = eval_one_epoch()
    print(f"epoch {ep:03d} | train {tr:.4f} | val {va:.4f}")

    if va < best:
        best = va
        torch.save(model.state_dict(), "best_unet_resnet34_fire_smoke.pth")
TEST_DIR = "/content/fire-smoke-segmentation.v4i.coco-segmentation/test"
CROP_TEST_JSON = "/content/fire-smoke-segmentation.v4i.coco-segmentation/_cache/crop_y_test_orig_K11_t0.06.json"

img_thresholds = (0.0001, 0.0003, 0.0005)

# если WINDOW=25
R = 12   # рамка (с каждой стороны) которую исключаем из метрик
images_meta, cats_map, anns_by_img = load_coco_split(TEST_DIR)
crop_map = load_crop_map(CROP_TEST_JSON)

# порядок как в JSON/комитетах
test_metas = list(images_meta.values())

def build_color_mask(fire_mask, smoke_mask, horizon_mask=None):
    """
    fire_mask, smoke_mask: (H,W) bool или 0/1
    horizon_mask: (H,W) bool — где был отсечён горизонт (до crop)
    """
    H, W = fire_mask.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # fire = red
    out[fire_mask > 0] = (255, 0, 0)

    # smoke = white (перекрывает non-fire)
    out[(smoke_mask > 0) & (fire_mask == 0)] = (255, 255, 255)

    # horizon (если передан)
    if horizon_mask is not None:
        out[horizon_mask > 0] = (0, 0, 255)

    return out

def visualize_one_sample(idx, thr_pixel=0.5):
    im = test_metas[idx]
    fn = im["file_name"]
    img_id = im["id"]

    img_path = get_img_path(TEST_DIR, fn)
    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    anns = anns_by_img.get(img_id, [])
    fire, smoke = build_fire_smoke_masks(im, anns, cats_map)

    crop_y = int(crop_map.get(fn, 0))

    # ---- horizon mask (до crop) ----
    horizon_mask_full = np.zeros_like(fire, dtype=np.uint8)
    if crop_y > 0:
        horizon_mask_full[:crop_y, :] = 1

    # ---- preprocess ----
    img_p, fire_p, smoke_p = preprocess_committee_like(
        img_rgb, fire, smoke, crop_y=crop_y, resize_to=(224,224)
    )

    # horizon mask после crop+resize
    horizon_mask = None
    if crop_y > 0:
        # создаём mask той же формы, что и fire_p
        horizon_mask = np.zeros_like(fire_p, dtype=np.uint8)
        # верхняя строка после crop соответствует горизонту
        # можно визуально обозначить 1 пиксельную линию
        horizon_mask[0:2, :] = 1  # тонкая синяя линия

    # ---- модель ----
    x = torch.from_numpy((img_p.astype(np.float32)/255.0).transpose(2,0,1)).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x))[0].detach().cpu()

    pred_fire  = (prob[0] >= thr_pixel).numpy()
    pred_smoke = (prob[1] >= thr_pixel).numpy()

    # ---- цветные маски ----
    gt_color  = build_color_mask(fire_p, smoke_p, horizon_mask=horizon_mask)
    pred_color = build_color_mask(pred_fire, pred_smoke, horizon_mask=horizon_mask)

    # ---- plot ----
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(img_p)
    plt.title("Image (224x224)")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(gt_color)
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(pred_color)
    plt.title(f"Predicted Mask (thr={thr_pixel})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def preprocess_committee_like_image_only(img_rgb, crop_y, resize_to=(224,224), enable_resize=True):
    # crop horizon
    if crop_y > 0:
        img_rgb = img_rgb[crop_y:, :, :]

    # conditional downscale only if bigger than 224
    if enable_resize:
        Wt, Ht = resize_to
        H, W = img_rgb.shape[:2]
        if H > Ht or W > Wt:
            img_rgb = cv2.resize(img_rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)

    # pad up to 224
    img_rgb, _, _ = pad_to_224(img_rgb, np.zeros(img_rgb.shape[:2], np.uint8), np.zeros(img_rgb.shape[:2], np.uint8), out_hw=(224,224))
    return img_rgb

@torch.no_grad()
def unet_process_frame(img_rgb, crop_y, thr_pixel=0.5):
    """
    Returns:
      pred_fire, pred_smoke (bool HxW) on 224x224
    """
    img_p = preprocess_committee_like_image_only(img_rgb, crop_y=crop_y, resize_to=(224,224), enable_resize=True)
    x = torch.from_numpy((img_p.astype(np.float32)/255.0).transpose(2,0,1)).unsqueeze(0).to(device, non_blocking=True)

    logits = model(x)                     # (1,2,224,224)
    prob = torch.sigmoid(logits)[0]       # (2,224,224)

    pred_fire  = (prob[0] >= thr_pixel)
    pred_smoke = (prob[1] >= thr_pixel)
    return pred_fire, pred_smoke

def read_rgb(path):
    img = np.array(Image.open(path).convert("RGB"))
    return img



def benchmark_with_disk_io(file_list, crop_map, thr_pixel=0.5, warmup=20, n_measure=200):
    # warmup (GPU прогрев)
    for i in range(min(warmup, len(file_list))):
        fn = file_list[i]
        img = read_rgb(fn)
        crop_y = int(compute_crop_y_original(img))
        _ = unet_process_frame(img, crop_y, thr_pixel=thr_pixel)
    if device == "cuda":
        torch.cuda.synchronize()

    # measure
    t0 = time.perf_counter()
    n_done = 0
    for i in range(min(n_measure, len(file_list))):
        fn = file_list[i]
        img = read_rgb(fn)
        crop_y = int(crop_map.get(os.path.basename(fn), 0))
        _ = unet_process_frame(img, crop_y, thr_pixel=thr_pixel)
        n_done += 1

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    dt = t1 - t0
    fps = n_done / max(dt, 1e-12)
    pixps_input = fps * PIXELS_INPUT
    pixps_interior = fps * PIXELS_INTERIOR
    return {
        "n": n_done,
        "sec": dt,
        "fps_images": fps,
        "pixels_per_sec_224": pixps_input,
        "pixels_per_sec_interior": pixps_interior,
    }    

def benchmark_in_memory(images_rgb, crop_y_list, thr_pixel=0.5, warmup=20, n_measure=200):
    # warmup
    for i in range(min(warmup, len(images_rgb))):
        _ = unet_process_frame(images_rgb[i], crop_y_list[i], thr_pixel=thr_pixel)
    if device == "cuda":
        torch.cuda.synchronize()

    # measure
    t0 = time.perf_counter()
    n_done = 0
    for i in range(min(n_measure, len(images_rgb))):
        _ = unet_process_frame(images_rgb[i], crop_y_list[i], thr_pixel=thr_pixel)
        n_done += 1
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    dt = t1 - t0
    fps = n_done / max(dt, 1e-12)
    return {
        "n": n_done,
        "sec": dt,
        "fps_images": fps,
        "pixels_per_sec_224": fps * PIXELS_INPUT,
        "pixels_per_sec_interior": fps * PIXELS_INTERIOR,
    }

# ============================================================
# UNet SPEED BENCHMARK — comparable to committee speed metric
# (same idea: px_per_sec = N / core_time, where core excludes load/crop/resize)
#
# Committee core: feat + norm + infer + committee
# UNet core     : infer (+sigmoid) + threshold  (you can include post if you want)
#
# We also report TOTAL (includes load + horizon + resize/pad + core).
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ---- paths ----
TEST_DIR = "/content/fire-smoke-segmentation.v4i.coco-segmentation/test"

# ---- settings ----
thr_pixel = 0.5
WINDOW = 25
R = WINDOW // 2  # 12

# IMPORTANT: UNet expects 224x224 input
resize_to = (224, 224)  # (W,H)

# comparable "pixels" count:
# committees use N = number of center-points (stride=1 => interior pixels)
N_PIX = (resize_to[1] - 2*R) * (resize_to[0] - 2*R)  # (H-2R)*(W-2R)

WARMUP = 20
N_MEASURE = 125  # set bigger if you want stable mean

def _now():
    return time.perf_counter()

def _sync():
    if device.type == "cuda":
        torch.cuda.synchronize()

def _to_u8_rgb(img):
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def pad_to_224_rgb(img_rgb, out_hw=(224,224)):
    Ht, Wt = out_hw[1], out_hw[0]
    H, W = img_rgb.shape[:2]
    pad_h = max(0, Ht - H)
    pad_w = max(0, Wt - W)
    if pad_h == 0 and pad_w == 0:
        return img_rgb
    return np.pad(img_rgb, ((0,pad_h),(0,pad_w),(0,0)), mode="constant", constant_values=0)

def preprocess_unet_like_committee(img0_u8):
    """
    Matches committee-style stages but ends with 224x224 for UNet:
      load -> crop(horizon) -> resize (always to 224 here) -> pad (safety)
    Returns: img_224_u8, crop_y
    """
    H0, W0 = img0_u8.shape[:2]

    # horizon crop (same function as committees)
    crop_y = int(compute_crop_y_original(img0_u8))
    crop_y = int(np.clip(crop_y, 0, H0))
    img_c = img0_u8[crop_y:, :, :] if crop_y > 0 else img0_u8

    # resize to 224x224 (UNet input)
    img_r = cv2.resize(img_c, resize_to, interpolation=cv2.INTER_AREA)

    # pad (usually not needed after resize, but keep consistent)
    img_p = pad_to_224_rgb(img_r, out_hw=resize_to)
    return img_p, crop_y

@torch.no_grad()
def unet_core_infer(img_224_u8, thr_pixel=0.5):
    """
    Core part analogous to committee 'infer+committee':
      H2D + forward + sigmoid + threshold
    Returns preds on GPU (bool tensors).
    """
    x = torch.from_numpy((img_224_u8.astype(np.float32)/255.0).transpose(2,0,1)).unsqueeze(0)
    x = x.to(device, non_blocking=True)
    logits = model(x)                 # (1,2,224,224)
    prob = torch.sigmoid(logits)[0]   # (2,224,224)
    pred_fire  = (prob[0] >= thr_pixel)
    pred_smoke = (prob[1] >= thr_pixel)
    return pred_fire, pred_smoke

def benchmark_unet_like_committee(paths, thr_pixel=0.5, warmup=20, n_measure=125):
    """
    Returns info comparable to committee speed:
      px_per_sec_core  = N_PIX / mean(core_time)
      px_per_sec_total = N_PIX / mean(total_time)
    where:
      core = infer (+sigmoid+threshold) [GPU-synced]
      total = load + crop + resize + core
    """
    n = min(n_measure, len(paths))
    assert n > 0

    # ---------- warmup (GPU only; minimize CPU noise) ----------
    img0 = _to_u8_rgb(Image.open(paths[0]).convert("RGB"))
    img224, _ = preprocess_unet_like_committee(img0)
    for _ in range(min(warmup, 50)):
        _ = unet_core_infer(img224, thr_pixel=thr_pixel)
    _sync()

    # ---------- measure ----------
    t_load_sum = 0.0
    t_crop_sum = 0.0
    t_resize_sum = 0.0
    t_core_sum = 0.0
    t_total_sum = 0.0

    for i in tqdm(range(n), desc="UNet speed", dynamic_ncols=True):
        p = paths[i]

        t0_total = _now()

        # load
        t0 = _now()
        img0 = _to_u8_rgb(Image.open(p).convert("RGB"))
        t_load = _now() - t0

        # crop + resize (time separately like committee stages)
        t0 = _now()
        H0 = img0.shape[0]
        crop_y = int(compute_crop_y_original(img0))
        crop_y = int(np.clip(crop_y, 0, H0))
        img_c = img0[crop_y:, :, :] if crop_y > 0 else img0
        t_crop = _now() - t0

        t0 = _now()
        img224 = cv2.resize(img_c, resize_to, interpolation=cv2.INTER_AREA)
        img224 = pad_to_224_rgb(img224, out_hw=resize_to)
        t_resize = _now() - t0

        # core (GPU-synced)
        t0 = _now()
        _ = unet_core_infer(img224, thr_pixel=thr_pixel)
        _sync()
        t_core = _now() - t0

        t_total = _now() - t0_total

        t_load_sum += t_load
        t_crop_sum += t_crop
        t_resize_sum += t_resize
        t_core_sum += t_core
        t_total_sum += t_total

    # means
    load_m = t_load_sum / n
    crop_m = t_crop_sum / n
    resize_m = t_resize_sum / n
    core_m = t_core_sum / n
    total_m = t_total_sum / n

    px_per_sec_core = N_PIX / max(core_m, 1e-12)
    px_per_sec_total = N_PIX / max(total_m, 1e-12)

    return {
        "n_images": n,
        "N_pix_interior": int(N_PIX),
        "timings_mean_sec": {
            "load": float(load_m),
            "crop_horizon": float(crop_m),
            "resize_pad": float(resize_m),
            "core_infer": float(core_m),
            "total": float(total_m),
        },
        "px_per_sec_core": float(px_per_sec_core),
        "px_per_sec_total": float(px_per_sec_total),
        "images_per_sec_total": float(n / max(t_total_sum, 1e-12)),
        "thr_pixel": float(thr_pixel),
        "resize_to": tuple(resize_to),
        "R": int(R),
    }

# ---- collect paths ----
paths = sorted(
    glob.glob(os.path.join(TEST_DIR, "*.jpg")) +
    glob.glob(os.path.join(TEST_DIR, "*.png")) +
    glob.glob(os.path.join(TEST_DIR, "images", "*.jpg")) +
    glob.glob(os.path.join(TEST_DIR, "images", "*.png"))
)
print(f"Found {len(paths)} images in TEST_DIR")

res = benchmark_unet_like_committee(paths, thr_pixel=thr_pixel, warmup=WARMUP, n_measure=min(N_MEASURE, len(paths)))

print("\n===== UNet SPEED (COMPARABLE FORMAT) =====")
print("n_images:", res["n_images"])
print("N_pix_interior:", res["N_pix_interior"])
print("timings_mean_sec:", res["timings_mean_sec"])
print(f"px_per_sec_core  (like committee core):  {res['px_per_sec_core']:,.0f}")
print(f"px_per_sec_total (full pipeline total): {res['px_per_sec_total']:,.0f}")
print(f"images/sec total: {res['images_per_sec_total']:.2f}")