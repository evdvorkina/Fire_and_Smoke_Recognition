from __future__ import annotations
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
from collections import defaultdict
from pycocotools import mask as maskUtils
import zipfile
import io

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
from typing import Callable, Dict, List, Optional, Tuple, Union
import h5py
import glob
from random import random
import random

# отсечение по линии горизонта
# Назначение: выполняет изменение размера изображения.
def resize_image(img: np.ndarray, size=(224, 224)):
    """
    Выполняет изменение размера изображения.

    Параметры
    ---------
    img :
        Параметр используется в соответствии с назначением функции.
    size :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `resize_image`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = resize_image(img=img_rgb, size=(224, 224))
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def local_delta_rmsd(gray: np.ndarray, K: int) -> np.ndarray:
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    gray :
        Параметр используется в соответствии с назначением функции.
    K :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `local_delta_rmsd`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = local_delta_rmsd(gray=..., K=...)
    """
    gray = gray.astype(np.float32)
    ex  = cv2.blur(gray, (K, K), borderType=cv2.BORDER_REFLECT)
    ex2 = cv2.blur(gray * gray, (K, K), borderType=cv2.BORDER_REFLECT)
    var = np.maximum(ex2 - ex*ex, 0.0)
    return np.sqrt(var)

# Назначение: выполняет нормализацию входных данных.
def colwise_minmax_norm(a: np.ndarray, eps=1e-8) -> np.ndarray:
    """
    Выполняет нормализацию входных данных.

    Параметры
    ---------
    a :
        Параметр используется в соответствии с назначением функции.
    eps :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `colwise_minmax_norm`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = colwise_minmax_norm(a=..., eps=...)
    """
    mn = a.min(axis=0, keepdims=True)
    mx = a.max(axis=0, keepdims=True)
    return (a - mn) / (mx - mn + eps)

# Назначение: вычисляет признаки, связанные с областями неба.
def compute_sky_score_map(img_rgb):
    """
    Вычисляет признаки, связанные с областями неба.

    Параметры
    ---------
    img_rgb :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `compute_sky_score_map`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = compute_sky_score_map(img_rgb=img_rgb)
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:,:,1].astype(np.float32) / 255.0
    V = hsv[:,:,2].astype(np.float32) / 255.0

    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    tex = np.sqrt(gx*gx + gy*gy)
    tex_n = tex / (tex.mean() + 1e-6)  # >1 = более текстурно

    s_low = 1.0 - np.clip(S, 0, 1)
    v_hi  = np.clip(V, 0, 1)
    tex_low = 1.0 / (1.0 + tex_n)      # меньше текстуры -> ближе к 1

    sky = 0.45*tex_low + 0.35*s_low + 0.20*v_hi
    return np.clip(sky, 0.0, 1.0)

# Назначение: определяет положение линии горизонта или связанные с ней характеристики.
def find_horizon_dp(img_rgb, K=15, t=0.10, alpha=10,
                    y_min_frac=0.10, y_max_frac=0.75,
                    w_info=1.0, w_sky=0.8, w_lowpen=1.2,
                    w_smooth=0.15):
    """
    Определяет положение линии горизонта или связанные с ней характеристики.

    Параметры
    ---------
    img_rgb :
        Параметр используется в соответствии с назначением функции.
    K :
        Параметр используется в соответствии с назначением функции.
    t :
        Параметр используется в соответствии с назначением функции.
    alpha :
        Параметр используется в соответствии с назначением функции.
    y_min_frac :
        Параметр используется в соответствии с назначением функции.
    y_max_frac :
        Параметр используется в соответствии с назначением функции.
    w_info :
        Параметр используется в соответствии с назначением функции.
    w_sky :
        Параметр используется в соответствии с назначением функции.
    w_lowpen :
        Параметр используется в соответствии с назначением функции.
    w_smooth :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `find_horizon_dp`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = find_horizon_dp(img_rgb=img_rgb, K=..., t=...)
    """
    H, W, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    N = (K-1)//2
    beta = max(alpha - N, 1)

    delta = local_delta_rmsd(gray, K)
    dn = colwise_minmax_norm(delta)

    # маска по t (как в статье), но DP может использовать и soft score
    mask = dn >= float(t)
    mask_u8 = (mask.astype(np.uint8)*255)
    mask_u8 = cv2.medianBlur(mask_u8, 3)
    mask = mask_u8 > 0

    # sky-map
    sky = compute_sky_score_map(img_rgb)

    y_min = int(np.floor(y_min_frac * H))
    y_max = int(np.ceil (y_max_frac * H))
    y_min = max(0, min(H-1, y_min))
    y_max = max(0, min(H-1, y_max))
    if y_max <= y_min:
        y_min, y_max = 0, H-1

    # unary score: хотим высокий dn, и чтобы "выше было похоже на небо"
    # sky_above(y,x) ~ среднее sky на полосе [y-N..y-1]
    # сделаем быстро через интегральное изображение по y
    sky_cum = np.cumsum(sky, axis=0)  # HxW
    def mean_sky_above(y):
        y1 = np.clip(y - N, 0, H-1)
        y2 = np.clip(y - 1, 0, H-1)
        # суммирование по вертикали
        top = sky_cum[y2, :] - (sky_cum[y1-1, :] if y1 > 0 else 0)
        denom = np.maximum((y2 - y1 + 1), 1)
        return top / denom

    unary = np.full((H, W), -1e9, dtype=np.float32)
    for y in range(y_min, y_max+1):
        info = dn[y, :]
        info = np.where(mask[y, :], info, info*0.3)  # если не прошёл порог — не запрещаем, но уменьшаем
        sky_ab = mean_sky_above(y)
        low_pen = (y / max(1, H-1))**2  # штраф за низкую линию
        unary[y, :] = (w_info*info + w_sky*sky_ab - w_lowpen*low_pen)

    # DP / Viterbi
    dp = np.full((H, W), -1e9, dtype=np.float32)
    prv = np.full((H, W), -1, dtype=np.int32)

    # init
    dp[:, 0] = unary[:, 0]
    prv[:, 0] = -1

    # transition penalty based on |dy|
    # ограничим |dy| <= beta (как в статье)
    for x in range(1, W):
        for y in range(y_min, y_max+1):
            y1 = max(y_min, y - beta)
            y2 = min(y_max, y + beta)

            prev_vals = dp[y1:y2+1, x-1]
            dys = np.arange(y1, y2+1) - y
            trans = -w_smooth * (dys.astype(np.float32)**2)

            cand = prev_vals + trans
            j = int(np.argmax(cand))
            best_prev_y = y1 + j

            dp[y, x] = unary[y, x] + cand[j]
            prv[y, x] = best_prev_y

    # backtrack
    y_last = int(np.argmax(dp[:, W-1]))
    ys = np.zeros(W, dtype=np.int32)
    ys[W-1] = y_last
    for x in range(W-1, 0, -1):
        ys[x-1] = prv[ys[x], x]

    # smooth
    ys_sm = cv2.medianBlur(ys.astype(np.uint8), 9).astype(np.int32)

    # quality diagnostics
    # truth margin (как (6) по идее): яркость выше vs ниже
    ya = np.clip(ys_sm - N, 0, H-1)
    yb = np.clip(ys_sm + N, 0, H-1)
    qa = gray[ya, np.arange(W)].mean()
    qb = gray[yb, np.arange(W)].mean()
    margin = float(qa - qb)

    # penalize too-low mean
    mean_y = float(ys_sm.mean()) / max(1, H-1)

    return {
        "ys": ys_sm,
        "dn": dn,
        "mask": mask,
        "sky": sky,
        "margin": margin,
        "mean_y": mean_y,
        "K": K,
        "t": float(t),
        "N": N,
        "beta": beta
    }

# Назначение: выполняет вычисление или применение отсечения изображения.
def choose_crop_y_conservative(img_rgb, ys, hazard_top=None):
    """
    Выполняет вычисление или применение отсечения изображения.

    Параметры
    ---------
    img_rgb :
        Параметр используется в соответствии с назначением функции.
    ys :
        Параметр используется в соответствии с назначением функции.
    hazard_top :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `choose_crop_y_conservative`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = choose_crop_y_conservative(img_rgb=img_rgb, ys=..., hazard_top=...)
    """
    H, W, _ = img_rgb.shape
    # базовый crop — низкий перцентиль по линии
    base = int(np.clip(np.percentile(ys, 5), 0, H-1))

    # если линия слишком низко (типичный провал в землю/воду) — отключаем crop
    if base > int(0.45 * H):
        return 0

    # защита от "отрезания" дыма/огня
    if hazard_top is not None and hazard_top > 0:
        base = min(base, max(0, hazard_top - 8))

    return base
# Назначение: формирует или обрабатывает бинарные маски объектов.
def smoke_mask_strict(img_rgb, ys_horizon=None):
    """
    Формирует или обрабатывает бинарные маски объектов.

    Параметры
    ---------
    img_rgb :
        Параметр используется в соответствии с назначением функции.
    ys_horizon :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `smoke_mask_strict`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = smoke_mask_strict(img_rgb=img_rgb, ys_horizon=...)
    """
    H, W, _ = img_rgb.shape
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:,:,1].astype(np.int32)
    V = hsv[:,:,2].astype(np.int32)

    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)

    sky = compute_sky_score_map(img_rgb)

    # базовое: светлый + малонасыщенный
    base = (S < 40) & (V > 120)

    # НЕ небо: sky-score должен быть не слишком высоким
    not_sky = sky < 0.62

    # немного структуры (дым обычно неоднороден, но облака тоже…)
    textured = grad > 7.0

    m = base & not_sky & textured

    # очень важное правило: дым ищем в основном НИЖЕ горизонта
    if ys_horizon is not None:
        yy = np.arange(H)[:, None]
        m = m & (yy >= (ys_horizon[None, :] - 5))  # чуть выше линии разрешим

    # убираем совсем верх кадра
    m[:int(0.15*H), :] = False

    return m

# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def hazard_top(img_rgb, ys_horizon=None):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    img_rgb :
        Параметр используется в соответствии с назначением функции.
    ys_horizon :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `hazard_top`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = hazard_top(img_rgb=img_rgb, ys_horizon=...)
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    Hh = hsv[:,:,0].astype(np.int32)
    S  = hsv[:,:,1].astype(np.int32)
    V  = hsv[:,:,2].astype(np.int32)

    fire = (S > 120) & (V > 80) & ((Hh < 25) | (Hh > 170))
    smoke = smoke_mask_strict(img_rgb, ys_horizon=ys_horizon)

    haz = fire | smoke
    ys = np.where(haz)[0]
    return None if len(ys) == 0 else int(ys.min())

#работа с масками

# Назначение: загружает данные, модель или служебную информацию.
def load_coco_split(split_dir):
    """
    Загружает данные, модель или служебную информацию.

    Параметры
    ---------
    split_dir :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `load_coco_split`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = load_coco_split(split_dir='путь/к/ресурсу')
    """
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
    return images, cats, anns_by_img

import numpy as np
from pycocotools import mask as maskUtils

# Назначение: формирует или обрабатывает бинарные маски объектов.
def _ann_to_binary_mask(ann, h, w):
    """
    Формирует или обрабатывает бинарные маски объектов.

    Параметры
    ---------
    ann :
        Параметр используется в соответствии с назначением функции.
    h :
        Параметр используется в соответствии с назначением функции.
    w :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_ann_to_binary_mask`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _ann_to_binary_mask(ann=..., h=..., w=...)
    """
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((h, w), dtype=np.uint8)

    # 1) polygon формат (COCO polygons): list of lists
    if isinstance(seg, list):
        # seg может быть [poly1, poly2, ...], где каждый poly — [x1,y1,x2,y2,...]
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles) if isinstance(rles, list) else rles
        m = maskUtils.decode(rle)  # (h,w) uint8
        return m.astype(np.uint8)

    # 2) RLE dict
    if isinstance(seg, dict):
        rle = seg

        # иногда size отсутствует/не совпадает — принудительно зададим
        if "size" not in rle:
            rle = dict(rle)
            rle["size"] = [h, w]

        # если counts = list (uncompressed rle), frPyObjects переведёт в compressed
        if isinstance(rle.get("counts", None), list):
            rle = maskUtils.frPyObjects(rle, h, w)

        m = maskUtils.decode(rle)
        # pycocotools иногда возвращает (h,w,1)
        if m.ndim == 3:
            m = m[:, :, 0]
        return m.astype(np.uint8)

    # иначе непонятный формат
    return np.zeros((h, w), dtype=np.uint8)

# Назначение: формирует или обрабатывает бинарные маски объектов.
def build_fire_smoke_masks(img_meta, anns, cats_map):
    """
    Формирует или обрабатывает бинарные маски объектов.

    Параметры
    ---------
    img_meta :
        Параметр используется в соответствии с назначением функции.
    anns :
        Параметр используется в соответствии с назначением функции.
    cats_map :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `build_fire_smoke_masks`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = build_fire_smoke_masks(img_meta=img_rgb, anns=..., cats_map=...)
    """
    h = int(img_meta["height"])
    w = int(img_meta["width"])

    fire = np.zeros((h, w), dtype=np.uint8)
    smoke = np.zeros((h, w), dtype=np.uint8)

    for ann in anns:
        cat_id = ann.get("category_id", None)
        if cat_id is None:
            continue

        # имя категории
        cat = cats_map.get(cat_id)
        if isinstance(cat, dict):
            name = cat.get("name", "")
        else:
            name = str(cat)

        name = name.lower()

        m = _ann_to_binary_mask(ann, h, w)
        if m is None:
            continue

        if "fire" in name:
            fire = np.maximum(fire, m)
        elif "smoke" in name:
            smoke = np.maximum(smoke, m)

    return fire, smoke

# Назначение: возвращает вычисляемый объект или конфигурацию.
def get_img_path(split_dir, file_name):
    """
    Возвращает вычисляемый объект или конфигурацию.

    Параметры
    ---------
    split_dir :
        Параметр используется в соответствии с назначением функции.
    file_name :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `get_img_path`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = get_img_path(split_dir='путь/к/ресурсу', file_name=...)
    """
    p1 = os.path.join(split_dir, file_name)
    if os.path.exists(p1): return p1
    p2 = os.path.join(split_dir, "images", file_name)
    if os.path.exists(p2): return p2
    raise FileNotFoundError(file_name)

# подготовка изображения
# горизонт: как ты попросила
HORIZON_K = 11
HORIZON_T = 0.06
ROOT = "/content/fire-smoke-segmentation.v4i.coco-segmentation"
SPLITS = ["train", "valid", "test"]
CACHE_DIR = os.path.join(ROOT, "_cache")
WINDOW = 25
R = WINDOW // 2
STRIDE = 1

# Назначение: выполняет вычисление или применение отсечения изображения.
def _process_image_for_crop(img_path):
    """
    Выполняет вычисление или применение отсечения изображения.

    Параметры
    ---------
    img_path :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_process_image_for_crop`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _process_image_for_crop(img_path='путь/к/ресурсу')
    """
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)
    return img_np, compute_crop_y_original(img_np)

# Назначение: выполняет вычисление или применение отсечения изображения.
def compute_crop_y_original(img_rgb):
    # строго как у тебя, только фиксируем K=11, t=0.06 и без сеток
    """
    Выполняет вычисление или применение отсечения изображения.

    Параметры
    ---------
    img_rgb :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `compute_crop_y_original`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = compute_crop_y_original(img_rgb=img_rgb)
    """
    r = find_horizon_dp(img_rgb, K=HORIZON_K, t=HORIZON_T)
    ys = r["ys"]
    ys = ys.squeeze() # Ensure ys is 1D
    haz_top_y = hazard_top(img_rgb, ys_horizon=ys)
    crop_y = choose_crop_y_conservative(img_rgb, ys, hazard_top=haz_top_y)
    return int(crop_y)

# Назначение: выполняет вычисление или применение отсечения изображения.
def build_crop_cache(split):
    """
    Выполняет вычисление или применение отсечения изображения.

    Параметры
    ---------
    split :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `build_crop_cache`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = build_crop_cache(split='train')
    """
    split_dir = os.path.join(ROOT, split)
    images, cats, anns_by_img = load_coco_split(split_dir)

    out_path = os.path.join(CACHE_DIR, f"crop_y_{split}_orig_K{HORIZON_K}_t{HORIZON_T:.2f}.json")
    if os.path.exists(out_path):
        print("Crop cache exists:", out_path)
        return out_path

    crop_map = {}
    image_paths_to_process = []
    for img_id, meta in images.items():
        fn = meta["file_name"]
        img_path = get_img_path(split_dir, fn)
        image_paths_to_process.append((fn, img_path))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks to the executor using the top-level helper function
        futures = {executor.submit(_process_image_for_crop, img_path): fn for fn, img_path in image_paths_to_process}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"crop_y cache {split}"):
            _img_np, crop_y_result = future.result()
            fn = futures[future]
            crop_map[fn] = crop_y_result

    payload = {
        "split": split,
        "mode": "original_no_resize",
        "K": HORIZON_K,
        "t": float(HORIZON_T),
        "crop_y_by_file": crop_map
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path, "count=", len(crop_map))
    return out_path

#вычисление признаков

# Назначение: возвращает вычисляемый объект или конфигурацию.
def get_device():
    """
    Возвращает вычисляемый объект или конфигурацию.

    Параметры
    ---------
    Функция не принимает обязательных пользовательских параметров.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `get_device`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = get_device()
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Laws kernels 49x7x7 ----------------
# Назначение: вычисляет маски лавса или карты энергий на их основе.
def get_laws_kernels_torch(device, dtype=torch.float32, normalize="l1", eps=1e-6):
    """
    Вычисляет маски Лавса или карты энергий на их основе.

    Параметры
    ---------
    device :
        Параметр используется в соответствии с назначением функции.
    dtype :
        Параметр используется в соответствии с назначением функции.
    normalize :
        Параметр используется в соответствии с назначением функции.
    eps :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `get_laws_kernels_torch`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = get_laws_kernels_torch(device='cuda', dtype=..., normalize=...)
    """
    vecs = {
        "L7": np.array([ 1,  6, 15, 20, 15,  6,  1], dtype=np.float32),
        "E7": np.array([-1, -4, -5,  0,  5,  4,  1], dtype=np.float32),
        "S7": np.array([-1, -2,  1,  4,  1, -2, -1], dtype=np.float32),
        "W7": np.array([-1,  0,  3,  0, -3,  0,  1], dtype=np.float32),
        "R7": np.array([ 1, -2, -1,  4, -1, -2,  1], dtype=np.float32),
        "U7": np.array([ 1, -4,  5,  0, -5,  4, -1], dtype=np.float32),
        "O7": np.array([-1,  6,-15, 20,-15,  6, -1], dtype=np.float32),
    }

    kernels = []
    for v1 in vecs.values():
        for v2 in vecs.values():
            kernels.append(np.outer(v1, v2).astype(np.float32))

    k = torch.from_numpy(np.stack(kernels, axis=0)).to(device=device, dtype=dtype)  # (49,7,7)
    k = k[:, None, :, :]  # (49,1,7,7)

    if normalize is not None:
        if normalize.lower() == "l1":
            den = k.abs().sum(dim=(2,3), keepdim=True).clamp_min(eps)
            k = k / den
        elif normalize.lower() == "l2":
            den = torch.sqrt((k*k).sum(dim=(2,3), keepdim=True)).clamp_min(eps)
            k = k / den
        else:
            raise ValueError("normalize must be None/'l1'/'l2'")

    return k

@torch.no_grad()
# Назначение: вычисляет маски лавса или карты энергий на их основе.
def laws_maps_gpu(
    img_rgb_u8: np.ndarray,
    window: int = 25,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    normalize_kernels: str = "l1",
    eps_energy: float = 1e-12,
    clamp_energy_max: float = 1e6,   # защита от редких выбросов перед log
    use_log1p: bool = True,
):
    """
    Вычисляет маски Лавса или карты энергий на их основе.

    Параметры
    ---------
    img_rgb_u8 :
        Параметр используется в соответствии с назначением функции.
    window :
        Параметр используется в соответствии с назначением функции.
    device :
        Параметр используется в соответствии с назначением функции.
    dtype :
        Параметр используется в соответствии с назначением функции.
    normalize_kernels :
        Параметр используется в соответствии с назначением функции.
    eps_energy :
        Параметр используется в соответствии с назначением функции.
    clamp_energy_max :
        Параметр используется в соответствии с назначением функции.
    use_log1p :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `laws_maps_gpu`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = laws_maps_gpu(img_rgb_u8=img_rgb, window=25, device='cuda')
    """
    assert img_rgb_u8.dtype == np.uint8
    H, W, _ = img_rgb_u8.shape

    x = torch.from_numpy(img_rgb_u8).to(device=device, dtype=dtype)  # (H,W,3) float32
    x = x.permute(2,0,1).unsqueeze(0)                                # (1,3,H,W)

    kernels = get_laws_kernels_torch(
        device=device, dtype=dtype, normalize=normalize_kernels
    )  # (49,1,7,7)

    pad7 = 3
    padw = window // 2

    outs = []
    for ch in range(3):
        xc = x[:, ch:ch+1, :, :]  # (1,1,H,W)

        # reflect padding как у filter2D BORDER_REFLECT
        xc_pad = F.pad(xc, (pad7,pad7,pad7,pad7), mode="reflect")
        resp = F.conv2d(xc_pad, kernels)  # (1,49,H,W), float32

        # энергия = среднее resp^2 в окне window×window с reflect border
        resp2 = resp * resp
        resp2 = F.pad(resp2, (padw,padw,padw,padw), mode="reflect")
        energy = F.avg_pool2d(resp2, kernel_size=window, stride=1, padding=0)  # (1,49,H,W)

        # численная защита + компрессия диапазона
        energy = torch.clamp(energy, min=0.0)
        if clamp_energy_max is not None:
            energy = torch.clamp(energy, max=float(clamp_energy_max))
        if use_log1p:
            energy = torch.log1p(energy + eps_energy)

        outs.append(energy.squeeze(0))  # (49,H,W)

    return torch.cat(outs, dim=0)  # (147,H,W)

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def stats_maps_gpu(img_rgb_u8, img_hls_u8, img_ycbcr_u8, window=25, device="cuda", dtype=torch.float32, eps=1e-12):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    img_rgb_u8 :
        Параметр используется в соответствии с назначением функции.
    img_hls_u8 :
        Параметр используется в соответствии с назначением функции.
    img_ycbcr_u8 :
        Параметр используется в соответствии с назначением функции.
    window :
        Параметр используется в соответствии с назначением функции.
    device :
        Параметр используется в соответствии с назначением функции.
    dtype :
        Параметр используется в соответствии с назначением функции.
    eps :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `stats_maps_gpu`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = stats_maps_gpu(img_rgb_u8=img_rgb, img_hls_u8=img_rgb, img_ycbcr_u8=img_rgb)
    """
    pad = window // 2

    rgb = torch.from_numpy(img_rgb_u8).to(device=device, dtype=dtype).permute(2,0,1).unsqueeze(0)   # (1,3,H,W)
    hls = torch.from_numpy(img_hls_u8).to(device=device, dtype=dtype).permute(2,0,1).unsqueeze(0)   # (1,3,H,W)
    ycc = torch.from_numpy(img_ycbcr_u8).to(device=device, dtype=dtype).permute(2,0,1).unsqueeze(0) # (1,3,H,W)

    Rm, Gm, Bm = rgb[:,0:1], rgb[:,1:2], rgb[:,2:3]
    Hm, Lm, Sm = hls[:,0:1], hls[:,1:2], hls[:,2:3]  # OpenCV HLS
    Gray = ycc[:,0:1]

    def mean_map(x): return F.avg_pool2d(x, window, stride=1, padding=pad)
    def var_map(x):
        m1 = mean_map(x)
        m2 = mean_map(x*x)
        return torch.clamp(m2 - m1*m1, min=0.0)
    def max_map(x): return F.max_pool2d(x, window, stride=1, padding=pad)
    def min_map(x): return -F.max_pool2d(-x, window, stride=1, padding=pad)

    # 1–7: центральные значения (просто карта канала)
    f = [Rm, Gm, Bm, Hm, Sm, Lm, Gray]

    # 8–10: H mean/var/ratio
    MH, DH = mean_map(Hm), var_map(Hm)
    f += [MH, DH, MH / torch.clamp(DH, min=1e-3)]

    # 11–13: S mean/var/ratio
    MS, DS = mean_map(Sm), var_map(Sm)
    f += [MS, DS, MS / torch.clamp(DS, min=1e-3)]

    # 14–16: min/max/range H
    minH, maxH = min_map(Hm), max_map(Hm)
    f += [minH, maxH, maxH-minH]

    # 17–19: min/max/range S
    minS, maxS = min_map(Sm), max_map(Sm)
    f += [minS, maxS, maxS-minS]

    # 20–22: min/max/range L
    minL, maxL = min_map(Lm), max_map(Lm)
    f += [minL, maxL, maxL-minL]

    # 23–31: RGB mean/var/ratio
    for C in (Rm, Gm, Bm):
        MC, DC = mean_map(C), var_map(C)
        f += [MC, DC, MC/ torch.clamp(DC, min=1e-3)]

    # 32–34: Gray mean/var/ratio
    MGr, DGr = mean_map(Gray), var_map(Gray)
    f += [MGr, DGr, MGr / torch.clamp(DGr, min=1e-3)]

    # 35: skew cube-root for Gray: cbrt(E[(x-μ)^3])
    Ex, Ex2, Ex3 = MGr, mean_map(Gray*Gray), mean_map(Gray*Gray*Gray)
    mu3 = Ex3 - 3*Ex*Ex2 + 2*(Ex*Ex*Ex)
    skew = torch.sign(mu3) * torch.pow(torch.abs(mu3) + eps, 1.0/3.0)
    f += [skew]

    return torch.cat(f, dim=1).squeeze(0)  # (35,H,W)

@torch.no_grad()
# Назначение: вычисляет смещение по заданным параметрам.
def alpha_to_offset(alpha_deg: int, r: int):
    """
    Вычисляет смещение по заданным параметрам.

    Параметры
    ---------
    alpha_deg :
        Параметр используется в соответствии с назначением функции.
    r :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `alpha_to_offset`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = alpha_to_offset(alpha_deg=..., r=...)
    """
    a = alpha_deg % 360
    if a == 0:   return ( r,  0)
    if a == 45:  return ( r,  r)
    if a == 90:  return ( 0,  r)
    if a == 135: return (-r,  r)
    if a == 180: return (-r,  0)
    if a == 225: return (-r, -r)
    if a == 270: return ( 0, -r)
    if a == 315: return ( r, -r)
    raise ValueError

# Назначение: возвращает вычисляемый объект или конфигурацию.
def _get_uv_torch(levels, device, dtype=torch.float32):
    """
    Возвращает вычисляемый объект или конфигурацию.

    Параметры
    ---------
    levels :
        Параметр используется в соответствии с назначением функции.
    device :
        Параметр используется в соответствии с назначением функции.
    dtype :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_get_uv_torch`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _get_uv_torch(levels=32, device='cuda', dtype=...)
    """
    u = torch.arange(levels, device=device, dtype=dtype)
    v = torch.arange(levels, device=device, dtype=dtype)
    U, V = torch.meshgrid(u, v, indexing="ij")
    return U, V

@torch.no_grad()
# Назначение: вычисляет признаки харалика или вспомогательные характеристики для них.
def haralick4_from_glcm_torch(C, U, V, eps=1e-12):
    # C: (B,L,L) prob
    """
    Вычисляет признаки Харалика или вспомогательные характеристики для них.

    Параметры
    ---------
    C :
        Параметр используется в соответствии с назначением функции.
    U :
        Параметр используется в соответствии с назначением функции.
    V :
        Параметр используется в соответствии с назначением функции.
    eps :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `haralick4_from_glcm_torch`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = haralick4_from_glcm_torch(C=..., U=..., V=...)
    """
    C = torch.clamp(C, min=1e-6)
    entropy = -(C * torch.log(C + eps)).sum(dim=(1,2))
    energy  = (C * C).sum(dim=(1,2))
    contrast = (((U - V) ** 2)[None,:,:] * C).sum(dim=(1,2))
    homogeneity = (C / (1.0 + torch.abs(U - V)[None,:,:])).sum(dim=(1,2))
    return torch.stack([entropy, energy, contrast, homogeneity], dim=1)  # (B,4)

@torch.no_grad()
# Назначение: строит или обрабатывает матрицы совместной встречаемости уровней серого.
def glcm_batch_bincount(q, dx, dy, levels):
    """
    Строит или обрабатывает матрицы совместной встречаемости уровней серого.

    Параметры
    ---------
    q :
        Параметр используется в соответствии с назначением функции.
    dx :
        Параметр используется в соответствии с назначением функции.
    dy :
        Параметр используется в соответствии с назначением функции.
    levels :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `glcm_batch_bincount`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = glcm_batch_bincount(q=..., dx=..., dy=...)
    """
    B, K, _ = q.shape

    # slicing to align shapes
    if dy >= 0:
        y0 = slice(0, K - dy); y1 = slice(dy, K)
    else:
        y0 = slice(-dy, K);    y1 = slice(0, K + dy)
    if dx >= 0:
        x0 = slice(0, K - dx); x1 = slice(dx, K)
    else:
        x0 = slice(-dx, K);    x1 = slice(0, K + dx)

    a = q[:, y0, x0].reshape(B, -1).to(torch.int64)
    b = q[:, y1, x1].reshape(B, -1).to(torch.int64)

    idx = a * levels + b  # (B,M)
    # batch offsets so we can single bincount
    base = (torch.arange(B, device=q.device, dtype=torch.int64) * (levels*levels))[:, None]
    flat = (idx + base).reshape(-1)

    counts = torch.bincount(flat, minlength=B*levels*levels).to(torch.float32)
    counts = counts.view(B, levels, levels)
    counts = counts.to(torch.float32)
    s = counts.sum(dim=(1,2), keepdim=True)
    C = counts / torch.clamp(s, min=1e-6)
    return C

@torch.no_grad()
# Назначение: вычисляет признаки харалика или вспомогательные характеристики для них.
def haralick96_patches_gpu(hsl_u8_patches, levels=32, neighbor_size=5):
    """
    Вычисляет признаки Харалика или вспомогательные характеристики для них.

    Параметры
    ---------
    hsl_u8_patches :
        Параметр используется в соответствии с назначением функции.
    levels :
        Параметр используется в соответствии с назначением функции.
    neighbor_size :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `haralick96_patches_gpu`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = haralick96_patches_gpu(hsl_u8_patches=..., levels=32, neighbor_size=(224, 224))
    """
    device = hsl_u8_patches.device
    assert neighbor_size % 2 == 1
    r = neighbor_size // 2  # 2 for 5

    # quantize once per channel
    q = (hsl_u8_patches.to(torch.int32) * levels // 256 ).clamp(0, levels-1) # (B,3,25,25)

    U, V = _get_uv_torch(levels, device=device, dtype=torch.float32)

    feats = []
    for c in range(3):
        qc = q[:, c, :, :]  # (B,25,25)
        for alpha in range(0, 360, 45):
            dx, dy = alpha_to_offset(alpha, r=r)
            C = glcm_batch_bincount(qc, dx=dx, dy=dy, levels=levels)  # (B,L,L)
            f4 = haralick4_from_glcm_torch(C, U, V)  # (B,4)
            feats.append(f4)

    return torch.cat(feats, dim=1)  # (B,96)

# объединение признаков в один вектор
@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def rgb_to_hsl255_cpu(img_rgb_u8):
    # полностью как у тебя, чтобы совпадать
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    img_rgb_u8 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `rgb_to_hsl255_cpu`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = rgb_to_hsl255_cpu(img_rgb_u8=img_rgb)
    """
    hls = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HLS)  # H,L,S
    h = (hls[...,0].astype(np.float32) * (255.0/180.0)).clip(0,255).astype(np.uint8)
    l = hls[...,1].astype(np.uint8)
    s = hls[...,2].astype(np.uint8)
    return np.stack([h, s, l], axis=-1)  # H,S,L in 0..255

# Назначение: формирует итератор по заданной области или набору объектов.
def _iter_outrow_chunks(H, r, chunk_out_rows=64):
    # output rows are [r .. H-r-1], total (H-2r)
    """
    Формирует итератор по заданной области или набору объектов.

    Параметры
    ---------
    H :
        Параметр используется в соответствии с назначением функции.
    r :
        Параметр используется в соответствии с назначением функции.
    chunk_out_rows :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_iter_outrow_chunks`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _iter_outrow_chunks(H=..., r=..., chunk_out_rows=...)
    """
    outH = H - 2*r
    start = 0
    while start < outH:
        end = min(outH, start + chunk_out_rows)
        yield start, end
        start = end

# Назначение: формирует итератор по заданной области или набору объектов.
def iter_centers_stride(H, W, r, stride=1):
    """
    Формирует итератор по заданной области или набору объектов.

    Параметры
    ---------
    H :
        Параметр используется в соответствии с назначением функции.
    W :
        Параметр используется в соответствии с назначением функции.
    r :
        Параметр используется в соответствии с назначением функции.
    stride :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `iter_centers_stride`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = iter_centers_stride(H=..., W=..., r=...)
    """
    ys = np.arange(r, H - r, stride, dtype=np.int32)
    xs = np.arange(r, W - r, stride, dtype=np.int32)
    cy, cx = np.meshgrid(ys, xs, indexing="ij")
    return cy.reshape(-1), cx.reshape(-1), ys, xs  # ys/xs пригодятся


# Назначение: формирует итератор по заданной области или набору объектов.
def _iter_centerrow_chunks(ys_centers, chunk_center_rows=64):
    """
    Формирует итератор по заданной области или набору объектов.

    Параметры
    ---------
    ys_centers :
        Параметр используется в соответствии с назначением функции.
    chunk_center_rows :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_iter_centerrow_chunks`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _iter_centerrow_chunks(ys_centers=..., chunk_center_rows=...)
    """
    i = 0
    while i < len(ys_centers):
        j = min(len(ys_centers), i + chunk_center_rows)
        yield i, j
        i = j

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def features_B_278_gpu_for_image_stride(
    img_rgb_u8: np.ndarray,
    window: int = 25,
    levels: int = 32,
    device: str = "cuda",
    chunk_center_rows: int = 64,   # сколько строк центров обрабатывать за раз (для Haralick)
    stride: int = 1,
):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    img_rgb_u8 :
        Параметр используется в соответствии с назначением функции.
    window :
        Параметр используется в соответствии с назначением функции.
    levels :
        Параметр используется в соответствии с назначением функции.
    device :
        Параметр используется в соответствии с назначением функции.
    chunk_center_rows :
        Параметр используется в соответствии с назначением функции.
    stride :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `features_B_278_gpu_for_image_stride`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = features_B_278_gpu_for_image_stride(img_rgb_u8=img_rgb, window=25, levels=32)
    """
    assert img_rgb_u8.dtype == np.uint8
    H, W, _ = img_rgb_u8.shape
    r = window // 2
    assert H > 2*r and W > 2*r
    assert stride >= 1

    # ---------- CPU colors (один раз) ----------
    hls = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HLS)
    ycc = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2YCrCb)

    # H,S,L in 0..255 как у тебя
    hsl255 = rgb_to_hsl255_cpu(img_rgb_u8)  # (H,W,3) uint8

    # ---------- GPU maps for Laws+Stats ----------
    laws  = laws_maps_gpu(img_rgb_u8, window=window, device=device)             # (147,H,W) float32
    stats = stats_maps_gpu(img_rgb_u8, hls, ycc, window=window, device=device)  # (35,H,W) float32
    maps182 = torch.cat([laws, stats], dim=0)                                   # (182,H,W)

    # ---------- centers grid (одна истина для всех) ----------
    ys = np.arange(r, H - r, stride, dtype=np.int32)
    xs = np.arange(r, W - r, stride, dtype=np.int32)
    cy2d, cx2d = np.meshgrid(ys, xs, indexing="ij")
    cy = cy2d.reshape(-1)
    cx = cx2d.reshape(-1)
    n_y = len(ys)
    n_x = len(xs)
    N = n_y * n_x

    # ---------- X182: берём карты в точках центров ----------
    # index_select индексы должны быть torch.long
    ys_t = torch.from_numpy(ys).to(device=device, dtype=torch.long)
    xs_t = torch.from_numpy(xs).to(device=device, dtype=torch.long)

    tmp = maps182.index_select(1, ys_t)    # (182, n_y, W)
    sel = tmp.index_select(2, xs_t)        # (182, n_y, n_x)
    X182 = sel.permute(1, 2, 0).reshape(N, 182)   # (N,182) on GPU
    X182 = torch.clamp(X182, -6e4, 6e4)
    X182_cpu = X182.to(torch.float16).cpu()       # (N,182) CPU float16

    # ---------- Haralick96: считаем только для нужных центров ----------
    # ВАЖНО: Unfold на CUDA НЕ работает с uint8 => режем как uint8, но в unfold подаём float16
    hsl_t_u8 = torch.from_numpy(hsl255).to(device=device, dtype=torch.uint8)    # (H,W,3)
    hsl_t_u8 = hsl_t_u8.permute(2,0,1).unsqueeze(0)                             # (1,3,H,W) uint8

    unfold = torch.nn.Unfold(kernel_size=window, stride=(stride, stride))

    har_parts = []
    for y0 in range(0, n_y, chunk_center_rows):
        y1 = min(n_y, y0 + chunk_center_rows)
        ys_chunk = ys[y0:y1]  # centers y (np.int32), уже stride-сетка

        # входные строки для этих центров: [center-r .. center+r]
        y_in0 = int(ys_chunk[0] - r)
        y_in1 = int(ys_chunk[-1] + r + 1)  # exclusive

        # crop как uint8 -> затем в float16 для unfold
        x_crop = hsl_t_u8[:, :, y_in0:y_in1, :]          # (1,3,chunkH+2r,W) uint8
        x_crop_f = x_crop.to(torch.float16)              # <<< FIX: unfold только float/half на CUDA

        patches = unfold(x_crop_f)                       # (1, 3*K*K, L) float16
        out_y = (x_crop_f.shape[2] - window) // stride + 1
        out_x = (x_crop_f.shape[3] - window) // stride + 1

        if out_x != n_x:
            raise RuntimeError(f"Haralick out_x={out_x} != n_x={n_x}. Проверь stride/границы.")
        if out_y != (y1 - y0):
            raise RuntimeError(f"Haralick out_y={out_y} != chunk_rows={(y1-y0)}. Проверь y_in0/y_in1/stride.")

        patches = patches.transpose(1,2).reshape(-1, 3, window, window)  # (B,3,25,25) float16

        # haralick ожидает uint8 (как у тебя) -> возвращаем назад
        patches_u8 = patches.clamp(0, 255).to(torch.uint8)               # <<< FIX

        har = haralick96_patches_gpu(patches_u8, levels=levels, neighbor_size=5)  # (B,96) float32 GPU
        har  = torch.clamp(har,  -6e4, 6e4)
        har_parts.append(har.to(torch.float16).cpu())                               # (B,96) CPU float16

    X96_cpu = torch.cat(har_parts, dim=0)  # (N,96) float16

    if X96_cpu.shape[0] != X182_cpu.shape[0]:
        raise RuntimeError(
            f"Mismatch rows: X182={X182_cpu.shape[0]} vs X96={X96_cpu.shape[0]} (stride={stride})"
        )

    X278 = torch.cat([X182_cpu, X96_cpu], dim=1).numpy()  # (N,278) float16 numpy
    return X278, cy.astype(np.int32), cx.astype(np.int32)


#сохранение признаков в h5 для обучения в рамках эксперимента
crop_cache_paths = {s: build_crop_cache(s) for s in SPLITS}
# ----------------------------
# helpers
# ----------------------------

# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def _select_subset(file_names, frac, rng):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    file_names :
        Параметр используется в соответствии с назначением функции.
    frac :
        Параметр используется в соответствии с назначением функции.
    rng :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_select_subset`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _select_subset(file_names=..., frac=..., rng=...)
    """
    if frac >= 1.0:
        return file_names
    k = int(np.ceil(len(file_names) * frac))
    if k <= 0:
        return []
    idx = rng.choice(len(file_names), size=k, replace=False)
    return [file_names[i] for i in idx]

# ============================================================
# 1) BUILD: single split -> one .h5 (with sanity checks + offsets)
# ============================================================
# Назначение: создаёт, проверяет или использует кэшированные данные.
def build_h5_cache_B_split_gpu(
    split_name: str,
    out_h5: str,
    keep_frac: float,
    stride: int,
    seed: int = 42,
    chunk_center_rows: int = 64,
    device: str | None = None,
    enable_resize: bool = True,
    resize_to: tuple[int,int] = (224,224),  # (W,H) for cv2.resize
    save_masks_png: bool = False,
    overwrite: bool = False,                # allow rebuild
    strict: bool = True,                    # raise on any mismatch / non-finite
):
    """
    Создаёт, проверяет или использует кэшированные данные.

    Параметры
    ---------
    split_name :
        Параметр используется в соответствии с назначением функции.
    out_h5 :
        Параметр используется в соответствии с назначением функции.
    keep_frac :
        Параметр используется в соответствии с назначением функции.
    stride :
        Параметр используется в соответствии с назначением функции.
    seed :
        Параметр используется в соответствии с назначением функции.
    chunk_center_rows :
        Параметр используется в соответствии с назначением функции.
    device :
        Параметр используется в соответствии с назначением функции.
    enable_resize :
        Параметр используется в соответствии с назначением функции.
    resize_to :
        Параметр используется в соответствии с назначением функции.
    save_masks_png :
        Параметр используется в соответствии с назначением функции.
    overwrite :
        Параметр используется в соответствии с назначением функции.
    strict :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `build_h5_cache_B_split_gpu`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = build_h5_cache_B_split_gpu(split_name='train', out_h5=..., keep_frac=...)
    """
    assert split_name in ["train", "valid", "test"], "split_name must be one of: train/valid/test"
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)

    if device is None:
        device = get_device()

    if os.path.exists(out_h5) and not overwrite:
        print("H5 exists:", out_h5)
        return out_h5

    rng = np.random.default_rng(seed)

    # crop map
    with open(crop_cache_paths[split_name], "r", encoding="utf-8") as f:
        crop_map = json.load(f)["crop_y_by_file"]

    # coco split
    split_dir = os.path.join(ROOT, split_name)
    images_meta, cats_map, anns_by_img = load_coco_split(split_dir)
    inv_map = {im_meta["file_name"]: im_id for im_id, im_meta in images_meta.items()}

    file_names = [im["file_name"] for im in images_meta.values()]
    file_names = [fn for fn in file_names if fn in crop_map]

    # subset
    file_names = _select_subset(file_names, keep_frac, rng)

    # meta list (fixed order)
    metas = []
    for fn in file_names:
        metas.append({
            "file_name": fn,
            "crop_y": int(crop_map[fn]),
            "img_id_original": inv_map[fn],
        })

    dt = h5py.string_dtype(encoding="utf-8")
    d = 278

    # internal constants expected in your notebook/project:
    # WINDOW, R, HORIZON_K, HORIZON_T, CACHE_DIR
    # functions: get_img_path, build_fire_smoke_masks, features_B_278_gpu_for_image_stride

    with h5py.File(out_h5, "w") as h5:
        # ---- meta per image ----
        h5.create_dataset("meta/file_name", data=np.array([m["file_name"] for m in metas], dtype=dt))
        h5.create_dataset("meta/crop_y",     data=np.array([m["crop_y"] for m in metas], dtype=np.int32))

        h5.create_dataset("meta/orig_h", shape=(len(metas),), dtype=np.int32)
        h5.create_dataset("meta/orig_w", shape=(len(metas),), dtype=np.int32)
        h5.create_dataset("meta/crop_h", shape=(len(metas),), dtype=np.int32)
        h5.create_dataset("meta/crop_w", shape=(len(metas),), dtype=np.int32)
        h5.create_dataset("meta/was_resized", shape=(len(metas),), dtype=np.uint8)
        h5.create_dataset("meta/resize_h", shape=(len(metas),), dtype=np.int32)
        h5.create_dataset("meta/resize_w", shape=(len(metas),), dtype=np.int32)

        # NEW: row ranges in samples for each image
        h5.create_dataset("meta/sample_start", shape=(len(metas),), dtype=np.int64)
        h5.create_dataset("meta/sample_count", shape=(len(metas),), dtype=np.int64)

        # ---- attrs ----
        h5.attrs["split"] = split_name
        h5.attrs["variant"] = "B"
        h5.attrs["window"] = int(WINDOW)
        h5.attrs["stride"] = int(stride)
        h5.attrs["K"] = int(HORIZON_K)
        h5.attrs["t"] = float(HORIZON_T)
        h5.attrs["enable_resize"] = int(bool(enable_resize))
        h5.attrs["resize_to_w"] = int(resize_to[0])
        h5.attrs["resize_to_h"] = int(resize_to[1])
        h5.attrs["keep_frac"] = float(keep_frac)
        h5.attrs["device"] = str(device)
        h5.attrs["chunk_center_rows"] = int(chunk_center_rows)

        # ---- samples (extendable) ----
        chunk_rows = 4096
        h5.create_dataset("samples/X", shape=(0, d), maxshape=(None, d),
                          dtype=np.float16, chunks=(chunk_rows, d), compression="lzf")
        h5.create_dataset("samples/img_idx", shape=(0,), maxshape=(None,),
                          dtype=np.int32, chunks=(65536,), compression="lzf")
        h5.create_dataset("samples/cy", shape=(0,), maxshape=(None,),
                          dtype=np.int32, chunks=(65536,), compression="lzf")
        h5.create_dataset("samples/cx", shape=(0,), maxshape=(None,),
                          dtype=np.int32, chunks=(65536,), compression="lzf")
        h5.create_dataset("samples/y_fire", shape=(0,), maxshape=(None,),
                          dtype=np.uint8, chunks=(65536,), compression="lzf")
        h5.create_dataset("samples/y_smoke", shape=(0,), maxshape=(None,),
                          dtype=np.uint8, chunks=(65536,), compression="lzf")

        def append_rows(dset, arr2d):
            n0 = dset.shape[0]
            n1 = n0 + arr2d.shape[0]
            dset.resize((n1,) + dset.shape[1:])
            dset[n0:n1, :] = arr2d

        def append_vec(dset, arr1d):
            n0 = dset.shape[0]
            n1 = n0 + arr1d.shape[0]
            dset.resize((n1,))
            dset[n0:n1] = arr1d

        if save_masks_png:
            masks_dir = os.path.join(CACHE_DIR, "masks_cropped", f"variant_B_{split_name}")
            os.makedirs(masks_dir, exist_ok=True)

        total_written = 0
        pbar = tqdm(range(len(metas)), desc=f"GPU cache B [{split_name}] stride={stride}", dynamic_ncols=True)

        for img_idx in pbar:
            m = metas[img_idx]
            fn = m["file_name"]
            crop_y = int(m["crop_y"])
            img_id_original = m["img_id_original"]

            meta_for_img = images_meta[img_id_original]
            anns_for_img = anns_by_img.get(img_id_original, [])

            img_path = get_img_path(split_dir, fn)

            # load image
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            H0, W0 = img_rgb.shape[:2]
            h5["meta/orig_h"][img_idx] = H0
            h5["meta/orig_w"][img_idx] = W0

            # masks original
            fire_orig, smoke_orig = build_fire_smoke_masks(meta_for_img, anns_for_img, cats_map)

            # crop
            if crop_y > 0:
                img_c = img_rgb[crop_y:, :, :]
                fire_c = fire_orig[crop_y:, :]
                smoke_c = smoke_orig[crop_y:, :]
            else:
                img_c = img_rgb
                fire_c = fire_orig
                smoke_c = smoke_orig

            Hc, Wc = img_c.shape[:2]
            h5["meta/crop_h"][img_idx] = Hc
            h5["meta/crop_w"][img_idx] = Wc

            # default offsets = empty
            h5["meta/sample_start"][img_idx] = total_written
            h5["meta/sample_count"][img_idx] = 0

            if Hc <= 2*R or Wc <= 2*R:
                h5["meta/was_resized"][img_idx] = 0
                h5["meta/resize_h"][img_idx] = Hc
                h5["meta/resize_w"][img_idx] = Wc
                continue

            # conditional resize
            was_resized = 0
            if enable_resize and (Hc > resize_to[1] or Wc > resize_to[0]):
                img_c = cv2.resize(img_c, resize_to, interpolation=cv2.INTER_AREA)
                fire_c = cv2.resize(fire_c.astype(np.uint8), resize_to,
                                    interpolation=cv2.INTER_NEAREST).astype(fire_c.dtype)
                smoke_c = cv2.resize(smoke_c.astype(np.uint8), resize_to,
                                     interpolation=cv2.INTER_NEAREST).astype(smoke_c.dtype)
                was_resized = 1

            Hr, Wr = img_c.shape[:2]
            h5["meta/was_resized"][img_idx] = was_resized
            h5["meta/resize_h"][img_idx] = Hr
            h5["meta/resize_w"][img_idx] = Wr

            if Hr <= 2*R or Wr <= 2*R:
                continue

            # sanity: masks align with image
            if strict:
                assert fire_c.shape[:2] == (Hr, Wr)
                assert smoke_c.shape[:2] == (Hr, Wr)

            # features with stride
            X278, cy, cx = features_B_278_gpu_for_image_stride(
                img_c,
                window=WINDOW,
                levels=32,
                device=device,
                chunk_center_rows=chunk_center_rows,
                stride=stride,
            )

            # sanity: finiteness
            if strict and (not np.isfinite(X278).all()):
                bad = np.argwhere(~np.isfinite(X278))
                i0, j0 = int(bad[0, 0]), int(bad[0, 1])
                raise RuntimeError(f"Non-finite X278 for img={fn}, first bad at row={i0}, col={j0}, val={X278[i0,j0]}")

            # sanity: bounds
            if strict and cy.size > 0:
                if cy.min() < 0 or cx.min() < 0 or cy.max() >= Hr or cx.max() >= Wr:
                    raise RuntimeError(
                        f"Out-of-bounds centers for img={fn}: "
                        f"cy[{cy.min()},{cy.max()}], cx[{cx.min()},{cx.max()}], size={Hr}x{Wr}"
                    )

            y_fire = fire_c[cy, cx].astype(np.uint8)
            y_smoke = smoke_c[cy, cx].astype(np.uint8)

            n = int(X278.shape[0])
            if strict:
                assert n == cy.shape[0] == cx.shape[0] == y_fire.shape[0] == y_smoke.shape[0]
                assert X278.shape[1] == d

            # record range BEFORE append
            start_row = h5["samples/X"].shape[0]
            img_idx_vec = np.full((n,), img_idx, dtype=np.int32)

            append_rows(h5["samples/X"], X278)
            append_vec(h5["samples/img_idx"], img_idx_vec)
            append_vec(h5["samples/cy"], cy.astype(np.int32))
            append_vec(h5["samples/cx"], cx.astype(np.int32))
            append_vec(h5["samples/y_fire"], y_fire)
            append_vec(h5["samples/y_smoke"], y_smoke)

            # record per-image offsets
            h5["meta/sample_start"][img_idx] = start_row
            h5["meta/sample_count"][img_idx] = n

            total_written += n
            pbar.set_postfix({"img": fn, "samples": total_written, "resized": was_resized})

            if save_masks_png:
                base = os.path.splitext(os.path.basename(fn))[0]
                cv2.imwrite(os.path.join(masks_dir, f"{base}__fire.png"),  (fire_c*255).astype(np.uint8))
                cv2.imwrite(os.path.join(masks_dir, f"{base}__smoke.png"), (smoke_c*255).astype(np.uint8))

        # final consistency check: all sample datasets same length
        nX = h5["samples/X"].shape[0]
        for k in ["img_idx", "cy", "cx", "y_fire", "y_smoke"]:
            nk = h5[f"samples/{k}"].shape[0]
            if nk != nX:
                msg = f"FINAL MISMATCH: samples/X has {nX} rows but samples/{k} has {nk}"
                if strict:
                    raise RuntimeError(msg)
                else:
                    print("WARN:", msg)

    print("Saved H5:", out_h5)
    return out_h5


# ============================================================
# 2) Convenience: build ONE chosen split with chosen params
# ============================================================
# Назначение: формирует результирующую структуру данных или объект обработки.
def build_one_h5_B278(
    split: str,
    cache_dir: str = CACHE_DIR,
    seed: int = 42,
    keep_frac: float = 1.0,
    stride: int = 1,
    chunk_center_rows: int = 64,
    device: str | None = None,
    enable_resize: bool = True,
    resize_to: tuple[int,int] = (224,224),
    overwrite: bool = False,
    strict: bool = True,
):
    """
    Формирует результирующую структуру данных или объект обработки.

    Параметры
    ---------
    split :
        Параметр используется в соответствии с назначением функции.
    cache_dir :
        Параметр используется в соответствии с назначением функции.
    seed :
        Параметр используется в соответствии с назначением функции.
    keep_frac :
        Параметр используется в соответствии с назначением функции.
    stride :
        Параметр используется в соответствии с назначением функции.
    chunk_center_rows :
        Параметр используется в соответствии с назначением функции.
    device :
        Параметр используется в соответствии с назначением функции.
    enable_resize :
        Параметр используется в соответствии с назначением функции.
    resize_to :
        Параметр используется в соответствии с назначением функции.
    overwrite :
        Параметр используется в соответствии с назначением функции.
    strict :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `build_one_h5_B278`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = build_one_h5_B278(split='train', cache_dir='путь/к/ресурсу', seed=42)
    """
    out_h5 = os.path.join(
        cache_dir,
        f"cache_B278_{split}_stride{stride}_keep{keep_frac:.3f}_K{HORIZON_K}_t{HORIZON_T:.2f}"
        f"{'_resize224' if enable_resize else ''}.h5"
    )
    return build_h5_cache_B_split_gpu(
        split_name=split,
        out_h5=out_h5,
        keep_frac=keep_frac,
        stride=stride,
        seed=seed,
        chunk_center_rows=chunk_center_rows,
        device=device,
        enable_resize=enable_resize,
        resize_to=resize_to,
        save_masks_png=False,
        overwrite=overwrite,
        strict=strict,
    )

# ============================================================
# 3) CHECK: validate an existing h5 (fast, chunked)
# ============================================================
# Назначение: проверяет корректность структуры данных и возвращает сводную информацию.
def check_h5(
    path: str,
    max_rows_to_scan: int | None = 200_000,  # None -> scan all
    scan_chunk: int = 50_000,
    check_finite_X: bool = True,
    check_offsets: bool = True,
    verbose: bool = True,
):
    """
    Проверяет корректность структуры данных и возвращает сводную информацию.

    Параметры
    ---------
    path :
        Параметр используется в соответствии с назначением функции.
    max_rows_to_scan :
        Параметр используется в соответствии с назначением функции.
    scan_chunk :
        Параметр используется в соответствии с назначением функции.
    check_finite_X :
        Параметр используется в соответствии с назначением функции.
    check_offsets :
        Параметр используется в соответствии с назначением функции.
    verbose :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `check_h5`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = check_h5(path='путь/к/ресурсу', max_rows_to_scan=..., scan_chunk=...)
    """
    assert os.path.exists(path), f"File not found: {path}"

    out = {}
    with h5py.File(path, "r") as h5:
        # required datasets
        required = [
            "meta/file_name","meta/crop_y","meta/orig_h","meta/orig_w","meta/crop_h","meta/crop_w",
            "meta/was_resized","meta/resize_h","meta/resize_w",
            "samples/X","samples/img_idx","samples/cy","samples/cx","samples/y_fire","samples/y_smoke",
        ]
        for r in required:
            if r not in h5:
                raise RuntimeError(f"Missing dataset: {r}")

        # sample sizes
        n = int(h5["samples/X"].shape[0])
        d = int(h5["samples/X"].shape[1])
        out["n_samples"] = n
        out["d"] = d

        # check other sample arrays length
        for k in ["img_idx","cy","cx","y_fire","y_smoke"]:
            nk = int(h5[f"samples/{k}"].shape[0])
            if nk != n:
                raise RuntimeError(f"Length mismatch: samples/{k}={nk} vs samples/X={n}")

        n_imgs = int(h5["meta/file_name"].shape[0])
        out["n_images"] = n_imgs

        # quick checks on labels
        y_fire = h5["samples/y_fire"]
        y_smoke = h5["samples/y_smoke"]
        # sample few blocks to compute counts
        def _count_ones(dset):
            total = 0
            ones = 0
            step = min(200_000, n)  # cap reads
            for i0 in range(0, step, 50_000):
                i1 = min(step, i0 + 50_000)
                arr = dset[i0:i1]
                ones += int((arr > 0).sum())
                total += int(arr.size)
            return ones, total
        ones_fire, tot_fire = _count_ones(y_fire)
        ones_smoke, tot_smoke = _count_ones(y_smoke)
        out["fire_pos_rate_sampled"] = ones_fire / max(tot_fire, 1)
        out["smoke_pos_rate_sampled"] = ones_smoke / max(tot_smoke, 1)

        # img_idx bounds
        img_idx = h5["samples/img_idx"]
        # scan a limited portion for speed
        scanN = n if max_rows_to_scan is None else min(n, int(max_rows_to_scan))
        min_idx = +10**18
        max_idx = -10**18

        min_cy = +10**18
        min_cx = +10**18

        # optional finiteness scan
        n_bad = 0
        first_bad = None

        for i0 in range(0, scanN, scan_chunk):
            i1 = min(scanN, i0 + scan_chunk)
            idx_block = img_idx[i0:i1]
            min_idx = min(min_idx, int(idx_block.min()))
            max_idx = max(max_idx, int(idx_block.max()))

            cyb = h5["samples/cy"][i0:i1]
            cxb = h5["samples/cx"][i0:i1]
            min_cy = min(min_cy, int(cyb.min()))
            min_cx = min(min_cx, int(cxb.min()))

            if check_finite_X:
                Xb = h5["samples/X"][i0:i1, :]
                bad_mask = ~np.isfinite(Xb)
                if bad_mask.any():
                    n_bad += int(bad_mask.sum())
                    if first_bad is None:
                        rr, cc = np.argwhere(bad_mask)[0]
                        first_bad = (i0 + int(rr), int(cc), float(Xb[int(rr), int(cc)]))

        if min_idx < 0 or max_idx >= n_imgs:
            raise RuntimeError(f"img_idx out of range: min={min_idx}, max={max_idx}, n_images={n_imgs}")

        if min_cy < 0 or min_cx < 0:
            raise RuntimeError(f"Negative coordinates found: min_cy={min_cy}, min_cx={min_cx}")

        out["img_idx_min"] = int(min_idx)
        out["img_idx_max"] = int(max_idx)
        out["min_cy"] = int(min_cy)
        out["min_cx"] = int(min_cx)

        if check_finite_X:
            out["nonfinite_count_in_scanned"] = int(n_bad)
            out["first_nonfinite"] = first_bad
            if n_bad > 0:
                raise RuntimeError(f"Non-finite values found in X. First: {first_bad}, count in scanned={n_bad}")

        # offsets consistency (if present)
        if check_offsets:
            if ("meta/sample_start" in h5) and ("meta/sample_count" in h5):
                starts = h5["meta/sample_start"][:]
                counts = h5["meta/sample_count"][:]

                # quick: verify each image's interval has matching img_idx (sample a few images)
                # (full check can be expensive; we sample)
                rng = np.random.default_rng(0)
                sample_imgs = rng.choice(n_imgs, size=min(50, n_imgs), replace=False)
                for ii in sample_imgs:
                    s = int(starts[ii])
                    c = int(counts[ii])
                    if c == 0:
                        continue
                    # sanity bounds
                    if s < 0 or s + c > n:
                        raise RuntimeError(f"Bad offset range for img {ii}: start={s}, count={c}, n={n}")
                    idx_slice = img_idx[s:s+c]
                    if not np.all(idx_slice == ii):
                        # Not necessarily fatal if you later shuffle writing, but in our build it should match.
                        raise RuntimeError(f"Offset/img_idx mismatch for img {ii}: slice has idx range [{idx_slice.min()},{idx_slice.max()}]")
                out["offsets_ok_sampled"] = True
            else:
                out["offsets_ok_sampled"] = None

        if verbose:
            print("OK:", os.path.basename(path))
            print("  n_samples:", out["n_samples"], "d:", out["d"], "n_images:", out["n_images"])
            print("  img_idx:", out["img_idx_min"], "..", out["img_idx_max"])
            print("  pos_rate(sampled): fire=", f"{out['fire_pos_rate_sampled']:.4f}",
                  "smoke=", f"{out['smoke_pos_rate_sampled']:.4f}")
            if check_finite_X:
                print("  finite_X: OK (scanned rows:", scanN, ")")
            if check_offsets:
                print("  offsets(sampled):", out.get("offsets_ok_sampled"))

    return out
#создание маски из h5
# Назначение: формирует или обрабатывает бинарные маски объектов.
def build_mask_from_samples_fast(
    h5_path: str,
    img_idx: int,
    pred: np.ndarray,
    threshold: float = 0.5,
    to_space: str = "crop",   # "resize" | "crop" | "orig"
    fill_value: int = 0
):
    """
    Формирует или обрабатывает бинарные маски объектов.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.
    img_idx :
        Параметр используется в соответствии с назначением функции.
    pred :
        Параметр используется в соответствии с назначением функции.
    threshold :
        Параметр используется в соответствии с назначением функции.
    to_space :
        Параметр используется в соответствии с назначением функции.
    fill_value :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `build_mask_from_samples_fast`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = build_mask_from_samples_fast(h5_path='путь/к/ресурсу', img_idx=img_rgb, pred=...)
    """
    assert to_space in ("resize", "crop", "orig")

    with h5py.File(h5_path, "r") as h5:
        crop_y = int(h5["meta/crop_y"][img_idx])

        orig_h = int(h5["meta/orig_h"][img_idx]); orig_w = int(h5["meta/orig_w"][img_idx])
        crop_h = int(h5["meta/crop_h"][img_idx]); crop_w = int(h5["meta/crop_w"][img_idx])

        rz_h   = int(h5["meta/resize_h"][img_idx]); rz_w   = int(h5["meta/resize_w"][img_idx])
        was_resized = int(h5["meta/was_resized"][img_idx])

        start = int(h5["meta/sample_start"][img_idx])
        count = int(h5["meta/sample_count"][img_idx])

        if count <= 0:
            # пустая маска нужного размера
            if to_space == "resize":
                return np.full((rz_h, rz_w), fill_value, dtype=np.uint8)
            if to_space == "crop":
                return np.full((crop_h, crop_w), fill_value, dtype=np.uint8)
            out = np.zeros((orig_h, orig_w), dtype=np.uint8)
            return out

        end = start + count

        cy = h5["samples/cy"][start:end].astype(np.int32)
        cx = h5["samples/cx"][start:end].astype(np.int32)

    pred = np.asarray(pred)
    if pred.shape[0] != count:
        raise ValueError(f"pred length {pred.shape[0]} != samples for img {count}")

    if pred.dtype != np.uint8:
        pred_bin = (pred >= threshold).astype(np.uint8)
    else:
        pred_bin = pred

    # 1) mask in resize-space (coords are in this space)
    mask_rz = np.full((rz_h, rz_w), fill_value, dtype=np.uint8)
    mask_rz[cy, cx] = pred_bin

    if to_space == "resize":
        return mask_rz

    # 2) resize -> crop
    if was_resized:
        mask_crop = cv2.resize(mask_rz, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_crop = mask_rz

    if to_space == "crop":
        return mask_crop

    # 3) crop -> orig
    mask_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)
    mask_orig[crop_y:crop_y+crop_h, :] = mask_crop
    return mask_orig
LAWS_SLICE = slice(0, 147)
STATS_SLICE = slice(147, 147+35)
HAR_SLICE = slice(182, 182+96)

DIM_A = 182
DIM_B = 278
DIM_S = 35

# Назначение: определяет структуру набора данных или доступ к нему.
class H5WindowsDatasetFast(Dataset):
    """
    Определяет структуру набора данных или доступ к нему.

    Параметры
    ---------
    Функция не принимает обязательных пользовательских параметров.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `H5WindowsDatasetFast`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> obj = H5WindowsDatasetFast()
    """
    # Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
    def __init__(self, h5_path: str, variant: str, task: str, return_fp16=True):
        """
        Выполняет вспомогательную операцию в вычислительном конвейере.

        Параметры
        ---------
        h5_path :
            Параметр используется в соответствии с назначением функции.
        variant :
            Параметр используется в соответствии с назначением функции.
        task :
            Параметр используется в соответствии с назначением функции.
        return_fp16 :
            Параметр используется в соответствии с назначением функции.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `__init__`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = __init__(h5_path='путь/к/ресурсу', variant=..., task=...)
        """
        assert variant in ("A", "B", "S")
        assert task in ("smoke", "fire")
        self.h5_path = h5_path
        self.variant = variant
        self.task = task
        self.return_fp16 = bool(return_fp16)
        self._h5 = None

        # длина = N samples
        with h5py.File(self.h5_path, "r") as h5:
            self.N = int(h5["samples/X"].shape[0])

    # Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
    def __len__(self):
        """
        Выполняет вспомогательную операцию в вычислительном конвейере.

        Параметры
        ---------
        Функция не принимает обязательных пользовательских параметров.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `__len__`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = __len__()
        """
        return self.N

    # Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
    def _ensure_open(self):
        """
        Выполняет вспомогательную операцию в вычислительном конвейере.

        Параметры
        ---------
        Функция не принимает обязательных пользовательских параметров.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `_ensure_open`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = _ensure_open()
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    # Назначение: возвращает вычисляемый объект или конфигурацию.
    def __getitem__(self, i):
        """
        Возвращает вычисляемый объект или конфигурацию.

        Параметры
        ---------
        i :
            Параметр используется в соответствии с назначением функции.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `__getitem__`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = __getitem__(i=...)
        """
        self._ensure_open()
        X = self._h5["samples/X"][i]  # (278,) float16

        if self.task == "smoke":
            y = self._h5["samples/y_smoke"][i]
        else:
            y = self._h5["samples/y_fire"][i]

        if self.variant == "B":
            x = X
        elif self.variant == "A":
            # Laws + Stats -> 182
            x = np.concatenate([X[LAWS_SLICE], X[STATS_SLICE]], axis=0)
        else:
            # Stats only -> 35
            x = X[STATS_SLICE]

        # torch
        xt = torch.from_numpy(x)  # float16 tensor on CPU
        if not self.return_fp16:
            xt = xt.float()

        yt = torch.tensor([float(y)], dtype=torch.float32)
        return xt, yt

    # Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
    def close(self):
        """
        Выполняет вспомогательную операцию в вычислительном конвейере.

        Параметры
        ---------
        Функция не принимает обязательных пользовательских параметров.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `close`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = close()
        """
        if self._h5 is not None:
            try: self._h5.close()
            except Exception: pass
            self._h5 = None


# Назначение: определяет структуру набора данных или доступ к нему.
class H5InferDatasetFast(Dataset):
    """
    Определяет структуру набора данных или доступ к нему.

    Параметры
    ---------
    Функция не принимает обязательных пользовательских параметров.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `H5InferDatasetFast`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> obj = H5InferDatasetFast()
    """
    # Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
    def __init__(self, h5_path: str, variant: str, return_fp16=True):
        """
        Выполняет вспомогательную операцию в вычислительном конвейере.

        Параметры
        ---------
        h5_path :
            Параметр используется в соответствии с назначением функции.
        variant :
            Параметр используется в соответствии с назначением функции.
        return_fp16 :
            Параметр используется в соответствии с назначением функции.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `__init__`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = __init__(h5_path='путь/к/ресурсу', variant=..., return_fp16=...)
        """
        assert variant in ("A","B","S")
        self.h5_path = h5_path
        self.variant = variant
        self.return_fp16 = bool(return_fp16)
        self._h5 = None

        with h5py.File(self.h5_path, "r") as h5:
            self.N = int(h5["samples/X"].shape[0])

    # Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
        """
        Выполняет вспомогательную операцию в вычислительном конвейере.

        Параметры
        ---------
        Функция не принимает обязательных пользовательских параметров.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `__len__`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = __len__()
        """
    def __len__(self): return self.N

    # Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
    def _ensure_open(self):
        """
        Выполняет вспомогательную операцию в вычислительном конвейере.

        Параметры
        ---------
        Функция не принимает обязательных пользовательских параметров.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `_ensure_open`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = _ensure_open()
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    # Назначение: возвращает вычисляемый объект или конфигурацию.
    def __getitem__(self, i):
        """
        Возвращает вычисляемый объект или конфигурацию.

        Параметры
        ---------
        i :
            Параметр используется в соответствии с назначением функции.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `__getitem__`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = __getitem__(i=...)
        """
        self._ensure_open()
        X = self._h5["samples/X"][i]
        img_idx = int(self._h5["samples/img_idx"][i])
        cy = int(self._h5["samples/cy"][i])
        cx = int(self._h5["samples/cx"][i])

        if self.variant == "B":
            x = X
        elif self.variant == "A":
            x = np.concatenate([X[LAWS_SLICE], X[STATS_SLICE]], axis=0)
        else:
            x = X[STATS_SLICE]

        xt = torch.from_numpy(x)
        if not self.return_fp16:
            xt = xt.float()

        return xt, torch.tensor(img_idx, dtype=torch.int32), torch.tensor(cy, dtype=torch.int32), torch.tensor(cx, dtype=torch.int32)

    # Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
    def close(self):
        """
        Выполняет вспомогательную операцию в вычислительном конвейере.

        Параметры
        ---------
        Функция не принимает обязательных пользовательских параметров.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `close`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = close()
        """
        if self._h5 is not None:
            try: self._h5.close()
            except Exception: pass
            self._h5 = None


# Назначение: загружает данные, модель или служебную информацию.
def make_loader_fast(ds, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True):
    """
    Загружает данные, модель или служебную информацию.

    Параметры
    ---------
    ds :
        Параметр используется в соответствии с назначением функции.
    batch_size :
        Параметр используется в соответствии с назначением функции.
    shuffle :
        Параметр используется в соответствии с назначением функции.
    num_workers :
        Параметр используется в соответствии с назначением функции.
    pin_memory :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `make_loader_fast`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = make_loader_fast(ds=..., batch_size=(224, 224), shuffle=...)
    """
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False
    )
# Назначение: загружает данные, модель или служебную информацию.
def load_h5_to_cuda(
    h5_path: str,
    y_key: str,                 # "y_smoke" or "y_fire"
    device: str = "cuda",
    x_dtype: torch.dtype = torch.float16,
    y_dtype: torch.dtype = torch.float32,
    max_rows: int | None = None,   # можно ограничить для быстрой сетки
    as_pm1: bool = True,
):
    """
    Загружает данные, модель или служебную информацию.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.
    y_key :
        Параметр используется в соответствии с назначением функции.
    device :
        Параметр используется в соответствии с назначением функции.
    x_dtype :
        Параметр используется в соответствии с назначением функции.
    y_dtype :
        Параметр используется в соответствии с назначением функции.
    max_rows :
        Параметр используется в соответствии с назначением функции.
    as_pm1 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `load_h5_to_cuda`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = load_h5_to_cuda(h5_path='путь/к/ресурсу', y_key=..., device='cuda')
    """
    with h5py.File(h5_path, "r") as h5:
        X = h5["samples/X"][:]  # (N,278) float16
        y = h5[f"samples/{y_key}"][:].astype(np.float32)  # 0/1
    if max_rows is not None and X.shape[0] > max_rows:
        X = X[:max_rows]
        y = y[:max_rows]
    if as_pm1:
        # 0/1 -> -1/+1
        y = y * 2.0 - 1.0

    X_t = torch.from_numpy(X).to(device=device, dtype=x_dtype, non_blocking=True)
    y01 = torch.from_numpy(y).to(device=device, dtype=y_dtype, non_blocking=True)

    return X_t, y01
@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def fit_standardizer_gpu(X: torch.Tensor, eps: float = 1e-6):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    X :
        Параметр используется в соответствии с назначением функции.
    eps :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `fit_standardizer_gpu`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = fit_standardizer_gpu(X=..., eps=...)
    """
    mu = X.float().mean(dim=0)
    var = (X.float() - mu).pow(2).mean(dim=0)
    sigma = torch.sqrt(var + eps)
    return mu, sigma

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def apply_standardizer_gpu(X: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, out_dtype=torch.float16):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    X :
        Параметр используется в соответствии с назначением функции.
    mu :
        Параметр используется в соответствии с назначением функции.
    sigma :
        Параметр используется в соответствии с назначением функции.
    out_dtype :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `apply_standardizer_gpu`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = apply_standardizer_gpu(X=..., mu=..., sigma=...)
    """
    Z = (X.float() - mu) / sigma
    return Z.to(dtype=out_dtype)
# Назначение: инкапсулирует прикладную логику и связанные параметры.
class SParabolaAct(nn.Module):
    """
    Инкапсулирует прикладную логику и связанные параметры.

    Параметры
    ---------
    Функция не принимает обязательных пользовательских параметров.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `SParabolaAct`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> obj = SParabolaAct()
    """
    # Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
    def __init__(self, p=1.0, beta=0.0, eps=1e-6):
        """
        Выполняет вспомогательную операцию в вычислительном конвейере.

        Параметры
        ---------
        p :
            Параметр используется в соответствии с назначением функции.
        beta :
            Параметр используется в соответствии с назначением функции.
        eps :
            Параметр используется в соответствии с назначением функции.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `__init__`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = __init__(p=..., beta=..., eps=...)
        """
        super().__init__()
        self.p = float(p)
        self.beta = float(beta)
        self.eps = float(eps)

    # Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
    def forward(self, s):
        # sqrt(2 p (|s| + eps)) чтобы не было бесконечной производной при s~0
        """
        Выполняет вспомогательную операцию в вычислительном конвейере.

        Параметры
        ---------
        s :
            Параметр используется в соответствии с назначением функции.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `forward`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = forward(s=...)
        """
        r = torch.sqrt(2.0 * self.p * (s.abs() + self.eps))
        return self.beta + torch.where(s >= 0, r, -r)

# Назначение: возвращает вычисляемый объект или конфигурацию.
def get_activation(name: str, p: float, beta: float):
    """
    Возвращает вычисляемый объект или конфигурацию.

    Параметры
    ---------
    name :
        Параметр используется в соответствии с назначением функции.
    p :
        Параметр используется в соответствии с назначением функции.
    beta :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `get_activation`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = get_activation(name=..., p=..., beta=...)
    """
    name = name.lower()
    if name in ["sparabola", "s-parabola", "s_parabola"]:
        return SParabolaAct(p=p, beta=beta, eps=1e-6)
    raise ValueError(name)
# Назначение: инкапсулирует прикладную логику и связанные параметры.
class FastMLP(nn.Module):
    # Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
    """
    Инкапсулирует прикладную логику и связанные параметры.

    Параметры
    ---------
    Функция не принимает обязательных пользовательских параметров.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `FastMLP`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> obj = FastMLP()
    """
    def __init__(self, input_dim, layer_sizes, activation="sparabola", p=1.0, beta=0.0):
        """
        Выполняет вспомогательную операцию в вычислительном конвейере.

        Параметры
        ---------
        input_dim :
            Параметр используется в соответствии с назначением функции.
        layer_sizes :
            Параметр используется в соответствии с назначением функции.
        activation :
            Параметр используется в соответствии с назначением функции.
        p :
            Параметр используется в соответствии с назначением функции.
        beta :
            Параметр используется в соответствии с назначением функции.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `__init__`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = __init__(input_dim=..., layer_sizes=(224, 224), activation=...)
        """
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in layer_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(get_activation(activation, p=p, beta=beta))
            prev = hidden
        layers.append(nn.Linear(prev, 1))  # logits
        self.net = nn.Sequential(*layers)

    # Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
    def forward(self, x):
        """
        Выполняет вспомогательную операцию в вычислительном конвейере.

        Параметры
        ---------
        x :
            Параметр используется в соответствии с назначением функции.

        Возвращает
        ----------
        Результат вычислений, формируемый объектом `forward`.

        Примечания
        ----------
        Документационная строка добавлена для упрощения сопровождения, повторного
        использования кода и включения файла в состав отчётной документации.

        Пример использования
        -------------------
        >>> result = forward(x=...)
        """
        return self.net(x)
@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def pr_from_outputs(out, y_pm1, thr_out=0.0):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    out :
        Параметр используется в соответствии с назначением функции.
    y_pm1 :
        Параметр используется в соответствии с назначением функции.
    thr_out :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `pr_from_outputs`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = pr_from_outputs(out=..., y_pm1=..., thr_out=0.5)
    """
    if out.ndim == 1: out = out[:, None]
    if y_pm1.ndim == 1: y_pm1 = y_pm1[:, None]

    pred = out >= thr_out
    yt   = y_pm1 > 0

    tp = (pred & yt).sum().item()
    fp = (pred & (~yt)).sum().item()
    fn = ((~pred) & yt).sum().item()

    P = tp / (tp + fp + 1e-12)
    R = tp / (tp + fn + 1e-12)
    return P, R

# Назначение: выполняет обучение модели.
def train_fastmlp_mae_fixed_thr(
    Xtr, ytr_pm1, Xva, yva_pm1,
    input_dim,
    hidden=[182,90,45,10],
    p=1.0, beta=0.5,
    lr=3e-4, epochs=20, bs=32768,
    thr_out=0.0, min_recall=0.2,
    device="cuda",
    use_huber=False, huber_delta=1.0,
    grad_clip=5.0,
):
    """
    Выполняет обучение модели.

    Параметры
    ---------
    Xtr :
        Параметр используется в соответствии с назначением функции.
    ytr_pm1 :
        Параметр используется в соответствии с назначением функции.
    Xva :
        Параметр используется в соответствии с назначением функции.
    yva_pm1 :
        Параметр используется в соответствии с назначением функции.
    input_dim :
        Параметр используется в соответствии с назначением функции.
    hidden :
        Параметр используется в соответствии с назначением функции.
    p :
        Параметр используется в соответствии с назначением функции.
    beta :
        Параметр используется в соответствии с назначением функции.
    lr :
        Параметр используется в соответствии с назначением функции.
    epochs :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.
    thr_out :
        Параметр используется в соответствии с назначением функции.
    min_recall :
        Параметр используется в соответствии с назначением функции.
    device :
        Параметр используется в соответствии с назначением функции.
    use_huber :
        Параметр используется в соответствии с назначением функции.
    huber_delta :
        Параметр используется в соответствии с назначением функции.
    grad_clip :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `train_fastmlp_mae_fixed_thr`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = train_fastmlp_mae_fixed_thr(Xtr=..., ytr_pm1=..., Xva=...)
    """
    # y: (-1/+1) float (N,1)
    def prep_y_pm1(y):
        if y.ndim == 1: y = y[:, None]
        y = y.to(device=device, dtype=torch.float32)
        # если вдруг пришло 0/1 — переведём
        if y.min().item() >= 0.0 and y.max().item() <= 1.0:
            y = y * 2.0 - 1.0
        # защита
        y = torch.where(y >= 0.0, torch.ones_like(y), -torch.ones_like(y))
        return y

    Xtr = Xtr.to(device)
    Xva = Xva.to(device)
    ytr = prep_y_pm1(ytr_pm1)
    yva = prep_y_pm1(yva_pm1)

    model = FastMLP(input_dim, hidden, activation="sparabola", p=p, beta=beta).to(device)

    # Loss: MAE или SmoothL1
    if use_huber:
        crit = nn.SmoothL1Loss(beta=huber_delta)  # PyTorch 2.x: параметр beta
    else:
        crit = nn.L1Loss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    Ntr = Xtr.shape[0]
    Nv  = Xva.shape[0]
    bs  = int(bs)

    best = None

    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(Ntr, device=device)

        loss_sum = 0.0
        seen = 0

        for i in range(0, Ntr, bs):
            idx = perm[i:i+bs]
            xb = Xtr[idx]
            yb = ytr[idx]

            opt.zero_grad(set_to_none=True)
            out = model(xb)          # (B,1) linear output
            loss = crit(out, yb)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            bsz = xb.shape[0]
            loss_sum += float(loss.item()) * bsz
            seen += bsz

        tr_loss = loss_sum / max(seen, 1)

        # ---- eval ----
        model.eval()
        with torch.no_grad():
            # val outputs (chunked)
            outs = []
            for j in range(0, Nv, bs):
                outs.append(model(Xva[j:j+bs]))
            out_va = torch.cat(outs, dim=0)

            Pva, Rva = pr_from_outputs(out_va, yva, thr_out=thr_out)

            # val_err как MAE на валидации (аналогично статье)
            val_err = torch.abs(yva - out_va).mean().item()

        print(f"ep {ep:03d} | loss_tr={tr_loss:.4f} | val_err={val_err:.4f} | Pva={Pva:.4f} Rva={Rva:.4f} | thr_out={thr_out:.3f}")

        # критерий: max Precision при recall>=min_recall
        if Rva >= min_recall:
            score = Pva
            if (best is None) or (score > best["score"]):
                best = {
                    "score": score, "ep": ep, "P": Pva, "R": Rva,
                    "state": {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                }

    if best is not None:
        model.load_state_dict(best["state"])
        print("BEST:", {k: best[k] for k in ["score","ep","P","R"]})
    else:
        print("No model met min_recall:", min_recall)

    return model, best
@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def metrics_from_outputs(out, y_pm1, thr_out=0.0):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    out :
        Параметр используется в соответствии с назначением функции.
    y_pm1 :
        Параметр используется в соответствии с назначением функции.
    thr_out :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `metrics_from_outputs`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = metrics_from_outputs(out=..., y_pm1=..., thr_out=0.5)
    """
    if out.ndim == 1: out = out[:, None]
    if y_pm1.ndim == 1: y_pm1 = y_pm1[:, None]

    pred = out >= thr_out
    yt   = y_pm1 > 0

    tp = (pred & yt).sum().item()
    fp = (pred & (~yt)).sum().item()
    fn = ((~pred) & yt).sum().item()
    tn = ((~pred) & (~yt)).sum().item()

    P = tp / (tp + fp + 1e-12)
    R = tp / (tp + fn + 1e-12)
    F1 = (2*P*R) / (P+R+1e-12)

    out_mean = float(out.mean().item())
    out_std  = float(out.std(unbiased=False).item())
    return P, R, F1, tp, fp, fn, tn, out_mean, out_std

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def pick_best_threshold_out(out, y_pm1, thr_grid, min_recall=0.0):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    out :
        Параметр используется в соответствии с назначением функции.
    y_pm1 :
        Параметр используется в соответствии с назначением функции.
    thr_grid :
        Параметр используется в соответствии с назначением функции.
    min_recall :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `pick_best_threshold_out`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = pick_best_threshold_out(out=..., y_pm1=..., thr_grid=0.5)
    """
    best = None
    for thr in thr_grid:
        P,R,F1,tp,fp,fn,tn,m,s = metrics_from_outputs(out, y_pm1, thr_out=float(thr))
        if R < min_recall:
            continue
        key = (P, F1, R)  # primary Precision
        if (best is None) or (key > best["key"]):
            best = dict(
                thr_out=float(thr), P=P, R=R, F1=F1,
                tp=tp, fp=fp, fn=fn, tn=tn,
                out_mean=m, out_std=s,
                key=key
            )
    return best

# Назначение: выполняет обучение модели.
def train_one_cfg_mae_thr(
    Xtr_n, ytr, Xva_n, yva,
    p, beta,
    epochs=10, batch_size=32768,
    lr=3e-4,
    min_recall=0.2,
    thr_grid=None,
    seed=42,
    use_huber=False,
    huber_delta=1.0,
    grad_clip=5.0,
    layer_sizes = [182,90,45,10]
):
    """
    Выполняет обучение модели.

    Параметры
    ---------
    Xtr_n :
        Параметр используется в соответствии с назначением функции.
    ytr :
        Параметр используется в соответствии с назначением функции.
    Xva_n :
        Параметр используется в соответствии с назначением функции.
    yva :
        Параметр используется в соответствии с назначением функции.
    p :
        Параметр используется в соответствии с назначением функции.
    beta :
        Параметр используется в соответствии с назначением функции.
    epochs :
        Параметр используется в соответствии с назначением функции.
    batch_size :
        Параметр используется в соответствии с назначением функции.
    lr :
        Параметр используется в соответствии с назначением функции.
    min_recall :
        Параметр используется в соответствии с назначением функции.
    thr_grid :
        Параметр используется в соответствии с назначением функции.
    seed :
        Параметр используется в соответствии с назначением функции.
    use_huber :
        Параметр используется в соответствии с назначением функции.
    huber_delta :
        Параметр используется в соответствии с назначением функции.
    grad_clip :
        Параметр используется в соответствии с назначением функции.
    layer_sizes :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `train_one_cfg_mae_thr`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = train_one_cfg_mae_thr(Xtr_n=..., ytr=..., Xva_n=...)
    """
    device = Xtr_n.device
    torch.manual_seed(seed)

    def prep_y_pm1(y):
        if y.ndim == 1: y = y[:,None]
        y = y.to(device=device, dtype=torch.float32)
        # 0/1 -> -1/+1
        if y.min().item() >= 0.0 and y.max().item() <= 1.0:
            y = y * 2.0 - 1.0
        y = torch.where(y >= 0.0, torch.ones_like(y), -torch.ones_like(y))
        return y

    ytr_pm1 = prep_y_pm1(ytr)
    yva_pm1 = prep_y_pm1(yva)

    model = FastMLP(input_dim=Xtr_n.shape[1], layer_sizes=layer_sizes,
                    activation="sparabola", p=p, beta=beta).to(device)

    # MAE / Huber
    if use_huber:
        crit = nn.SmoothL1Loss(beta=huber_delta)
    else:
        crit = nn.L1Loss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    Ntr = Xtr_n.shape[0]
    Nv  = Xva_n.shape[0]
    bs  = int(batch_size)

    # ---- train epochs ----
    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(Ntr, device=device)
        loss_sum = 0.0
        seen = 0

        for i in range(0, Ntr, bs):
            idx = perm[i:i+bs]
            xb = Xtr_n[idx]
            yb = ytr_pm1[idx]

            opt.zero_grad(set_to_none=True)
            out = model(xb.float()) # Ensure input is float32
            loss = crit(out, yb)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            bsz = xb.shape[0]
            loss_sum += float(loss.item()) * bsz
            seen += bsz

        tr_loss = loss_sum / max(seen, 1)

        # быстрый мониторинг: MAE на val (без подбора порога)
        model.eval()
        with torch.no_grad():
            outs = []
            for j in range(0, Nv, bs):
                outs.append(model(Xva_n[j:j+bs].float())) # Ensure input is float32
            out_va = torch.cat(outs, dim=0)
            val_err = torch.abs(yva_pm1 - out_va).mean().item()

        print(f"ep {ep:03d} | loss_tr={tr_loss:.4f} | val_err={val_err:.4f}")

    # ---- val outputs once (for thr picking) ----
    model.eval()
    with torch.no_grad():
        outs = []
        for j in range(0, Nv, bs):
            outs.append(model(Xva_n[j:j+bs].float())) # Ensure input is float32
        out_va = torch.cat(outs, dim=0)

    if thr_grid is None:
        # разумная сетка по выходу: центр 0, чуть шире
        thr_grid = torch.linspace(-1.0, 1.0, 81).cpu().tolist()

    best_thr = pick_best_threshold_out(out_va, yva_pm1, thr_grid, min_recall=min_recall)
    if best_thr is None:
        best_thr = pick_best_threshold_out(out_va, yva_pm1, thr_grid, min_recall=0.0)

    out = {
        "p": float(p),
        "beta": float(beta),
        "lr": float(lr),
        "epochs": int(epochs),
        "batch_size": int(bs),
        "thr_out": best_thr["thr_out"],
        "P": best_thr["P"],
        "R": best_thr["R"],
        "F1": best_thr["F1"],
        "tp": best_thr["tp"],
        "fp": best_thr["fp"],
        "fn": best_thr["fn"],
        "tn": best_thr["tn"],
        "out_mean": best_thr["out_mean"],
        "out_std": best_thr["out_std"],
        "model_state": {k: v.detach().cpu() for k,v in model.state_dict().items()},
        "loss_kind": "SmoothL1" if use_huber else "L1",
        "huber_delta": float(huber_delta),
    }
    return out
@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def make_thr_grid_from_outputs(
    out_va: torch.Tensor,
    n: int = 81,
    k: float = 3.0,
    include_zero: bool = True,
    clip_abs: float | None = None,
):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    out_va :
        Параметр используется в соответствии с назначением функции.
    n :
        Параметр используется в соответствии с назначением функции.
    k :
        Параметр используется в соответствии с назначением функции.
    include_zero :
        Параметр используется в соответствии с назначением функции.
    clip_abs :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `make_thr_grid_from_outputs`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = make_thr_grid_from_outputs(out_va=..., n=..., k=...)
    """
    if out_va.ndim == 2:
        out_va = out_va[:, 0]
    out_va = out_va.detach().float()

    m = out_va.mean().item()
    s = out_va.std(unbiased=False).item()
    lo = m - k * s
    hi = m + k * s

    # страховка от "почти константа"
    if not np.isfinite(s) or s < 1e-6:
        lo, hi = m - 1.0, m + 1.0

    if clip_abs is not None:
        lo = max(lo, -clip_abs)
        hi = min(hi, +clip_abs)

    grid = torch.linspace(lo, hi, n, device="cpu").tolist()
    if include_zero and (0.0 < lo or 0.0 > hi):
        grid.append(0.0)
    grid = sorted(set(float(x) for x in grid))
    return grid

# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def make_thr_grid_two_stage(out_va, n_wide=81, n_dense=81, dense_half=0.5, k=3.0):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    out_va :
        Параметр используется в соответствии с назначением функции.
    n_wide :
        Параметр используется в соответствии с назначением функции.
    n_dense :
        Параметр используется в соответствии с назначением функции.
    dense_half :
        Параметр используется в соответствии с назначением функции.
    k :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `make_thr_grid_two_stage`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = make_thr_grid_two_stage(out_va=..., n_wide=..., n_dense=...)
    """
    g1 = make_thr_grid_from_outputs(out_va, n=n_wide, k=k, include_zero=True)
    g2 = torch.linspace(-dense_half, dense_half, n_dense).tolist()
    grid = sorted(set([float(x) for x in (g1 + g2 + [0.0])]))
    return grid
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def _as_col(y: torch.Tensor) -> torch.Tensor:
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    y :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_as_col`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _as_col(y=...)
    """
    if y.ndim == 1:
        return y.unsqueeze(1)
    if y.ndim == 2 and y.shape[1] == 1:
        return y
    return y.reshape(-1, 1)

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def metrics_pm1_from_out(out, y_pm1, thr_out=0.0):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    out :
        Параметр используется в соответствии с назначением функции.
    y_pm1 :
        Параметр используется в соответствии с назначением функции.
    thr_out :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `metrics_pm1_from_out`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = metrics_pm1_from_out(out=..., y_pm1=..., thr_out=0.5)
    """
    out2 = _as_col(out).float()      # (N,1)
    y2   = _as_col(y_pm1).float()    # (N,1)

    pred_pos = out2 >= float(thr_out)
    true_pos = y2 > 0

    tp = (pred_pos &  true_pos).sum().item()
    fp = (pred_pos & ~true_pos).sum().item()
    fn = (~pred_pos &  true_pos).sum().item()
    tn = (~pred_pos & ~true_pos).sum().item()

    P = tp / (tp + fp + 1e-12)
    R = tp / (tp + fn + 1e-12)
    F1 = (2*P*R) / (P + R + 1e-12)

    # mean/std по выходу
    out_flat = out2[:, 0]
    return P, R, F1, tp, fp, fn, tn, float(out_flat.mean().item()), float(out_flat.std(unbiased=False).item())

@torch.no_grad()
# Назначение: выполняет оценку модели.
def eval_out_chunked(model, X, bs=32768):
    """
    Выполняет оценку модели.

    Параметры
    ---------
    model :
        Параметр используется в соответствии с назначением функции.
    X :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `eval_out_chunked`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = eval_out_chunked(model=..., X=..., bs=...)
    """
    model.eval()
    outs = []
    N = X.shape[0]
    for i in range(0, N, bs):
        outs.append(model(X[i:i+bs].float()))
    return torch.cat(outs, dim=0)  # (N,1)

# Назначение: проверяет корректность структуры данных и возвращает сводную информацию.
def save_checkpoint(path, model, extra: dict):
    """
    Проверяет корректность структуры данных и возвращает сводную информацию.

    Параметры
    ---------
    path :
        Параметр используется в соответствии с назначением функции.
    model :
        Параметр используется в соответствии с назначением функции.
    extra :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `save_checkpoint`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = save_checkpoint(path='путь/к/ресурсу', model=..., extra=...)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "extra": extra}, path)
@torch.no_grad()
# Назначение: загружает данные, модель или служебную информацию.
def load_ckpt_to_model(model, path, device):
    """
    Загружает данные, модель или служебную информацию.

    Параметры
    ---------
    model :
        Параметр используется в соответствии с назначением функции.
    path :
        Параметр используется в соответствии с назначением функции.
    device :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `load_ckpt_to_model`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = load_ckpt_to_model(model=..., path='путь/к/ресурсу', device='cuda')
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt.get("extra", {})

@torch.no_grad()
# Назначение: выполняет получение предсказаний модели.
def predict_pm1_out_chunked(model, X, bs=32768):
    # returns out (N,1) float32 on device
    """
    Выполняет получение предсказаний модели.

    Параметры
    ---------
    model :
        Параметр используется в соответствии с назначением функции.
    X :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `predict_pm1_out_chunked`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = predict_pm1_out_chunked(model=..., X=..., bs=...)
    """
    model.eval()
    outs = []
    for i in range(0, X.shape[0], bs):
        outs.append(model(X[i:i+bs].float()))
    return torch.cat(outs, dim=0)

@torch.no_grad()
# Назначение: формирует или обрабатывает бинарные маски объектов.
def hard_mask_from_two_models(
    ins1, thr1, X1,  # INS-1 on 182
    ins2, thr2, X2,  # INS-2 on 278
    y_pm1,
    bs=32768
):
    # y_pm1: (N,) or (N,1) in {-1,+1}
    """
    Формирует или обрабатывает бинарные маски объектов.

    Параметры
    ---------
    ins1 :
        Параметр используется в соответствии с назначением функции.
    thr1 :
        Параметр используется в соответствии с назначением функции.
    X1 :
        Параметр используется в соответствии с назначением функции.
    ins2 :
        Параметр используется в соответствии с назначением функции.
    thr2 :
        Параметр используется в соответствии с назначением функции.
    X2 :
        Параметр используется в соответствии с назначением функции.
    y_pm1 :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `hard_mask_from_two_models`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = hard_mask_from_two_models(ins1=..., thr1=0.5, X1=...)
    """
    y2 = y_pm1 if y_pm1.ndim == 2 else y_pm1[:,None]
    y_pos = (y2 > 0)

    out1 = predict_pm1_out_chunked(ins1, X1, bs=bs)  # (N,1)
    out2 = predict_pm1_out_chunked(ins2, X2, bs=bs)  # (N,1)

    pred1_pos = out1 >= float(thr1)
    pred2_pos = out2 >= float(thr2)

    err1 = (pred1_pos != y_pos)
    err2 = (pred2_pos != y_pos)

    hard = (err1 | err2)[:, 0]   # (N,)
    return hard, out1, out2
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def _as_col(y: torch.Tensor) -> torch.Tensor:
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    y :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_as_col`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _as_col(y=...)
    """
    if y.ndim == 1: return y.unsqueeze(1)
    if y.ndim == 2 and y.shape[1] == 1: return y
    return y.reshape(-1,1)

# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def weighted_mae(out, y, w):
    # out,y: (B,1); w: (B,1)
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    out :
        Параметр используется в соответствии с назначением функции.
    y :
        Параметр используется в соответствии с назначением функции.
    w :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `weighted_mae`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = weighted_mae(out=..., y=..., w=...)
    """
    return (w * (out - y).abs()).sum() / (w.sum() + 1e-12)

@torch.no_grad()
# Назначение: выполняет оценку модели.
def eval_mae_chunked(model, X, y_pm1, bs=32768):
    """
    Выполняет оценку модели.

    Параметры
    ---------
    model :
        Параметр используется в соответствии с назначением функции.
    X :
        Параметр используется в соответствии с назначением функции.
    y_pm1 :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `eval_mae_chunked`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = eval_mae_chunked(model=..., X=..., y_pm1=...)
    """
    model.eval()
    y2 = _as_col(y_pm1).float()
    s = 0.0
    n = 0
    for i in range(0, X.shape[0], bs):
        out = model(X[i:i+bs].float())
        yb  = y2[i:i+bs]
        s += (out - yb).abs().sum().item()
        n += yb.numel()
    return s / max(n, 1)

# Назначение: выполняет обучение модели.
def train_fastmlp_weighted_mae(
    Xtr, ytr_pm1, hard_mask,         # train
    Xva, yva_pm1,                    # valid
    input_dim, hidden,
    p, beta,
    lr=3e-4, epochs=20, bs=32768,
    w_hard=5.0,
    clip_norm=5.0,
    device="cuda",
    save_path=None,
):
    """
    Выполняет обучение модели.

    Параметры
    ---------
    Xtr :
        Параметр используется в соответствии с назначением функции.
    ytr_pm1 :
        Параметр используется в соответствии с назначением функции.
    hard_mask :
        Параметр используется в соответствии с назначением функции.
    Xva :
        Параметр используется в соответствии с назначением функции.
    yva_pm1 :
        Параметр используется в соответствии с назначением функции.
    input_dim :
        Параметр используется в соответствии с назначением функции.
    hidden :
        Параметр используется в соответствии с назначением функции.
    p :
        Параметр используется в соответствии с назначением функции.
    beta :
        Параметр используется в соответствии с назначением функции.
    lr :
        Параметр используется в соответствии с назначением функции.
    epochs :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.
    w_hard :
        Параметр используется в соответствии с назначением функции.
    clip_norm :
        Параметр используется в соответствии с назначением функции.
    device :
        Параметр используется в соответствии с назначением функции.
    save_path :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `train_fastmlp_weighted_mae`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = train_fastmlp_weighted_mae(Xtr=..., ytr_pm1=..., hard_mask=mask)
    """
    model = FastMLP(
        input_dim=input_dim,
        layer_sizes=hidden,
        activation="sparabola",
        p=p, beta=beta
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    ytr2 = _as_col(ytr_pm1).float()
    yva2 = _as_col(yva_pm1).float()

    hard = hard_mask.to(device=device)
    best = None

    N = Xtr.shape[0]
    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(N, device=device)

        loss_sum = 0.0
        n_sum = 0

        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            xb = Xtr[idx].float()
            yb = ytr2[idx]

            hb = hard[idx].float().unsqueeze(1)
            w  = 1.0 + (w_hard - 1.0) * hb

            opt.zero_grad(set_to_none=True)
            out = model(xb)  # (B,1)
            loss = weighted_mae(out, yb, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            opt.step()

            loss_sum += loss.item() * xb.size(0)
            n_sum += xb.size(0)

        tr_loss = loss_sum / max(n_sum, 1)

        va_err = eval_mae_chunked(model, Xva, yva2, bs=bs)

        print(f"ep {ep:02d} | loss_tr={tr_loss:.4f} | val_err={va_err:.4f}")

        key = (-va_err,)
        if (best is None) or (key > best["key"]):
            best = {"key": key, "epoch": ep, "val_err": float(va_err)}
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "extra": best}, save_path)

    return model, best
@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def pick_best_thr_out_from_out(
    out_va, yva_pm1,
    thr_grid,
    min_recall=0.0,
    mode="precision",  # "precision" or "f1"
):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    out_va :
        Параметр используется в соответствии с назначением функции.
    yva_pm1 :
        Параметр используется в соответствии с назначением функции.
    thr_grid :
        Параметр используется в соответствии с назначением функции.
    min_recall :
        Параметр используется в соответствии с назначением функции.
    mode :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `pick_best_thr_out_from_out`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = pick_best_thr_out_from_out(out_va=..., yva_pm1=..., thr_grid=0.5)
    """
    if out_va.ndim == 2: out_va = out_va[:,0]
    if yva_pm1.ndim == 2: yva_pm1 = yva_pm1[:,0]
    out_va = out_va.float()
    yva_pm1 = yva_pm1.float()

    true_pos = (yva_pm1 > 0)

    best = None
    for thr in thr_grid:
        pred_pos = out_va >= float(thr)
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & (~true_pos)).sum().item()
        fn = ((~pred_pos) & true_pos).sum().item()
        tn = ((~pred_pos) & (~true_pos)).sum().item()

        P = tp / (tp + fp + 1e-12)
        R = tp / (tp + fn + 1e-12)
        F1 = (2*P*R) / (P + R + 1e-12)

        if R < min_recall:
            continue

        key = (P, F1, R) if mode == "precision" else (F1, P, R)
        if (best is None) or (key > best["key"]):
            best = dict(
                thr=float(thr), P=P, R=R, F1=F1,
                tp=tp, fp=fp, fn=fn, tn=tn,
                out_mean=float(out_va.mean().item()),
                out_std=float(out_va.std(unbiased=False).item()),
                key=key
            )
    return best

@torch.no_grad()
# Назначение: выполняет оценку модели.
def eval_out_chunked(model, X, bs=32768):
    """
    Выполняет оценку модели.

    Параметры
    ---------
    model :
        Параметр используется в соответствии с назначением функции.
    X :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `eval_out_chunked`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = eval_out_chunked(model=..., X=..., bs=...)
    """
    model.eval()
    outs = []
    for i in range(0, X.shape[0], bs):
        outs.append(model(X[i:i+bs].float()))
    return torch.cat(outs, dim=0)
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def grid_search_ins3_pbeta_thr(
    Xtr, ytr, hard_tr,
    Xva, yva,
    input_dim, hidden,
    p_grid, beta_grid,
    lr=3e-4, epochs=3, bs=32768,
    w_hard=7.0,
    min_recall=0.2,
    thr_grid=None,
    device="cuda",
):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    Xtr :
        Параметр используется в соответствии с назначением функции.
    ytr :
        Параметр используется в соответствии с назначением функции.
    hard_tr :
        Параметр используется в соответствии с назначением функции.
    Xva :
        Параметр используется в соответствии с назначением функции.
    yva :
        Параметр используется в соответствии с назначением функции.
    input_dim :
        Параметр используется в соответствии с назначением функции.
    hidden :
        Параметр используется в соответствии с назначением функции.
    p_grid :
        Параметр используется в соответствии с назначением функции.
    beta_grid :
        Параметр используется в соответствии с назначением функции.
    lr :
        Параметр используется в соответствии с назначением функции.
    epochs :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.
    w_hard :
        Параметр используется в соответствии с назначением функции.
    min_recall :
        Параметр используется в соответствии с назначением функции.
    thr_grid :
        Параметр используется в соответствии с назначением функции.
    device :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `grid_search_ins3_pbeta_thr`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = grid_search_ins3_pbeta_thr(Xtr=..., ytr=..., hard_tr=...)
    """
    if thr_grid is None:
        thr_grid = torch.linspace(-2.0, 2.0, 161).tolist()

    results = []
    best = None

    for p, beta in product(p_grid, beta_grid):
        model, _ = train_fastmlp_weighted_mae(
            Xtr=Xtr, ytr_pm1=ytr, hard_mask=hard_tr,
            Xva=Xva, yva_pm1=yva,
            input_dim=input_dim, hidden=hidden,
            p=p, beta=beta,
            lr=lr, epochs=epochs, bs=bs,
            w_hard=w_hard,
            clip_norm=5.0,
            device=device,
            save_path=None
        )

        out_va = eval_out_chunked(model, Xva, bs=bs)
        m = float(out_va.mean().item())
        s = float(out_va.std(unbiased=False).item())
        thr_grid2 = torch.linspace(m - 4*s, m + 4*s, 161).tolist()

        best_thr = pick_best_thr_out_from_out(out_va, yva, thr_grid2, min_recall=min_recall, mode="precision")
        if best_thr is None:
            best_thr = pick_best_thr_out_from_out(out_va, yva, thr_grid2, min_recall=0.0, mode="precision")

        row = {
            "p": float(p), "beta": float(beta),
            "thr": float(best_thr["thr"]),
            "P": best_thr["P"], "R": best_thr["R"], "F1": best_thr["F1"],
            "fp": best_thr["fp"], "fn": best_thr["fn"],
            "out_mean": best_thr["out_mean"], "out_std": best_thr["out_std"],
        }
        results.append(row)

        key = (row["P"], row["F1"], row["R"])
        if (best is None) or (key > best["key"]):
            best = {"key": key, **row}

        print(f"p={p:<4} beta={beta:<4} -> thr={row['thr']:+.3f} | P={row['P']:.4f} R={row['R']:.4f} F1={row['F1']:.4f} | fp={row['fp']} fn={row['fn']}")

    print("\nBEST:", best)
    return results, best

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def infer_out_chunked(model, X, bs=32768):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    model :
        Параметр используется в соответствии с назначением функции.
    X :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `infer_out_chunked`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = infer_out_chunked(model=..., X=..., bs=...)
    """
    model.eval()
    outs = []
    N = X.shape[0]
    for i in range(0, N, bs):
        outs.append(model(X[i:i+bs].float()))  # (B,1)
    return torch.cat(outs, dim=0)  # (N,1)

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def binary_metrics_pm1(out, y_pm1, thr_out=0.0):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    out :
        Параметр используется в соответствии с назначением функции.
    y_pm1 :
        Параметр используется в соответствии с назначением функции.
    thr_out :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `binary_metrics_pm1`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = binary_metrics_pm1(out=..., y_pm1=..., thr_out=0.5)
    """
    if out.ndim == 2: out = out[:, 0]
    if y_pm1.ndim == 2: y_pm1 = y_pm1[:, 0]
    out = out.float()
    y_pm1 = y_pm1.float()

    pred_pos = out >= float(thr_out)
    true_pos = y_pm1 > 0

    tp = (pred_pos & true_pos).sum().item()
    fp = (pred_pos & (~true_pos)).sum().item()
    fn = ((~pred_pos) & true_pos).sum().item()
    tn = ((~pred_pos) & (~true_pos)).sum().item()

    P = tp / (tp + fp + 1e-12)
    R = tp / (tp + fn + 1e-12)
    F1 = (2*P*R) / (P + R + 1e-12)

    # extra
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    tpr = R
    tnr = tn / (tn + fp + 1e-12)
    bal_acc = 0.5 * (tpr + tnr)

    return {
        "P": P, "R": R, "F1": F1,
        "acc": acc, "bal_acc": bal_acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "out_mean": float(out.mean().item()),
        "out_std": float(out.std(unbiased=False).item()),
    }

# ----------------------------
# 3-class from two binary predictions with fire priority
# labels: 0=non-fire, 1=smoke, 2=fire
# ----------------------------
@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def make_3class_pred(fire_pos: torch.Tensor, smoke_pos: torch.Tensor):
    # fire_pos, smoke_pos: bool (N,)
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    fire_pos :
        Параметр используется в соответствии с назначением функции.
    smoke_pos :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `make_3class_pred`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = make_3class_pred(fire_pos=..., smoke_pos=...)
    """
    y3 = torch.zeros_like(fire_pos, dtype=torch.int64)
    y3[smoke_pos] = 1
    y3[fire_pos]  = 2
    return y3

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def confusion_3x3(y_true3, y_pred3):
    # y_true3/y_pred3: int64 (N,)
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    y_true3 :
        Параметр используется в соответствии с назначением функции.
    y_pred3 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `confusion_3x3`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = confusion_3x3(y_true3=..., y_pred3=...)
    """
    cm = torch.zeros((3,3), dtype=torch.int64, device=y_true3.device)
    for t in range(3):
        for p in range(3):
            cm[t,p] = ((y_true3==t) & (y_pred3==p)).sum()
    return cm

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def prf_from_cm_3class(cm):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    cm :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `prf_from_cm_3class`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = prf_from_cm_3class(cm=...)
    """
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

# Назначение: вычисляет смещение по заданным параметрам.
def image_level_stats_from_offsets(
    h5, y3_pred_all, valid_mask_all=None, thresholds=(0.02,0.05,0.07,0.10)
):
    """
    Вычисляет смещение по заданным параметрам.

    Параметры
    ---------
    h5 :
        Параметр используется в соответствии с назначением функции.
    y3_pred_all :
        Параметр используется в соответствии с назначением функции.
    valid_mask_all :
        Параметр используется в соответствии с назначением функции.
    thresholds :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `image_level_stats_from_offsets`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = image_level_stats_from_offsets(h5=..., y3_pred_all=..., valid_mask_all=mask)
    """
    starts = h5["meta/sample_start"][:]
    counts = h5["meta/sample_count"][:]
    n_imgs = len(starts)

    thr_list = list(thresholds)
    det_counts = {t: 0 for t in thr_list}
    det_counts_fire = {t: 0 for t in thr_list}
    det_counts_smoke = {t: 0 for t in thr_list}

    ratios = []  # store per-image (hazard, fire, smoke, n)
    for i in range(n_imgs):
        s = int(starts[i]); c = int(counts[i])
        if c <= 0:
            ratios.append((0.0,0.0,0.0,0))
            continue
        sl = slice(s, s+c)
        y3 = y3_pred_all[sl]
        if valid_mask_all is not None:
            vm = valid_mask_all[sl]
            y3 = y3[vm]
        n = int(y3.numel())
        if n == 0:
            ratios.append((0.0,0.0,0.0,0))
            continue

        fire = (y3==2).sum().item()
        smoke = (y3==1).sum().item()
        hazard = fire + smoke

        fire_r = fire / n
        smoke_r = smoke / n
        haz_r = hazard / n
        ratios.append((haz_r, fire_r, smoke_r, n))

        for t in thr_list:
            if haz_r >= t: det_counts[t] += 1
            if fire_r >= t: det_counts_fire[t] += 1
            if smoke_r >= t: det_counts_smoke[t] += 1

    # summarize
    haz = np.array([r[0] for r in ratios], dtype=np.float64)
    fir = np.array([r[1] for r in ratios], dtype=np.float64)
    smo = np.array([r[2] for r in ratios], dtype=np.float64)

    rep = {
        "n_images": int(n_imgs),
        "hazard_ratio_mean": float(haz.mean()),
        "hazard_ratio_p90": float(np.quantile(haz, 0.90)),
        "fire_ratio_mean": float(fir.mean()),
        "smoke_ratio_mean": float(smo.mean()),
    }
    for t in thr_list:
        rep[f"det_hazard@{t:.4f}"] = det_counts[t] / max(n_imgs,1)
        rep[f"det_fire@{t:.4f}"]   = det_counts_fire[t] / max(n_imgs,1)
        rep[f"det_smoke@{t:.4f}"]  = det_counts_smoke[t] / max(n_imgs,1)
    return rep
# ----------------------------
# hazard (fire OR smoke) metrics
# hazard = 1 if (fire==1) or (smoke==1)
# ----------------------------
@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def hazard_metrics_global(
    fire_pos_pred: torch.Tensor,   # bool (N,)
    smoke_pos_pred: torch.Tensor,  # bool (N,)
    y_fire_pm1: torch.Tensor,      # {-1,+1} (N,) or (N,1)
    y_smoke_pm1: torch.Tensor,     # {-1,+1} (N,) or (N,1)
):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    fire_pos_pred :
        Параметр используется в соответствии с назначением функции.
    smoke_pos_pred :
        Параметр используется в соответствии с назначением функции.
    y_fire_pm1 :
        Параметр используется в соответствии с назначением функции.
    y_smoke_pm1 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `hazard_metrics_global`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = hazard_metrics_global(fire_pos_pred=..., smoke_pos_pred=..., y_fire_pm1=...)
    """
    if y_fire_pm1.ndim == 2: y_fire_pm1 = y_fire_pm1[:,0]
    if y_smoke_pm1.ndim == 2: y_smoke_pm1 = y_smoke_pm1[:,0]

    pred_h = fire_pos_pred | smoke_pos_pred
    true_h = (y_fire_pm1 > 0) | (y_smoke_pm1 > 0)

    tp = (pred_h & true_h).sum().item()
    fp = (pred_h & (~true_h)).sum().item()
    fn = ((~pred_h) & true_h).sum().item()
    tn = ((~pred_h) & (~true_h)).sum().item()

    P = tp / (tp + fp + 1e-12)
    R = tp / (tp + fn + 1e-12)
    F1 = (2*P*R) / (P + R + 1e-12)

    return {
        "haz_P_global": P,
        "haz_R_global": R,
        "haz_F1_global": F1,
        "haz_tp": tp, "haz_fp": fp, "haz_fn": fn, "haz_tn": tn,
    }


# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def hazard_metrics_mean_per_image(
    h5_path: str,
    fire_pos_pred: torch.Tensor,   # bool (N,)
    smoke_pos_pred: torch.Tensor,  # bool (N,)
    y_fire_pm1: torch.Tensor,
    y_smoke_pm1: torch.Tensor,
):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.
    fire_pos_pred :
        Параметр используется в соответствии с назначением функции.
    smoke_pos_pred :
        Параметр используется в соответствии с назначением функции.
    y_fire_pm1 :
        Параметр используется в соответствии с назначением функции.
    y_smoke_pm1 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `hazard_metrics_mean_per_image`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = hazard_metrics_mean_per_image(h5_path='путь/к/ресурсу', fire_pos_pred=..., smoke_pos_pred=...)
    """
    if y_fire_pm1.ndim == 2: y_fire_pm1 = y_fire_pm1[:,0]
    if y_smoke_pm1.ndim == 2: y_smoke_pm1 = y_smoke_pm1[:,0]

    pred_h_all = fire_pos_pred | smoke_pos_pred
    true_h_all = (y_fire_pm1 > 0) | (y_smoke_pm1 > 0)

    P_list, R_list, F1_list = [], [], []
    with h5py.File(h5_path, "r") as h5:
        starts = h5["meta/sample_start"][:]
        counts = h5["meta/sample_count"][:]
        n_imgs = len(starts)

        for i in range(n_imgs):
            s = int(starts[i]); c = int(counts[i])
            if c <= 0:
                continue
            sl = slice(s, s+c)

            pred_h = pred_h_all[sl]
            true_h = true_h_all[sl]

            tp = (pred_h & true_h).sum().item()
            fp = (pred_h & (~true_h)).sum().item()
            fn = ((~pred_h) & true_h).sum().item()

            P = tp / (tp + fp + 1e-12)
            R = tp / (tp + fn + 1e-12)
            F1 = (2*P*R) / (P + R + 1e-12)

            P_list.append(P); R_list.append(R); F1_list.append(F1)

    if len(P_list) == 0:
        return {"haz_P_mean": 0.0, "haz_R_mean": 0.0, "haz_F1_mean": 0.0, "n_images_used": 0}

    return {
        "haz_P_mean": float(np.mean(P_list)),
        "haz_R_mean": float(np.mean(R_list)),
        "haz_F1_mean": float(np.mean(F1_list)),
        "n_images_used": int(len(P_list)),
    }
# Назначение: загружает данные, модель или служебную информацию.
def load_test_from_h5_cpu(h5_path):
    """
    Загружает данные, модель или служебную информацию.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `load_test_from_h5_cpu`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = load_test_from_h5_cpu(h5_path='путь/к/ресурсу')
    """
    with h5py.File(h5_path, "r") as h5:
        X = h5["samples/X"][:]              # (N,278) float16
        y_fire01  = h5["samples/y_fire"][:] # 0/1
        y_smoke01 = h5["samples/y_smoke"][:]
    return X, y_fire01, y_smoke01

# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def to_pm1(y01):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    y01 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `to_pm1`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = to_pm1(y01=...)
    """
    y = y01.astype(np.float32)
    return y*2.0 - 1.0
@torch.no_grad()
# Назначение: выполняет оценку модели.
def eval_one_model_on_test(
    name: str,
    model,
    Xte_n,
    y_task_pm1,             # y_smoke или y_fire в {-1,+1}
    thr_out: float,
    bs: int = 32768,
):
    """
    Выполняет оценку модели.

    Параметры
    ---------
    name :
        Параметр используется в соответствии с назначением функции.
    model :
        Параметр используется в соответствии с назначением функции.
    Xte_n :
        Параметр используется в соответствии с назначением функции.
    y_task_pm1 :
        Параметр используется в соответствии с назначением функции.
    thr_out :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `eval_one_model_on_test`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = eval_one_model_on_test(name=..., model=..., Xte_n=...)
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = infer_out_chunked(model, Xte_n, bs=bs)   # (N,1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    sec = max(t1 - t0, 1e-9)
    N = Xte_n.shape[0]
    pps = N / sec

    m = binary_metrics_pm1(out, y_task_pm1, thr_out=thr_out)

    rep = {
        "name": name,
        "thr_out": float(thr_out),
        "N": int(N),
        "time_sec": float(sec),
        "pts_per_sec": float(pps),
        **m,
    }
    return rep, out
@torch.no_grad()
# Назначение: выполняет оценку модели.
def eval_3class_from_two_outputs(
    out_fire, thr_fire,
    out_smoke, thr_smoke,
    y_fire_pm1, y_smoke_pm1,
    h5_test_path: str,
    bs_dummy: int = 32768,
    thresholds=(0.02,0.05,0.07,0.10),
):
    """
    Выполняет оценку модели.

    Параметры
    ---------
    out_fire :
        Параметр используется в соответствии с назначением функции.
    thr_fire :
        Параметр используется в соответствии с назначением функции.
    out_smoke :
        Параметр используется в соответствии с назначением функции.
    thr_smoke :
        Параметр используется в соответствии с назначением функции.
    y_fire_pm1 :
        Параметр используется в соответствии с назначением функции.
    y_smoke_pm1 :
        Параметр используется в соответствии с назначением функции.
    h5_test_path :
        Параметр используется в соответствии с назначением функции.
    bs_dummy :
        Параметр используется в соответствии с назначением функции.
    thresholds :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `eval_3class_from_two_outputs`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = eval_3class_from_two_outputs(out_fire=..., thr_fire=0.5, out_smoke=...)
    """
    device = out_fire.device

    fire_pos  = (out_fire[:,0] if out_fire.ndim==2 else out_fire) >= float(thr_fire)
    smoke_pos = (out_smoke[:,0] if out_smoke.ndim==2 else out_smoke) >= float(thr_smoke)

    smoke_pos = smoke_pos & (~fire_pos)

    true_fire  = y_fire_pm1 > 0
    true_smoke = (y_smoke_pm1 > 0) & (~true_fire)  # чтобы не было двойной истины
    y_true3 = make_3class_pred(true_fire, true_smoke)   # 0/1/2
    y_pred3 = make_3class_pred(fire_pos, smoke_pos)

    cm = confusion_3x3(y_true3, y_pred3)
    rep = {"cm_3x3": cm.detach().cpu().numpy()}
    rep.update(prf_from_cm_3class(cm))

    with h5py.File(h5_test_path, "r") as h5:
        img_rep = image_level_stats_from_offsets(
            h5, y_pred3, thresholds=thresholds
        )
    rep.update(img_rep)
    return rep

@torch.no_grad()
# Назначение: определяет или применяет логику комитета классификаторов.
def eval_committee_two_outputs(
    h5_test_path: str,
    out_fire, thr_fire,
    out_smoke, thr_smoke,
    y_fire_pm1, y_smoke_pm1,
    thresholds=(0.02,0.05,0.07,0.10),
):
    """
    Определяет или применяет логику комитета классификаторов.

    Параметры
    ---------
    h5_test_path :
        Параметр используется в соответствии с назначением функции.
    out_fire :
        Параметр используется в соответствии с назначением функции.
    thr_fire :
        Параметр используется в соответствии с назначением функции.
    out_smoke :
        Параметр используется в соответствии с назначением функции.
    thr_smoke :
        Параметр используется в соответствии с назначением функции.
    y_fire_pm1 :
        Параметр используется в соответствии с назначением функции.
    y_smoke_pm1 :
        Параметр используется в соответствии с назначением функции.
    thresholds :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `eval_committee_two_outputs`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = eval_committee_two_outputs(h5_test_path='путь/к/ресурсу', out_fire=..., thr_fire=0.5)
    """
    device = out_fire.device

    # (N,)
    of = out_fire[:,0] if out_fire.ndim==2 else out_fire
    os_ = out_smoke[:,0] if out_smoke.ndim==2 else out_smoke

    fire_pos  = of >= float(thr_fire)
    smoke_pos = os_ >= float(thr_smoke)

    # mutually exclusive: fire overrides smoke
    smoke_pos = smoke_pos & (~fire_pos)

    # ----- 3-class true/pred -----
    true_fire  = (y_fire_pm1[:,0] if y_fire_pm1.ndim==2 else y_fire_pm1) > 0
    true_smoke = ((y_smoke_pm1[:,0] if y_smoke_pm1.ndim==2 else y_smoke_pm1) > 0) & (~true_fire)

    y_true3 = make_3class_pred(true_fire, true_smoke)
    y_pred3 = make_3class_pred(fire_pos, smoke_pos)

    cm = confusion_3x3(y_true3, y_pred3)
    rep = {"cm_3x3": cm.detach().cpu().numpy()}
    rep.update(prf_from_cm_3class(cm))

    # ----- hazard (2-class) -----
    rep.update(hazard_metrics_global(fire_pos, smoke_pos, y_fire_pm1, y_smoke_pm1))
    rep.update(hazard_metrics_mean_per_image(h5_test_path, fire_pos, smoke_pos, y_fire_pm1, y_smoke_pm1))

    # ----- image-level ratios/detections (your existing logic) -----
    with h5py.File(h5_test_path, "r") as h5:
        rep.update(image_level_stats_from_offsets(h5, y_pred3, thresholds=thresholds))

    return rep, y_pred3
# Назначение: сохраняет данные на диск.
def save_outs_h5(path, **outs):
    """
    Сохраняет данные на диск.

    Параметры
    ---------
    path :
        Параметр используется в соответствии с назначением функции.
    **outs :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `save_outs_h5`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = save_outs_h5(path='путь/к/ресурсу')
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as h5:
        for k,v in outs.items():
            if torch.is_tensor(v):
                v = v.detach().cpu()
                if v.ndim == 2 and v.shape[1] == 1:
                    v = v[:,0]
                v = v.numpy()
            h5.create_dataset(k, data=v.astype(np.float16), compression="lzf")
        # meta
        first_key = list(outs.keys())[0]
        N = h5[first_key].shape[0]
        h5.attrs["N"] = int(N)
    print("Saved outs:", path)

# Назначение: загружает данные, модель или служебную информацию.
def load_outs_h5(path, device="cuda"):
    """
    Загружает данные, модель или служебную информацию.

    Параметры
    ---------
    path :
        Параметр используется в соответствии с назначением функции.
    device :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `load_outs_h5`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = load_outs_h5(path='путь/к/ресурсу', device='cuda')
    """
    outs = {}
    with h5py.File(path, "r") as h5:
        for k in h5.keys():
            outs[k] = torch.from_numpy(h5[k][:].astype(np.float32)).to(device=device)
        N = int(h5.attrs.get("N", outs[list(outs.keys())[0]].shape[0]))
    print("Loaded outs:", path, "| N=", N)
    return outs
# Назначение: вычисляет смещение по заданным параметрам.
def binary_metrics_mean_per_image_from_offsets(
    h5_path: str,
    pred_pos_all: torch.Tensor,  # bool (N,)
    y_pm1_all: torch.Tensor,     # {-1,+1} (N,) or (N,1)
):
    """
    Вычисляет смещение по заданным параметрам.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.
    pred_pos_all :
        Параметр используется в соответствии с назначением функции.
    y_pm1_all :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `binary_metrics_mean_per_image_from_offsets`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = binary_metrics_mean_per_image_from_offsets(h5_path='путь/к/ресурсу', pred_pos_all=..., y_pm1_all=...)
    """
    if y_pm1_all.ndim == 2:
        y_pm1_all = y_pm1_all[:, 0]

    P_list, R_list, F1_list = [], [], []
    with h5py.File(h5_path, "r") as h5:
        starts = h5["meta/sample_start"][:]
        counts = h5["meta/sample_count"][:]
        n_imgs = len(starts)

        for i in range(n_imgs):
            s = int(starts[i]); c = int(counts[i])
            if c <= 0:
                continue
            sl = slice(s, s+c)

            pred_pos = pred_pos_all[sl]
            true_pos = (y_pm1_all[sl] > 0)

            tp = (pred_pos & true_pos).sum().item()
            fp = (pred_pos & (~true_pos)).sum().item()
            fn = ((~pred_pos) & true_pos).sum().item()

            P = tp / (tp + fp + 1e-12)
            R = tp / (tp + fn + 1e-12)
            F1 = (2*P*R) / (P + R + 1e-12)

            P_list.append(P); R_list.append(R); F1_list.append(F1)

    if len(P_list) == 0:
        return {"P_mean": 0.0, "R_mean": 0.0, "F1_mean": 0.0, "n_images_used": 0}

    return {
        "P_mean": float(np.mean(P_list)),
        "R_mean": float(np.mean(R_list)),
        "F1_mean": float(np.mean(F1_list)),
        "n_images_used": int(len(P_list)),
    }
@torch.no_grad()
# Назначение: выполняет оценку модели.
def eval_one_model_on_test_full(
    name: str,
    model,
    Xte,
    y_pm1,
    thr_out: float,
    h5_test_path: str,
    bs: int = 32768,
    prefix: str | None = None,
):
    """
    Выполняет оценку модели.

    Параметры
    ---------
    name :
        Параметр используется в соответствии с назначением функции.
    model :
        Параметр используется в соответствии с назначением функции.
    Xte :
        Параметр используется в соответствии с назначением функции.
    y_pm1 :
        Параметр используется в соответствии с назначением функции.
    thr_out :
        Параметр используется в соответствии с назначением функции.
    h5_test_path :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.
    prefix :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `eval_one_model_on_test_full`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = eval_one_model_on_test_full(name=..., model=..., Xte=...)
    """
    if prefix is None:
        prefix = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

    # 1) инференс
    out = infer_out_chunked(model, Xte, bs=bs)  # (N,1)

    rep_g = binary_metrics_pm1(out, y_pm1, thr_out=thr_out)  # dict

    out1 = out[:, 0] if out.ndim == 2 else out
    pred_pos = out1 >= float(thr_out)
    rep_l = binary_metrics_mean_per_image_from_offsets(
        h5_test_path, pred_pos_all=pred_pos, y_pm1_all=y_pm1
    )

    rep = {
        "name": name,
        "thr_out": float(thr_out),
        "n_samples": int(out1.numel()),
    }

    for k, v in rep_g.items():
        rep[f"global_{k}"] = v

    for k, v in rep_l.items():
        rep[f"mean_{k}"] = v

    return rep, out
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def _as_1d_pm1(y_pm1: torch.Tensor) -> torch.Tensor:
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    y_pm1 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_as_1d_pm1`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _as_1d_pm1(y_pm1=...)
    """
    if y_pm1.ndim == 2:
        y_pm1 = y_pm1[:, 0]
    return y_pm1.float()

# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def _as_1d_out(out: torch.Tensor) -> torch.Tensor:
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    out :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_as_1d_out`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _as_1d_out(out=...)
    """
    if out.ndim == 2:
        out = out[:, 0]
    return out.float()

# Назначение: считывает данные из внешнего источника.
def _read_offsets(h5_path: str):
    """
    Считывает данные из внешнего источника.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_read_offsets`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _read_offsets(h5_path='путь/к/ресурсу')
    """
    with h5py.File(h5_path, "r") as h5:
        starts = h5["meta/sample_start"][:].astype(np.int64)
        counts = h5["meta/sample_count"][:].astype(np.int64)
    return starts, counts

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def infer_out_chunked(model, X, bs=32768, sync_cuda=True):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    model :
        Параметр используется в соответствии с назначением функции.
    X :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.
    sync_cuda :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `infer_out_chunked`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = infer_out_chunked(model=..., X=..., bs=...)
    """
    model.eval()
    device = X.device
    N = X.shape[0]

    # timing
    if device.type == "cuda" and sync_cuda:
        torch.cuda.synchronize()
    t0 = time.time()

    outs = []
    for i in range(0, N, bs):
        outs.append(model(X[i:i+bs].float()))  # (B,1)
    out = torch.cat(outs, dim=0)              # (N,1)

    if device.type == "cuda" and sync_cuda:
        torch.cuda.synchronize()
    t1 = time.time()

    elapsed = max(t1 - t0, 1e-12)
    return out, elapsed, (N / elapsed)


@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def binary_metrics_pm1_from_out(out, y_pm1, thr_out=0.0):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    out :
        Параметр используется в соответствии с назначением функции.
    y_pm1 :
        Параметр используется в соответствии с назначением функции.
    thr_out :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `binary_metrics_pm1_from_out`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = binary_metrics_pm1_from_out(out=..., y_pm1=..., thr_out=0.5)
    """
    out1 = _as_1d_out(out)
    y1  = _as_1d_pm1(y_pm1)

    pred_pos = out1 >= float(thr_out)
    true_pos = y1 > 0

    tp = (pred_pos & true_pos).sum().item()
    fp = (pred_pos & (~true_pos)).sum().item()
    fn = ((~pred_pos) & true_pos).sum().item()
    tn = ((~pred_pos) & (~true_pos)).sum().item()

    P = tp / (tp + fp + 1e-12)
    R = tp / (tp + fn + 1e-12)
    F1 = (2*P*R) / (P + R + 1e-12)

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    bal_acc = 0.5 * (R + tnr)

    return {
        "P": P, "R": R, "F1": F1,
        "acc": acc, "bal_acc": bal_acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "out_mean": float(out1.mean().item()),
        "out_std":  float(out1.std(unbiased=False).item()),
        "pos_rate_true": float(true_pos.float().mean().item()),
        "pos_rate_pred": float(pred_pos.float().mean().item()),
    }

# Назначение: вычисляет смещение по заданным параметрам.
def binary_metrics_mean_per_image_from_offsets(
    h5_path: str,
    pred_pos_all: torch.Tensor,   # bool (N,)
    y_pm1_all: torch.Tensor,      # {-1,+1} (N,) or (N,1)
    thresholds=(0.02, 0.05, 0.07, 0.10),
):
    """
    Вычисляет смещение по заданным параметрам.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.
    pred_pos_all :
        Параметр используется в соответствии с назначением функции.
    y_pm1_all :
        Параметр используется в соответствии с назначением функции.
    thresholds :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `binary_metrics_mean_per_image_from_offsets`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = binary_metrics_mean_per_image_from_offsets(h5_path='путь/к/ресурсу', pred_pos_all=..., y_pm1_all=...)
    """
    if y_pm1_all.ndim == 2:
        y_pm1_all = y_pm1_all[:, 0]
    y_pm1_all = y_pm1_all.float()

    starts, counts = _read_offsets(h5_path)
    n_imgs = int(len(starts))

    P_list, R_list, F1_list = [], [], []
    pred_ratio_list = []
    true_ratio_list = []

    thr_list = list(thresholds)
    det_all = {t: 0 for t in thr_list}
    det_tp  = {t: 0 for t in thr_list}  # GT-positive images detected
    det_fp  = {t: 0 for t in thr_list}  # GT-negative images falsely detected
    n_gt_pos = 0
    n_gt_neg = 0
    n_used   = 0

    for i in range(n_imgs):
        s = int(starts[i]); c = int(counts[i])
        if c <= 0:
            continue
        n_used += 1
        sl = slice(s, s+c)

        pred_pos = pred_pos_all[sl]
        true_pos = (y_pm1_all[sl] > 0)

        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & (~true_pos)).sum().item()
        fn = ((~pred_pos) & true_pos).sum().item()

        P = tp / (tp + fp + 1e-12)
        R = tp / (tp + fn + 1e-12)
        F1 = (2*P*R) / (P + R + 1e-12)
        P_list.append(P); R_list.append(R); F1_list.append(F1)

        n = int(c)
        pred_ratio = float(pred_pos.float().mean().item())
        true_ratio = float(true_pos.float().mean().item())
        pred_ratio_list.append(pred_ratio)
        true_ratio_list.append(true_ratio)

        gt_pos_img = (true_ratio > 0.0)
        if gt_pos_img:
            n_gt_pos += 1
        else:
            n_gt_neg += 1

        for t in thr_list:
            if pred_ratio >= t:
                det_all[t] += 1
                if gt_pos_img:
                    det_tp[t] += 1
                else:
                    det_fp[t] += 1

    if n_used == 0:
        rep = {"n_images_used": 0}
        for t in thr_list:
            rep[f"det_img@{t:.4f}"] = 0.0
            rep[f"recall_img@{t:.4f}"] = 0.0
            rep[f"fpr_img@{t:.4f}"] = 0.0
        rep.update({"P_mean": 0.0, "R_mean": 0.0, "F1_mean": 0.0})
        return rep

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
        rep[f"det_img@{t:.4f}"]      = det_all[t] / max(n_used, 1)
        rep[f"recall_img@{t:.4f}"]   = det_tp[t]  / max(n_gt_pos, 1)  # “процент найденных пожаров” среди GT-пожаров
        rep[f"fpr_img@{t:.4f}"]      = det_fp[t]  / max(n_gt_neg, 1)  # ложные срабатывания по изображениям
        rep[f"tp_img@{t:.4f}"]       = int(det_tp[t])
        rep[f"fp_img@{t:.4f}"]       = int(det_fp[t])
    return rep

@torch.no_grad()
# Назначение: выполняет оценку модели.
def eval_one_model_on_test_full(
    name: str,
    model,
    Xte: torch.Tensor,
    y_pm1: torch.Tensor,
    thr_out: float,
    h5_test_path: str,
    bs: int = 32768,
    thresholds=(0.02, 0.05, 0.07, 0.10),
    save_out_path: str | None = None,
    prefix: str | None = None,
):
    """
    Выполняет оценку модели.

    Параметры
    ---------
    name :
        Параметр используется в соответствии с назначением функции.
    model :
        Параметр используется в соответствии с назначением функции.
    Xte :
        Параметр используется в соответствии с назначением функции.
    y_pm1 :
        Параметр используется в соответствии с назначением функции.
    thr_out :
        Параметр используется в соответствии с назначением функции.
    h5_test_path :
        Параметр используется в соответствии с назначением функции.
    bs :
        Параметр используется в соответствии с назначением функции.
    thresholds :
        Параметр используется в соответствии с назначением функции.
    save_out_path :
        Параметр используется в соответствии с назначением функции.
    prefix :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `eval_one_model_on_test_full`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = eval_one_model_on_test_full(name=..., model=..., Xte=...)
    """
    if prefix is None:
        prefix = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

    # 1) inference + speed
    out, elapsed_s, pix_per_s = infer_out_chunked(model, Xte, bs=bs)
    out1 = _as_1d_out(out)

    # 2) global metrics
    rep_g = binary_metrics_pm1_from_out(out1, y_pm1, thr_out=thr_out)

    # 3) mean-per-image metrics (and image-level detection rates)
    pred_pos = out1 >= float(thr_out)
    rep_l = binary_metrics_mean_per_image_from_offsets(
        h5_test_path,
        pred_pos_all=pred_pos,
        y_pm1_all=y_pm1,
        thresholds=thresholds,
    )

    # 4) pack into one flat dict (so it goes nicely into a DataFrame)
    rep = {
        "name": name,
        "thr_out": float(thr_out),
        "n_samples": int(out1.numel()),
        "bs_infer": int(bs),
        "speed_seconds": float(elapsed_s),
        "speed_pixels_per_sec": float(pix_per_s),
    }
    for k, v in rep_g.items():
        rep[f"global_{k}"] = v
    for k, v in rep_l.items():
        rep[f"mean_{k}"] = v

    # 5) optional save outputs (GPU->CPU to keep file small/stable)
    #    Save out as float16 to reduce disk, plus the threshold and name for reuse.
    if save_out_path is not None:
        os.makedirs(os.path.dirname(save_out_path), exist_ok=True)
        payload = {
            "name": name,
            "thr_out": float(thr_out),
            "out_f16": out1.detach().to("cpu", dtype=torch.float16),
            "n_samples": int(out1.numel()),
        }
        torch.save(payload, save_out_path)
        rep["saved_out_path"] = save_out_path

    return rep, out  
# Назначение: загружает данные, модель или служебную информацию.
def load_saved_out_pt(path: str, map_location="cpu"):
    """
    Загружает данные, модель или служебную информацию.

    Параметры
    ---------
    path :
        Параметр используется в соответствии с назначением функции.
    map_location :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `load_saved_out_pt`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = load_saved_out_pt(path='путь/к/ресурсу', map_location=...)
    """
    ckpt = torch.load(path, map_location=map_location)
    out = ckpt["out_f16"]
    if out.ndim == 2:
        out = out[:, 0]
    out = out.float().contiguous()
    return ckpt, out

# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def pm1_to_bool(y_pm1: torch.Tensor):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    y_pm1 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `pm1_to_bool`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = pm1_to_bool(y_pm1=...)
    """
    if y_pm1.ndim == 2: y_pm1 = y_pm1[:, 0]
    return (y_pm1.float() > 0)

# Назначение: считывает данные из внешнего источника.
def read_offsets(h5_path: str):
    """
    Считывает данные из внешнего источника.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `read_offsets`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = read_offsets(h5_path='путь/к/ресурсу')
    """
    with h5py.File(h5_path, "r") as h5:
        starts = h5["meta/sample_start"][:].astype(np.int64)
        counts = h5["meta/sample_count"][:].astype(np.int64)
    return starts, counts

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def global_metrics_from_pred_bool(pred_pos: torch.Tensor, y_pm1: torch.Tensor):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    pred_pos :
        Параметр используется в соответствии с назначением функции.
    y_pm1 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `global_metrics_from_pred_bool`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = global_metrics_from_pred_bool(pred_pos=..., y_pm1=...)
    """
    if y_pm1.ndim == 2: y_pm1 = y_pm1[:, 0]
    y_true = (y_pm1.float() > 0)

    tp = (pred_pos & y_true).sum().item()
    fp = (pred_pos & (~y_true)).sum().item()
    fn = ((~pred_pos) & y_true).sum().item()
    tn = ((~pred_pos) & (~y_true)).sum().item()

    P = tp / (tp + fp + 1e-12)
    R = tp / (tp + fn + 1e-12)
    F1 = (2*P*R) / (P + R + 1e-12)

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    bal_acc = 0.5 * (R + tnr)

    return dict(P=P, R=R, F1=F1, acc=acc, bal_acc=bal_acc,
                tp=tp, fp=fp, fn=fn, tn=tn,
                pos_rate_true=float(y_true.float().mean().item()),
                pos_rate_pred=float(pred_pos.float().mean().item()))

# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def mean_per_image_and_detection(
    h5_path: str,
    pred_pos_all: torch.Tensor,   # bool (N,)
    y_pm1_all: torch.Tensor,      # {-1,+1}
    thresholds=(0.02,0.05,0.07,0.10),
):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.
    pred_pos_all :
        Параметр используется в соответствии с назначением функции.
    y_pm1_all :
        Параметр используется в соответствии с назначением функции.
    thresholds :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `mean_per_image_and_detection`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = mean_per_image_and_detection(h5_path='путь/к/ресурсу', pred_pos_all=..., y_pm1_all=...)
    """
    if y_pm1_all.ndim == 2:
        y_pm1_all = y_pm1_all[:, 0]
    y_pm1_all = y_pm1_all.float()

    starts, counts = read_offsets(h5_path)
    n_imgs = int(len(starts))

    P_list, R_list, F1_list = [], [], []
    pred_ratio_list = []
    true_ratio_list = []

    thr_list = list(thresholds)
    det_all = {t: 0 for t in thr_list}
    det_tp  = {t: 0 for t in thr_list}  # GT-positive images detected
    det_fp  = {t: 0 for t in thr_list}  # GT-negative images falsely detected
    n_gt_pos = 0
    n_gt_neg = 0
    n_used   = 0

    for i in range(n_imgs):
        s = int(starts[i]); c = int(counts[i])
        if c <= 0:
            continue
        n_used += 1
        sl = slice(s, s+c)

        pred_pos = pred_pos_all[sl]
        true_pos = (y_pm1_all[sl] > 0)

        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & (~true_pos)).sum().item()
        fn = ((~pred_pos) & true_pos).sum().item()

        P = tp / (tp + fp + 1e-12)
        R = tp / (tp + fn + 1e-12)
        F1 = (2*P*R) / (P + R + 1e-12)

        P_list.append(P); R_list.append(R); F1_list.append(F1)

        pred_ratio = float(pred_pos.float().mean().item())
        true_ratio = float(true_pos.float().mean().item())
        pred_ratio_list.append(pred_ratio)
        true_ratio_list.append(true_ratio)

        gt_pos_img = (true_ratio > 0.0)
        if gt_pos_img: n_gt_pos += 1
        else:          n_gt_neg += 1

        for t in thr_list:
            if pred_ratio >= t:
                det_all[t] += 1
                if gt_pos_img: det_tp[t] += 1
                else:          det_fp[t] += 1

    if n_used == 0:
        rep = {"n_images_used": 0, "P_mean": 0.0, "R_mean": 0.0, "F1_mean": 0.0}
        for t in thr_list:
            rep[f"det_img@{t:.4f}"] = 0.0
            rep[f"recall_img@{t:.4f}"] = 0.0
            rep[f"fpr_img@{t:.4f}"] = 0.0
        return rep

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
        rep[f"recall_img@{t:.4f}"] = det_tp[t]  / max(n_gt_pos, 1)  # “процент найденных пожаров” среди GT-дыма
        rep[f"fpr_img@{t:.4f}"]    = det_fp[t]  / max(n_gt_neg, 1)
        rep[f"tp_img@{t:.4f}"]     = int(det_tp[t])
        rep[f"fp_img@{t:.4f}"]     = int(det_fp[t])
    return rep

# Назначение: выполняет обработку признаков или меток, относящихся к дыму.
def committee_pred_smoke(
    pred1: torch.Tensor, pred2: torch.Tensor,
    pred3: torch.Tensor | None,
    mode: str,
):
    """
    Выполняет обработку признаков или меток, относящихся к дыму.

    Параметры
    ---------
    pred1 :
        Параметр используется в соответствии с назначением функции.
    pred2 :
        Параметр используется в соответствии с назначением функции.
    pred3 :
        Параметр используется в соответствии с назначением функции.
    mode :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `committee_pred_smoke`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = committee_pred_smoke(pred1=..., pred2=..., pred3=...)
    """
    mode = mode.lower()
    if mode == "voting_12":
        return (pred1 | pred2)
    if mode == "confirm_12":
        return (pred1 & pred2)
    if mode == "voting_123":
        assert pred3 is not None
        votes = pred1.int() + pred2.int() + pred3.int()
        return (votes >= 2)
    if mode == "voting12_confirm3":
        assert pred3 is not None
        return (pred1 | pred2) & pred3
    raise ValueError(f"Unknown mode: {mode}")

# Назначение: реализует процедуру голосования нескольких классификаторов.
def committee_vote_fraction(preds: list[torch.Tensor]):
    """
    Реализует процедуру голосования нескольких классификаторов.

    Параметры
    ---------
    preds :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `committee_vote_fraction`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = committee_vote_fraction(preds=...)
    """
    k = len(preds)
    v = torch.zeros_like(preds[0], dtype=torch.float32)
    for p in preds:
        v += p.float()
    return v / float(k)

# Назначение: выполняет обработку признаков или меток, относящихся к дыму.
def eval_one_smoke_committee(
    name: str,
    h5_test_path: str,
    y_smoke_pm1: torch.Tensor,
    pred_ins1: torch.Tensor,
    pred_ins2: torch.Tensor,
    pred_ins3: torch.Tensor | None,
    mode: str,
    thresholds=(0.02,0.05,0.07,0.10),
    save_dir: str | None = None,
):
    """
    Выполняет обработку признаков или меток, относящихся к дыму.

    Параметры
    ---------
    name :
        Параметр используется в соответствии с назначением функции.
    h5_test_path :
        Параметр используется в соответствии с назначением функции.
    y_smoke_pm1 :
        Параметр используется в соответствии с назначением функции.
    pred_ins1 :
        Параметр используется в соответствии с назначением функции.
    pred_ins2 :
        Параметр используется в соответствии с назначением функции.
    pred_ins3 :
        Параметр используется в соответствии с назначением функции.
    mode :
        Параметр используется в соответствии с назначением функции.
    thresholds :
        Параметр используется в соответствии с назначением функции.
    save_dir :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `eval_one_smoke_committee`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = eval_one_smoke_committee(name=..., h5_test_path='путь/к/ресурсу', y_smoke_pm1=...)
    """
    # предсказания комитета
    pred_comm = committee_pred_smoke(pred_ins1, pred_ins2, pred_ins3, mode=mode)

    # “уверенность”
    voters = [pred_ins1, pred_ins2] + ([] if pred_ins3 is None else [pred_ins3])
    vote_frac = committee_vote_fraction(voters)

    rep_g = global_metrics_from_pred_bool(pred_comm, y_smoke_pm1)
    rep_l = mean_per_image_and_detection(h5_test_path, pred_comm, y_smoke_pm1, thresholds=thresholds)

    rep = {"name": name, "mode": mode, "n_samples": int(pred_comm.numel())}
    for k,v in rep_g.items():
        rep[f"global_{k}"] = v
    for k,v in rep_l.items():
        rep[f"mean_{k}"] = v

    rep["vote_frac_mean"] = float(vote_frac.mean().item())
    rep["vote_frac_std"]  = float(vote_frac.std(unbiased=False).item())

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path_pred = os.path.join(save_dir, f"{name.replace(' ','_')}_pred_u8.pt")
        path_vote = os.path.join(save_dir, f"{name.replace(' ','_')}_vote_f16.pt")

        torch.save({"name": name, "pred_u8": pred_comm.to("cpu", dtype=torch.uint8)}, path_pred)
        torch.save({"name": name, "vote_f16": vote_frac.to("cpu", dtype=torch.float16)}, path_vote)

        rep["saved_pred_path"] = path_pred
        rep["saved_vote_path"] = path_vote

    return rep

# Назначение: выполняет обработку признаков или меток, относящихся к дыму.
def run_smoke_committees_from_saved_outs(
    h5_test_path: str,
    y_smoke_pm1: torch.Tensor,
    out_ins1_path: str,
    out_ins2_path: str,
    out_ins3A_path: str,
    out_ins3B_path: str,
    thr_ins1: float,
    thr_ins2: float,
    thr_ins3A: float,
    thr_ins3B: float,
    thresholds_img=(0.02,0.05,0.07,0.10),
    save_dir="/content/committee_preds_smoke",
):
    """
    Выполняет обработку признаков или меток, относящихся к дыму.

    Параметры
    ---------
    h5_test_path :
        Параметр используется в соответствии с назначением функции.
    y_smoke_pm1 :
        Параметр используется в соответствии с назначением функции.
    out_ins1_path :
        Параметр используется в соответствии с назначением функции.
    out_ins2_path :
        Параметр используется в соответствии с назначением функции.
    out_ins3A_path :
        Параметр используется в соответствии с назначением функции.
    out_ins3B_path :
        Параметр используется в соответствии с назначением функции.
    thr_ins1 :
        Параметр используется в соответствии с назначением функции.
    thr_ins2 :
        Параметр используется в соответствии с назначением функции.
    thr_ins3A :
        Параметр используется в соответствии с назначением функции.
    thr_ins3B :
        Параметр используется в соответствии с назначением функции.
    thresholds_img :
        Параметр используется в соответствии с назначением функции.
    save_dir :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `run_smoke_committees_from_saved_outs`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = run_smoke_committees_from_saved_outs(h5_test_path='путь/к/ресурсу', y_smoke_pm1=..., out_ins1_path='путь/к/ресурсу')
    """
    _, out1 = load_saved_out_pt(out_ins1_path, map_location="cpu")
    _, out2 = load_saved_out_pt(out_ins2_path, map_location="cpu")
    _, out3A = load_saved_out_pt(out_ins3A_path, map_location="cpu")
    _, out3B = load_saved_out_pt(out_ins3B_path, map_location="cpu")

    # sanity length check
    N = int(out1.numel())
    assert out2.numel() == N and out3A.numel() == N and out3B.numel() == N, "Out lengths mismatch"

    # predictions (bool on CPU)
    pred1  = out1 >= float(thr_ins1)
    pred2  = out2 >= float(thr_ins2)
    pred3A = out3A >= float(thr_ins3A)
    pred3B = out3B >= float(thr_ins3B)

    # build committees list
    committees = [
        ("C1 INS1+INS2 voting",                    "voting_12",         pred3A, False, "no3"),
        ("C2 INS1+INS2 confirmation",              "confirm_12",        pred3A, False, "no3"),
        ("C3 INS1+INS2+INS3A voting",              "voting_123",        pred3A, True,  "3A"),
        ("C4 INS1+INS2+INS3B voting",              "voting_123",        pred3B, True,  "3B"),
        ("C5 (INS1,INS2 voting) + INS3A confirm",  "voting12_confirm3", pred3A, True,  "3A"),
        ("C6 (INS1,INS2 voting) + INS3B confirm",  "voting12_confirm3", pred3B, True,  "3B"),
    ]

    results = []
    for cname, mode, p3, use3, tag in committees:
        rep = eval_one_smoke_committee(
            name=cname,
            h5_test_path=h5_test_path,
            y_smoke_pm1=y_smoke_pm1,
            pred_ins1=pred1,
            pred_ins2=pred2,
            pred_ins3=(p3 if use3 else None),
            mode=mode,
            thresholds=thresholds_img,
            save_dir=save_dir,
        )
        results.append(rep)

    df = pd.DataFrame(results)
    return df


# Назначение: выполняет обработку признаков или меток, относящихся к огню.
def run_fire_committees_from_saved_outs(
    h5_test_path: str,
    y_fire_pm1: torch.Tensor,
    out_ins4_path: str,
    out_ins5_path: str,
    thr_ins4: float,
    thr_ins5: float,
    thresholds_img=(0.02,0.05,0.07,0.10),
    save_dir="/content/committee_preds_fire",
):
    """
    Выполняет обработку признаков или меток, относящихся к огню.

    Параметры
    ---------
    h5_test_path :
        Параметр используется в соответствии с назначением функции.
    y_fire_pm1 :
        Параметр используется в соответствии с назначением функции.
    out_ins4_path :
        Параметр используется в соответствии с назначением функции.
    out_ins5_path :
        Параметр используется в соответствии с назначением функции.
    thr_ins4 :
        Параметр используется в соответствии с назначением функции.
    thr_ins5 :
        Параметр используется в соответствии с назначением функции.
    thresholds_img :
        Параметр используется в соответствии с назначением функции.
    save_dir :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `run_fire_committees_from_saved_outs`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = run_fire_committees_from_saved_outs(h5_test_path='путь/к/ресурсу', y_fire_pm1=..., out_ins4_path='путь/к/ресурсу')
    """
    _, out4 = load_saved_out_pt(out_ins4_path, map_location="cpu")
    _, out5 = load_saved_out_pt(out_ins5_path, map_location="cpu")

    N = int(out4.numel())
    assert out5.numel() == N, "Out lengths mismatch (INS4 vs INS5)"

    # predictions (bool on CPU)
    pred4 = out4 >= float(thr_ins4)
    pred5 = out5 >= float(thr_ins5)

    results = []

    # F1: voting (для двух = OR)
    rep_f1 = eval_one_smoke_committee(
        name="F1 INS4+INS5 voting",
        h5_test_path=h5_test_path,
        y_smoke_pm1=y_fire_pm1,         
        pred_ins1=pred4,                
        pred_ins2=pred5,
        pred_ins3=None,
        mode="voting_12",
        thresholds=thresholds_img,
        save_dir=save_dir,
    )
    results.append(rep_f1)

    # F2: confirmation (AND)
    rep_f2 = eval_one_smoke_committee(
        name="F2 INS4+INS5 confirmation",
        h5_test_path=h5_test_path,
        y_smoke_pm1=y_fire_pm1,
        pred_ins1=pred4,
        pred_ins2=pred5,
        pred_ins3=None,
        mode="confirm_12",
        thresholds=thresholds_img,
        save_dir=save_dir,
    )
    results.append(rep_f2)

    return pd.DataFrame(results)
# ============================================================
# Helpers (small + safe; do not require re-running models)
# ============================================================

# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def _to_cpu_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    x :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_to_cpu_1d`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _to_cpu_1d(x=...)
    """
    if x.ndim == 2: x = x[:, 0]
    return x.detach().to("cpu")

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def binary_metrics_from_pred(pred_pos: torch.Tensor, true_pos: torch.Tensor):
    # pred_pos/true_pos: bool (N,)
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    pred_pos :
        Параметр используется в соответствии с назначением функции.
    true_pos :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `binary_metrics_from_pred`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = binary_metrics_from_pred(pred_pos=..., true_pos=...)
    """
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

# Назначение: вычисляет смещение по заданным параметрам.
def binary_metrics_mean_per_image_from_offsets(
    h5_path: str,
    pred_pos_all: torch.Tensor,  # bool (N,)
    true_pos_all: torch.Tensor,  # bool (N,)
):
    """
    Вычисляет смещение по заданным параметрам.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.
    pred_pos_all :
        Параметр используется в соответствии с назначением функции.
    true_pos_all :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `binary_metrics_mean_per_image_from_offsets`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = binary_metrics_mean_per_image_from_offsets(h5_path='путь/к/ресурсу', pred_pos_all=..., true_pos_all=...)
    """
    P_list, R_list, F1_list = [], [], []
    with h5py.File(h5_path, "r") as h5:
        starts = h5["meta/sample_start"][:]
        counts = h5["meta/sample_count"][:]
        n_imgs = len(starts)

        for i in range(n_imgs):
            s = int(starts[i]); c = int(counts[i])
            if c <= 0:
                continue
            sl = slice(s, s+c)
            pred_pos = pred_pos_all[sl]
            true_pos = true_pos_all[sl]

            tp = (pred_pos & true_pos).sum().item()
            fp = (pred_pos & (~true_pos)).sum().item()
            fn = ((~pred_pos) & true_pos).sum().item()

            P = tp / (tp + fp + 1e-12)
            R = tp / (tp + fn + 1e-12)
            F1 = (2*P*R) / (P + R + 1e-12)

            P_list.append(P); R_list.append(R); F1_list.append(F1)

    if len(P_list) == 0:
        return {"P_mean": 0.0, "R_mean": 0.0, "F1_mean": 0.0, "n_images_used": 0}

    return {
        "P_mean": float(np.mean(P_list)),
        "R_mean": float(np.mean(R_list)),
        "F1_mean": float(np.mean(F1_list)),
        "n_images_used": int(len(P_list)),
    }

# Назначение: вычисляет смещение по заданным параметрам.
def image_detection_rate_from_offsets(
    h5_path: str,
    pred_pos_all: torch.Tensor,         # bool (N,)
    true_pos_all: torch.Tensor,         # bool (N,)
    thresholds=(0.02, 0.05, 0.07, 0.10) 
):
    """
    Вычисляет смещение по заданным параметрам.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.
    pred_pos_all :
        Параметр используется в соответствии с назначением функции.
    true_pos_all :
        Параметр используется в соответствии с назначением функции.
    thresholds :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `image_detection_rate_from_offsets`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = image_detection_rate_from_offsets(h5_path='путь/к/ресурсу', pred_pos_all=..., true_pos_all=...)
    """
    thr_list = list(thresholds)
    det_pred = {t: 0 for t in thr_list}
    det_true = {t: 0 for t in thr_list}  # сколько "истинно-пожарных" изображений по true_ratio>=t (опционально)
    det_tp   = {t: 0 for t in thr_list}  # pred>=t AND true>=t (image-level TP)

    with h5py.File(h5_path, "r") as h5:
        starts = h5["meta/sample_start"][:]
        counts = h5["meta/sample_count"][:]
        n_imgs = len(starts)

        for i in range(n_imgs):
            s = int(starts[i]); c = int(counts[i])
            if c <= 0:
                continue
            sl = slice(s, s+c)

            p = pred_pos_all[sl]
            t = true_pos_all[sl]
            n = int(p.numel())
            if n == 0:
                continue

            pred_ratio = float(p.sum().item()) / n
            true_ratio = float(t.sum().item()) / n

            for thr in thr_list:
                p_hit = pred_ratio >= thr
                t_hit = true_ratio >= thr
                if p_hit: det_pred[thr] += 1
                if t_hit: det_true[thr] += 1
                if p_hit and t_hit: det_tp[thr] += 1

    rep = {}
    rep["n_images"] = int(n_imgs)
    for thr in thr_list:
        rep[f"img_det_pred@{thr:.4f}"] = det_pred[thr] / max(n_imgs, 1)
        rep[f"img_true@{thr:.4f}"]     = det_true[thr] / max(n_imgs, 1)
        denom = max(det_true[thr], 1)
        rep[f"img_recall@{thr:.4f}"]   = det_tp[thr] / denom
    return rep

# 3-class helpers (fire priority)
@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def make_3class_pred(fire_pos: torch.Tensor, smoke_pos: torch.Tensor):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    fire_pos :
        Параметр используется в соответствии с назначением функции.
    smoke_pos :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `make_3class_pred`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = make_3class_pred(fire_pos=..., smoke_pos=...)
    """
    y3 = torch.zeros_like(fire_pos, dtype=torch.int64)
    y3[smoke_pos] = 1
    y3[fire_pos]  = 2
    return y3

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def make_3class_true(y_fire_pm1: torch.Tensor, y_smoke_pm1: torch.Tensor):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    y_fire_pm1 :
        Параметр используется в соответствии с назначением функции.
    y_smoke_pm1 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `make_3class_true`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = make_3class_true(y_fire_pm1=..., y_smoke_pm1=...)
    """
    fire_true  = y_fire_pm1 > 0
    smoke_true = y_smoke_pm1 > 0
    return make_3class_pred(fire_true, smoke_true)

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def confusion_3x3(y_true3, y_pred3):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    y_true3 :
        Параметр используется в соответствии с назначением функции.
    y_pred3 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `confusion_3x3`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = confusion_3x3(y_true3=..., y_pred3=...)
    """
    cm = torch.zeros((3,3), dtype=torch.int64)
    for t in range(3):
        for p in range(3):
            cm[t,p] = ((y_true3==t) & (y_pred3==p)).sum()
    return cm

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def prf_from_cm_3class(cm):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    cm :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `prf_from_cm_3class`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = prf_from_cm_3class(cm=...)
    """
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

    # micro/macro over smoke+fire
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

# Назначение: выполняет обработку признаков или меток, относящихся к дыму.
def smoke_committee_pred(mode: str, p1: torch.Tensor, p2: torch.Tensor, p3b: torch.Tensor | None):
    """
    Выполняет обработку признаков или меток, относящихся к дыму.

    Параметры
    ---------
    mode :
        Параметр используется в соответствии с назначением функции.
    p1 :
        Параметр используется в соответствии с назначением функции.
    p2 :
        Параметр используется в соответствии с назначением функции.
    p3b :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `smoke_committee_pred`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = smoke_committee_pred(mode=..., p1=..., p2=...)
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

# Назначение: выполняет обработку признаков или меток, относящихся к огню.
def fire_committee_pred(mode: str, p4: torch.Tensor, p5: torch.Tensor):
    """
    Выполняет обработку признаков или меток, относящихся к огню.

    Параметры
    ---------
    mode :
        Параметр используется в соответствии с назначением функции.
    p4 :
        Параметр используется в соответствии с назначением функции.
    p5 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `fire_committee_pred`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = fire_committee_pred(mode=..., p4=..., p5=...)
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

# Назначение: определяет или применяет логику комитета классификаторов.
def run_K_committees(
    h5_test_path: str,
    y_smoke_pm1: torch.Tensor,
    y_fire_pm1: torch.Tensor,
    out_ins1_path: str,
    out_ins2_path: str,
    out_ins3B_path: str,
    out_ins4_path: str,
    out_ins5_path: str,
    thr_ins1: float,
    thr_ins2: float,
    thr_ins3B: float,
    thr_ins4: float,
    thr_ins5: float,
    img_thresholds=(0.02, 0.05, 0.07, 0.10),
    save_dir="/content/committee_preds_K",
):
    """
    Определяет или применяет логику комитета классификаторов.

    Параметры
    ---------
    h5_test_path :
        Параметр используется в соответствии с назначением функции.
    y_smoke_pm1 :
        Параметр используется в соответствии с назначением функции.
    y_fire_pm1 :
        Параметр используется в соответствии с назначением функции.
    out_ins1_path :
        Параметр используется в соответствии с назначением функции.
    out_ins2_path :
        Параметр используется в соответствии с назначением функции.
    out_ins3B_path :
        Параметр используется в соответствии с назначением функции.
    out_ins4_path :
        Параметр используется в соответствии с назначением функции.
    out_ins5_path :
        Параметр используется в соответствии с назначением функции.
    thr_ins1 :
        Параметр используется в соответствии с назначением функции.
    thr_ins2 :
        Параметр используется в соответствии с назначением функции.
    thr_ins3B :
        Параметр используется в соответствии с назначением функции.
    thr_ins4 :
        Параметр используется в соответствии с назначением функции.
    thr_ins5 :
        Параметр используется в соответствии с назначением функции.
    img_thresholds :
        Параметр используется в соответствии с назначением функции.
    save_dir :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `run_K_committees`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = run_K_committees(h5_test_path='путь/к/ресурсу', y_smoke_pm1=..., y_fire_pm1=...)
    """
    os.makedirs(save_dir, exist_ok=True)

    _, o1  = load_saved_out_pt(out_ins1_path, map_location="cpu")
    _, o2  = load_saved_out_pt(out_ins2_path, map_location="cpu")
    _, o3b = load_saved_out_pt(out_ins3B_path, map_location="cpu")
    _, o4  = load_saved_out_pt(out_ins4_path, map_location="cpu")
    _, o5  = load_saved_out_pt(out_ins5_path, map_location="cpu")

    o1  = _to_cpu_1d(o1)
    o2  = _to_cpu_1d(o2)
    o3b = _to_cpu_1d(o3b)
    o4  = _to_cpu_1d(o4)
    o5  = _to_cpu_1d(o5)

    N = int(o1.numel())
    assert o2.numel()==N and o3b.numel()==N and o4.numel()==N and o5.numel()==N, "Out length mismatch"

    y_sm = _to_cpu_1d(y_smoke_pm1) > 0
    y_fi = _to_cpu_1d(y_fire_pm1) > 0
    y_hz = y_sm | y_fi  # hazard true

    p1  = o1  >= float(thr_ins1)
    p2  = o2  >= float(thr_ins2)
    p3b = o3b >= float(thr_ins3B)
    p4  = o4  >= float(thr_ins4)
    p5  = o5  >= float(thr_ins5)
    committees = [
        ("K1",  "INS1 + INS4",                         dict(sm="v12",     fi="4",   use_1=True, use_2=False, use_3=False, use_4=True, use_5=False)),
        ("K2",  "INS2 + INS4",                         dict(sm="v12",     fi="4",   use_1=False,use_2=True,  use_3=False, use_4=True, use_5=False)),
        ("K3",  "INS1+INS2 + INS4 (smoke voting)",     dict(sm="v12",     fi="4",   use_1=True, use_2=True,  use_3=False, use_4=True, use_5=False)),
        ("K4",  "INS1+INS2 + INS4 (smoke confirm)",    dict(sm="c12",     fi="4",   use_1=True, use_2=True,  use_3=False, use_4=True, use_5=False)),

        ("K5",  "INS1 + INS5",                         dict(sm="v12",     fi="5",   use_1=True, use_2=False, use_3=False, use_4=False,use_5=True)),
        ("K6",  "INS2 + INS5",                         dict(sm="v12",     fi="5",   use_1=False,use_2=True,  use_3=False, use_4=False,use_5=True)),
        ("K7",  "INS1+INS2 + INS5 (smoke voting)",     dict(sm="v12",     fi="5",   use_1=True, use_2=True,  use_3=False, use_4=False,use_5=True)),
        ("K8",  "INS1+INS2 + INS5 (smoke confirm)",    dict(sm="c12",     fi="5",   use_1=True, use_2=True,  use_3=False, use_4=False,use_5=True)),

        ("K9",  "INS1+INS2+INS3B + INS4 (smoke voting3)", dict(sm="v12_v3", fi="4", use_1=True,use_2=True,use_3=True,use_4=True,use_5=False)),
        ("K10", "INS1+INS2 voting + INS3B confirm + INS4", dict(sm="v12_c3", fi="4", use_1=True,use_2=True,use_3=True,use_4=True,use_5=False)),

        ("K11", "INS1+INS2+INS3B + INS5 (smoke voting3)", dict(sm="v12_v3", fi="5", use_1=True,use_2=True,use_3=True,use_4=False,use_5=True)),
        ("K12", "INS1+INS2 voting + INS3B confirm + INS5", dict(sm="v12_c3", fi="5", use_1=True,use_2=True,use_3=True,use_4=False,use_5=True)),

        ("K13", "INS1 + (INS4+INS5 fire voting)",      dict(sm="v12",     fi="v45", use_1=True,use_2=False,use_3=False,use_4=True,use_5=True)),
        ("K14", "INS2 + (INS4+INS5 fire voting)",      dict(sm="v12",     fi="v45", use_1=False,use_2=True,use_3=False,use_4=True,use_5=True)),
        ("K15", "INS1+INS2 smoke voting + fire voting",dict(sm="v12",     fi="v45", use_1=True,use_2=True,use_3=False,use_4=True,use_5=True)),
        ("K16", "INS1+INS2 smoke confirm + fire voting",dict(sm="c12",    fi="v45", use_1=True,use_2=True,use_3=False,use_4=True,use_5=True)),

        ("K17", "INS1+INS2+INS3B smoke voting3 + fire voting", dict(sm="v12_v3", fi="v45", use_1=True,use_2=True,use_3=True,use_4=True,use_5=True)),
        ("K18", "INS1+INS2 voting + INS3B confirm + fire voting", dict(sm="v12_c3", fi="v45", use_1=True,use_2=True,use_3=True,use_4=True,use_5=True)),
    ]

    rows = []

    y_true3 = make_3class_pred(y_fi, y_sm)

    for kid, desc, cfg in committees:
        pp1  = p1  if cfg["use_1"] else torch.zeros_like(p1)
        pp2  = p2  if cfg["use_2"] else torch.zeros_like(p2)
        pp3b = p3b if cfg["use_3"] else None

        if cfg["use_3"] is False:
            pred_sm = smoke_committee_pred(cfg["sm"], pp1, pp2, p3b=None)
        else:
            pred_sm = smoke_committee_pred(cfg["sm"], pp1, pp2, pp3b)

        pp4 = p4 if cfg["use_4"] else torch.zeros_like(p4)
        pp5 = p5 if cfg["use_5"] else torch.zeros_like(p5)
        pred_fi = fire_committee_pred(cfg["fi"], pp4, pp5)

        pred_hz = pred_sm | pred_fi

        y_pred3 = make_3class_pred(pred_fi, pred_sm)

        rep = {"id": kid, "desc": desc}

        # smoke
        rep_sm_g = binary_metrics_from_pred(pred_sm, y_sm)
        rep_sm_l = binary_metrics_mean_per_image_from_offsets(h5_test_path, pred_sm, y_sm)
        rep_sm_i = image_detection_rate_from_offsets(h5_test_path, pred_sm, y_sm, thresholds=img_thresholds)

        # fire
        rep_fi_g = binary_metrics_from_pred(pred_fi, y_fi)
        rep_fi_l = binary_metrics_mean_per_image_from_offsets(h5_test_path, pred_fi, y_fi)
        rep_fi_i = image_detection_rate_from_offsets(h5_test_path, pred_fi, y_fi, thresholds=img_thresholds)

        # hazard
        rep_hz_g = binary_metrics_from_pred(pred_hz, y_hz)
        rep_hz_l = binary_metrics_mean_per_image_from_offsets(h5_test_path, pred_hz, y_hz)
        rep_hz_i = image_detection_rate_from_offsets(h5_test_path, pred_hz, y_hz, thresholds=img_thresholds)

        # pack
        for k,v in rep_sm_g.items(): rep[f"smoke_global_{k}"] = v
        for k,v in rep_sm_l.items(): rep[f"smoke_mean_{k}"] = v
        for k,v in rep_sm_i.items(): rep[f"smoke_{k}"] = v

        for k,v in rep_fi_g.items(): rep[f"fire_global_{k}"] = v
        for k,v in rep_fi_l.items(): rep[f"fire_mean_{k}"] = v
        for k,v in rep_fi_i.items(): rep[f"fire_{k}"] = v

        for k,v in rep_hz_g.items(): rep[f"hazard_global_{k}"] = v
        for k,v in rep_hz_l.items(): rep[f"hazard_mean_{k}"] = v
        for k,v in rep_hz_i.items(): rep[f"hazard_{k}"] = v

        # -------- metrics: 3-class --------
        cm = confusion_3x3(y_true3, y_pred3)
        rep3 = prf_from_cm_3class(cm)
        for k,v in rep3.items(): rep[f"cls3_{k}"] = v
        rep["cls3_cm_00"] = int(cm[0,0].item()); rep["cls3_cm_01"] = int(cm[0,1].item()); rep["cls3_cm_02"] = int(cm[0,2].item())
        rep["cls3_cm_10"] = int(cm[1,0].item()); rep["cls3_cm_11"] = int(cm[1,1].item()); rep["cls3_cm_12"] = int(cm[1,2].item())
        rep["cls3_cm_20"] = int(cm[2,0].item()); rep["cls3_cm_21"] = int(cm[2,1].item()); rep["cls3_cm_22"] = int(cm[2,2].item())

        save_payload = {
            "id": kid,
            "desc": desc,
            "thr": dict(ins1=thr_ins1, ins2=thr_ins2, ins3B=thr_ins3B, ins4=thr_ins4, ins5=thr_ins5),
            "pred_smoke_u8": pred_sm.to(torch.uint8),
            "pred_fire_u8":  pred_fi.to(torch.uint8),
            "pred_hazard_u8": pred_hz.to(torch.uint8),
            "pred_3class_u8": y_pred3.to(torch.uint8),
        }
        torch.save(save_payload, os.path.join(save_dir, f"{kid}_preds.pt"))

        rows.append(rep)

        print(f"{kid}: smokeP={rep_sm_g['P']:.4f} fireP={rep_fi_g['P']:.4f} hazP={rep_hz_g['P']:.4f} | cls3 microF1={rep3['micro_F1_fire_smoke']:.4f}")

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(save_dir, "metrics_K_committees.xlsx"), index=False)
    print("Saved:", os.path.join(save_dir, "metrics_K_committees.xlsx"))
    return df

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def ytrue_3class_from_pm1(y_fire_pm1, y_smoke_pm1):
    # fire priority
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    y_fire_pm1 :
        Параметр используется в соответствии с назначением функции.
    y_smoke_pm1 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `ytrue_3class_from_pm1`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = ytrue_3class_from_pm1(y_fire_pm1=..., y_smoke_pm1=...)
    """
    fire_t = (y_fire_pm1 > 0)
    smoke_t = (y_smoke_pm1 > 0) & (~fire_t)
    y3 = torch.zeros_like(y_fire_pm1, dtype=torch.int64)
    y3[smoke_t] = 1
    y3[fire_t]  = 2
    return y3

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def confusion_3x3_fast(y_true3, y_pred3):
    # y_true3/y_pred3: int64 (N,)
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    y_true3 :
        Параметр используется в соответствии с назначением функции.
    y_pred3 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `confusion_3x3_fast`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = confusion_3x3_fast(y_true3=..., y_pred3=...)
    """
    cm = torch.zeros((3,3), dtype=torch.int64, device=y_true3.device)
    for t in range(3):
        mt = (y_true3 == t)
        for p in range(3):
            cm[t,p] = (mt & (y_pred3 == p)).sum()
    return cm

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def prf_from_cm_3class(cm):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    cm :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `prf_from_cm_3class`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = prf_from_cm_3class(cm=...)
    """
    out = {}
    cmf = cm.to(torch.float64)

    def one(cls, name):
        tp = cmf[cls, cls]
        fp = cmf[:, cls].sum() - tp
        fn = cmf[cls, :].sum() - tp
        P  = float(tp / (tp + fp + 1e-12))
        R  = float(tp / (tp + fn + 1e-12))
        F1 = float((2*P*R) / (P+R+1e-12))
        out[f"{name}_P"] = P
        out[f"{name}_R"] = R
        out[f"{name}_F1"] = F1

    one(0,"non")
    one(1,"smoke")
    one(2,"fire")

    # micro + macro over smoke+fire
    cls_set = [1,2]
    TP = sum(cmf[c,c] for c in cls_set)
    FP = sum(cmf[:,c].sum() - cmf[c,c] for c in cls_set)
    FN = sum(cmf[c,:].sum() - cmf[c,c] for c in cls_set)
    Pm = float(TP / (TP + FP + 1e-12))
    Rm = float(TP / (TP + FN + 1e-12))
    F1m = float((2*Pm*Rm) / (Pm+Rm+1e-12))
    out["micro_P_fire_smoke"]  = Pm
    out["micro_R_fire_smoke"]  = Rm
    out["micro_F1_fire_smoke"] = F1m

    out["macro_P_fire_smoke"]  = 0.5*(out["smoke_P"] + out["fire_P"])
    out["macro_R_fire_smoke"]  = 0.5*(out["smoke_R"] + out["fire_R"])
    out["macro_F1_fire_smoke"] = 0.5*(out["smoke_F1"] + out["fire_F1"])

    # overall acc
    acc = float(cmf.trace() / (cmf.sum() + 1e-12))
    out["acc_3class"] = acc
    return out

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def hazard_metrics_from_bool(pred_h, true_h):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    pred_h :
        Параметр используется в соответствии с назначением функции.
    true_h :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `hazard_metrics_from_bool`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = hazard_metrics_from_bool(pred_h=..., true_h=...)
    """
    tp = (pred_h & true_h).sum().item()
    fp = (pred_h & (~true_h)).sum().item()
    fn = ((~pred_h) & true_h).sum().item()
    tn = ((~pred_h) & (~true_h)).sum().item()
    P  = tp / (tp + fp + 1e-12)
    R  = tp / (tp + fn + 1e-12)
    F1 = (2*P*R) / (P+R+1e-12)
    acc = (tp+tn) / (tp+tn+fp+fn+1e-12)
    tnr = tn / (tn+fp+1e-12)
    bal = 0.5*(R + tnr)
    return dict(P=P,R=R,F1=F1,acc=acc,bal_acc=bal,tp=tp,fp=fp,fn=fn,tn=tn)

# Назначение: считывает данные из внешнего источника.
def _read_offsets(h5_test_path):
    """
    Считывает данные из внешнего источника.

    Параметры
    ---------
    h5_test_path :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_read_offsets`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _read_offsets(h5_test_path='путь/к/ресурсу')
    """
    with h5py.File(h5_test_path, "r") as h5:
        starts = h5["meta/sample_start"][:].astype(np.int64)
        counts = h5["meta/sample_count"][:].astype(np.int64)
    return starts, counts

@torch.no_grad()
# Назначение: вычисляет смещение по заданным параметрам.
def mean_per_image_prf_binary_from_offsets(pred_pos, true_pos, starts, counts):
    """
    Вычисляет смещение по заданным параметрам.

    Параметры
    ---------
    pred_pos :
        Параметр используется в соответствии с назначением функции.
    true_pos :
        Параметр используется в соответствии с назначением функции.
    starts :
        Параметр используется в соответствии с назначением функции.
    counts :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `mean_per_image_prf_binary_from_offsets`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = mean_per_image_prf_binary_from_offsets(pred_pos=..., true_pos=..., starts=...)
    """
    P_list, R_list, F1_list = [], [], []
    n_imgs = len(starts)
    for i in range(n_imgs):
        s = int(starts[i]); c = int(counts[i])
        if c <= 0:
            continue
        sl = slice(s, s+c)
        pp = pred_pos[sl]
        tp = (pp & true_pos[sl]).sum().item()
        fp = (pp & (~true_pos[sl])).sum().item()
        fn = ((~pp) & true_pos[sl]).sum().item()
        P = tp/(tp+fp+1e-12)
        R = tp/(tp+fn+1e-12)
        F1= (2*P*R)/(P+R+1e-12)
        P_list.append(P); R_list.append(R); F1_list.append(F1)
    if len(P_list)==0:
        return dict(P_mean=0.0,R_mean=0.0,F1_mean=0.0,n_images_used=0)
    return dict(
        P_mean=float(np.mean(P_list)),
        R_mean=float(np.mean(R_list)),
        F1_mean=float(np.mean(F1_list)),
        n_images_used=int(len(P_list))
    )

@torch.no_grad()
# Назначение: вычисляет смещение по заданным параметрам.
def mean_per_image_prf_3class_from_offsets(y_true3, y_pred3, starts, counts):
    """
    Вычисляет смещение по заданным параметрам.

    Параметры
    ---------
    y_true3 :
        Параметр используется в соответствии с назначением функции.
    y_pred3 :
        Параметр используется в соответствии с назначением функции.
    starts :
        Параметр используется в соответствии с назначением функции.
    counts :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `mean_per_image_prf_3class_from_offsets`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = mean_per_image_prf_3class_from_offsets(y_true3=..., y_pred3=..., starts=...)
    """
    P_list, R_list, F1_list = [], [], []
    n_imgs = len(starts)
    for i in range(n_imgs):
        s = int(starts[i]); c = int(counts[i])
        if c <= 0:
            continue
        sl = slice(s, s+c)
        yt = y_true3[sl]
        yp = y_pred3[sl]

        true_h = (yt > 0)
        pred_h = (yp > 0)
        m = hazard_metrics_from_bool(pred_h, true_h)
        P_list.append(m["P"]); R_list.append(m["R"]); F1_list.append(m["F1"])
    if len(P_list)==0:
        return dict(P_mean=0.0,R_mean=0.0,F1_mean=0.0,n_images_used=0)
    return dict(
        P_mean=float(np.mean(P_list)),
        R_mean=float(np.mean(R_list)),
        F1_mean=float(np.mean(F1_list)),
        n_images_used=int(len(P_list))
    )

@torch.no_grad()
# Назначение: выполняет обработку признаков или меток, относящихся к огню.
def found_fire_percent_by_gt_recall_thresholds(
    pred_h, true_h, starts, counts, thresholds=(0.02,0.05,0.07,0.10)
):
    """
    Выполняет обработку признаков или меток, относящихся к огню.

    Параметры
    ---------
    pred_h :
        Параметр используется в соответствии с назначением функции.
    true_h :
        Параметр используется в соответствии с назначением функции.
    starts :
        Параметр используется в соответствии с назначением функции.
    counts :
        Параметр используется в соответствии с назначением функции.
    thresholds :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `found_fire_percent_by_gt_recall_thresholds`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = found_fire_percent_by_gt_recall_thresholds(pred_h=..., true_h=..., starts=...)
    """
    thr_list = list(thresholds)
    ok = {t:0 for t in thr_list}
    denom = 0
    n_imgs = len(starts)

    for i in range(n_imgs):
        s = int(starts[i]); c = int(counts[i])
        if c <= 0:
            continue
        sl = slice(s, s+c)
        gt = true_h[sl]
        gt_cnt = gt.sum().item()
        if gt_cnt <= 0:
            continue
        denom += 1
        tp_cnt = (pred_h[sl] & gt).sum().item()
        rec_img = tp_cnt / (gt_cnt + 1e-12)
        for t in thr_list:
            if rec_img >= t:
                ok[t] += 1

    rep = {"n_images_gt_hazard": int(denom)}
    for t in thr_list:
        rep[f"found_hazard@gtRecall>={t:.2f}"] = (ok[t] / max(denom,1))
    return rep


# Назначение: определяет или применяет логику комитета классификаторов.
def define_committees():
    """
    Определяет или применяет логику комитета классификаторов.

    Параметры
    ---------
    Функция не принимает обязательных пользовательских параметров.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `define_committees`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = define_committees()
    """
    C = []

    # helpers inside rules
    def pos(out, thr):  # bool (N,)
        o = out[:,0] if out.ndim==2 else out
        return o >= float(thr)

    # --- smoke ensembles ---
    # strict: (ins1 AND ins2) OR ins3   (ins3 "rescue" для ошибок)
    def smoke_rescue_A(outs, thr):
        s12 = pos(outs["ins1"],thr["ins1"]) & pos(outs["ins2"],thr["ins2"])
        s3  = pos(outs["ins3A"],thr["ins3A"])
        return s12 | s3

    def smoke_rescue_B(outs, thr):
        s12 = pos(outs["ins1"],thr["ins1"]) & pos(outs["ins2"],thr["ins2"])
        s3  = pos(outs["ins3B"],thr["ins3B"])
        return s12 | s3

    # lenient: OR всех smoke
    def smoke_or_12(outs, thr):
        return pos(outs["ins1"],thr["ins1"]) | pos(outs["ins2"],thr["ins2"])

    # --- fire ensembles ---
    def fire_or_45(outs, thr):
        return pos(outs["ins4"],thr["ins4"]) | pos(outs["ins5"],thr["ins5"])

    def fire_and_45(outs, thr):
        return pos(outs["ins4"],thr["ins4"]) & pos(outs["ins5"],thr["ins5"])

    # Committees you listed (examples; легко расширять)
    C.append(dict(
        name="C1 ins1+ins4 (OR single)",
        uses=["ins1","ins4"],
        rule=lambda outs,thr: (pos(outs["ins1"],thr["ins1"]), pos(outs["ins4"],thr["ins4"]))
    ))
    C.append(dict(
        name="C2 ins2+ins4",
        uses=["ins2","ins4"],
        rule=lambda outs,thr: (pos(outs["ins2"],thr["ins2"]), pos(outs["ins4"],thr["ins4"]))
    ))
    C.append(dict(
        name="C3 ins1+ins2+ins4 (smoke OR12, fire ins4)",
        uses=["ins1","ins2","ins4"],
        rule=lambda outs,thr: (smoke_or_12(outs,thr), pos(outs["ins4"],thr["ins4"]))
    ))
    C.append(dict(
        name="C4 ins1+ins2+ins4+ins5 (smoke OR12, fire OR45)",
        uses=["ins1","ins2","ins4","ins5"],
        rule=lambda outs,thr: (smoke_or_12(outs,thr), fire_or_45(outs,thr))
    ))
    C.append(dict(
        name="C5 ins1+ins2+ins3A+ins4 (smoke rescueA, fire ins4)",
        uses=["ins1","ins2","ins3A","ins4"],
        rule=lambda outs,thr: (smoke_rescue_A(outs,thr), pos(outs["ins4"],thr["ins4"]))
    ))
    C.append(dict(
        name="C6 ins1+ins2+ins3B+ins4 (smoke rescueB, fire ins4)",
        uses=["ins1","ins2","ins3B","ins4"],
        rule=lambda outs,thr: (smoke_rescue_B(outs,thr), pos(outs["ins4"],thr["ins4"]))
    ))
    C.append(dict(
        name="C7 ins1+ins2+ins3A+ins4+ins5 (smoke rescueA, fire OR45)",
        uses=["ins1","ins2","ins3A","ins4","ins5"],
        rule=lambda outs,thr: (smoke_rescue_A(outs,thr), fire_or_45(outs,thr))
    ))
    C.append(dict(
        name="C8 ins1+ins2+ins3B+ins4+ins5 (smoke rescueB, fire OR45)",
        uses=["ins1","ins2","ins3B","ins4","ins5"],
        rule=lambda outs,thr: (smoke_rescue_B(outs,thr), fire_or_45(outs,thr))
    ))

    # варианты "точность fire выше": fire AND45
    C.append(dict(
        name="C9 ins1+ins2+ins3A+ins4+ins5 (smoke rescueA, fire AND45)",
        uses=["ins1","ins2","ins3A","ins4","ins5"],
        rule=lambda outs,thr: (smoke_rescue_A(outs,thr), fire_and_45(outs,thr))
    ))
    return C


# Назначение: определяет или применяет логику комитета классификаторов.
def run_committees_and_save(
    h5_test_path: str,
    out_save_path: str,
    outs: dict,      # e.g. {"ins1":out_ins1, ...}  all on GPU
    thrs: dict,      # e.g. {"ins1":0.98, ...}
    y_fire_pm1: torch.Tensor,
    y_smoke_pm1: torch.Tensor,
    thresholds_gt_recall=(0.02,0.05,0.07,0.10),
):
    """
    Определяет или применяет логику комитета классификаторов.

    Параметры
    ---------
    h5_test_path :
        Параметр используется в соответствии с назначением функции.
    out_save_path :
        Параметр используется в соответствии с назначением функции.
    outs :
        Параметр используется в соответствии с назначением функции.
    thrs :
        Параметр используется в соответствии с назначением функции.
    y_fire_pm1 :
        Параметр используется в соответствии с назначением функции.
    y_smoke_pm1 :
        Параметр используется в соответствии с назначением функции.
    thresholds_gt_recall :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `run_committees_and_save`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = run_committees_and_save(h5_test_path='путь/к/ресурсу', out_save_path='путь/к/ресурсу', outs=...)
    """
    os.makedirs(os.path.dirname(out_save_path), exist_ok=True)

    starts, counts = _read_offsets(h5_test_path)
    starts_t = starts  # np
    counts_t = counts

    y_true3 = ytrue_3class_from_pm1(y_fire_pm1, y_smoke_pm1)
    true_h  = (y_true3 > 0)

    committees = define_committees()

    with h5py.File(out_save_path, "w") as h5:
        h5.attrs["source_test_h5"] = str(h5_test_path)
        h5.create_dataset("meta/sample_start", data=starts_t, dtype=np.int64)
        h5.create_dataset("meta/sample_count", data=counts_t, dtype=np.int64)

        h5.create_dataset("y_true3", data=y_true3.detach().cpu().to(torch.int8).numpy(), compression="lzf")

        for c in committees:
            name = c["name"]
            key  = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

            outs_c = {k: outs[k] for k in c["uses"]}
            thrs_c = {k: thrs[k] for k in c["uses"]}

            smoke_pos, fire_pos = c["rule"](outs_c, thrs_c)

            y3 = torch.zeros_like(smoke_pos, dtype=torch.int64)
            y3[smoke_pos] = 1
            y3[fire_pos]  = 2

            h5.create_dataset(f"committees/{key}/y3_pred",
                              data=y3.detach().cpu().to(torch.uint8).numpy(),
                              compression="lzf")
            # сохранение использованных трешхолдов
            for k2,v2 in thrs_c.items():
                h5[f"committees/{key}"].attrs[f"thr_{k2}"] = float(v2)

    print("Saved committee preds to:", out_save_path)
# Назначение: определяет или применяет логику комитета классификаторов.
def eval_committees_from_saved(
    saved_committees_h5: str,
    thresholds_gt_recall=(0.02,0.05,0.07,0.10),
):
    """
    Определяет или применяет логику комитета классификаторов.

    Параметры
    ---------
    saved_committees_h5 :
        Параметр используется в соответствии с назначением функции.
    thresholds_gt_recall :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `eval_committees_from_saved`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = eval_committees_from_saved(saved_committees_h5=..., thresholds_gt_recall=0.5)
    """
    results = []

    with h5py.File(saved_committees_h5, "r") as h5:
        starts = h5["meta/sample_start"][:].astype(np.int64)
        counts = h5["meta/sample_count"][:].astype(np.int64)

        y_true3 = torch.from_numpy(h5["y_true3"][:].astype(np.int64))  # CPU tensor ok
        true_h  = (y_true3 > 0)

        # список комитетов
        comm_keys = list(h5["committees"].keys())

        for key in comm_keys:
            y3 = torch.from_numpy(h5[f"committees/{key}/y3_pred"][:].astype(np.int64))
            pred_h = (y3 > 0)

            cm = confusion_3x3_fast(y_true3, y3)
            rep3 = prf_from_cm_3class(cm)

            repH = hazard_metrics_from_bool(pred_h, true_h)

            rep3_mean = mean_per_image_prf_3class_from_offsets(y_true3, y3, starts, counts)
            repH_mean = mean_per_image_prf_binary_from_offsets(pred_h, true_h, starts, counts)

            repFound = found_fire_percent_by_gt_recall_thresholds(
                pred_h, true_h, starts, counts, thresholds=thresholds_gt_recall
            )

            rep = {
                "committee": key,
                "n_samples": int(y3.numel()),
            }

            for k,v in rep3.items():     rep[f"3c_global_{k}"] = v
            for k,v in repH.items():     rep[f"haz_global_{k}"] = v
            for k,v in rep3_mean.items():rep[f"haz_mean_from3_{k}"] = v   # это mean per image по hazard из 3-класс pred
            for k,v in repH_mean.items():rep[f"haz_mean_{k}"] = v
            for k,v in repFound.items(): rep[f"img_{k}"] = v

            grp = h5[f"committees/{key}"]
            for a in grp.attrs:
                rep[a] = grp.attrs[a]

            results.append(rep)

    df = pd.DataFrame(results).sort_values(by=["haz_global_P", "haz_global_R"], ascending=False)
    return df
# ----------------------------
# voting primitives
# ----------------------------
@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def pos_from_out(out, thr_out: float):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    out :
        Параметр используется в соответствии с назначением функции.
    thr_out :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `pos_from_out`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = pos_from_out(out=..., thr_out=0.5)
    """
    if out.ndim == 2: out = out[:, 0]
    return out >= float(thr_out)

@torch.no_grad()
# Назначение: реализует процедуру голосования нескольких классификаторов.
def majority_vote(pos_list, k=None):
    """
    Реализует процедуру голосования нескольких классификаторов.

    Параметры
    ---------
    pos_list :
        Параметр используется в соответствии с назначением функции.
    k :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `majority_vote`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = majority_vote(pos_list=..., k=...)
    """
    assert len(pos_list) >= 1
    n = len(pos_list)
    if k is None:
        k = (n // 2) + 1
    s = torch.zeros_like(pos_list[0], dtype=torch.int16)
    for p in pos_list:
        s += p.to(torch.int16)
    return s >= int(k)

@torch.no_grad()
# Назначение: реализует процедуру голосования нескольких классификаторов.
def weighted_vote(pos_list, weights, thr_weight=None):
    """
    Реализует процедуру голосования нескольких классификаторов.

    Параметры
    ---------
    pos_list :
        Параметр используется в соответствии с назначением функции.
    weights :
        Параметр используется в соответствии с назначением функции.
    thr_weight :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `weighted_vote`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = weighted_vote(pos_list=..., weights=..., thr_weight=0.5)
    """
    assert len(pos_list) == len(weights)
    wsum = float(sum(weights))
    if thr_weight is None:
        thr_weight = 0.5 * wsum
    acc = torch.zeros_like(pos_list[0], dtype=torch.float32)
    for p, w in zip(pos_list, weights):
        acc += (p.float() * float(w))
    return acc >= float(thr_weight)

# ----------------------------
# committees (voting)
# ----------------------------
# Назначение: определяет или применяет логику комитета классификаторов.
def define_committees_voting():
    """
    Определяет или применяет логику комитета классификаторов.

    Параметры
    ---------
    Функция не принимает обязательных пользовательских параметров.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `define_committees_voting`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = define_committees_voting()
    """
    C = []

    # --- Smoke voters ---
    def smoke_votes_12(outs, thrs):
        return [
            pos_from_out(outs["ins1"], thrs["ins1"]),
            pos_from_out(outs["ins2"], thrs["ins2"]),
        ]

    def smoke_votes_123A(outs, thrs):
        return [
            pos_from_out(outs["ins1"], thrs["ins1"]),
            pos_from_out(outs["ins2"], thrs["ins2"]),
            pos_from_out(outs["ins3A"], thrs["ins3A"]),
        ]

    def smoke_votes_123B(outs, thrs):
        return [
            pos_from_out(outs["ins1"], thrs["ins1"]),
            pos_from_out(outs["ins2"], thrs["ins2"]),
            pos_from_out(outs["ins3B"], thrs["ins3B"]),
        ]

    # --- Fire voters ---
    def fire_votes_45(outs, thrs):
        return [
            pos_from_out(outs["ins4"], thrs["ins4"]),
            pos_from_out(outs["ins5"], thrs["ins5"]),
        ]

    # ---------- Voting committees ----------
    # 1) Smoke majority of (1,2,3A), Fire single (ins4)
    C.append(dict(
        name="V1 smoke maj(1,2,3A), fire ins4",
        uses=["ins1","ins2","ins3A","ins4"],
        rule=lambda outs, thrs: (
            majority_vote(smoke_votes_123A(outs,thrs), k=2),  # 2/3
            pos_from_out(outs["ins4"], thrs["ins4"])
        )
    ))

    # 2) Smoke weighted(1,2,3A) (emphasize 1&2), Fire OR(4,5)
    C.append(dict(
        name="V2 smoke wvote(1,2,3A w=[1,1,0.7]), fire OR(4,5)",
        uses=["ins1","ins2","ins3A","ins4","ins5"],
        rule=lambda outs, thrs: (
            weighted_vote(smoke_votes_123A(outs,thrs), weights=[1.0,1.0,0.7]),
            pos_from_out(outs["ins4"], thrs["ins4"]) | pos_from_out(outs["ins5"], thrs["ins5"])
        )
    ))

    # 3) Smoke maj(1,2,3B), Fire OR(4,5)
    C.append(dict(
        name="V3 smoke maj(1,2,3B), fire OR(4,5)",
        uses=["ins1","ins2","ins3B","ins4","ins5"],
        rule=lambda outs, thrs: (
            majority_vote(smoke_votes_123B(outs,thrs), k=2),
            pos_from_out(outs["ins4"], thrs["ins4"]) | pos_from_out(outs["ins5"], thrs["ins5"])
        )
    ))

    # 4) FIRE precision-first: fire AND(4,5), smoke maj(1,2,3A)
    C.append(dict(
        name="V4 FIRE-prec-first: smoke maj(1,2,3A), fire AND(4,5)",
        uses=["ins1","ins2","ins3A","ins4","ins5"],
        rule=lambda outs, thrs: (
            majority_vote(smoke_votes_123A(outs,thrs), k=2),
            pos_from_out(outs["ins4"], thrs["ins4"]) & pos_from_out(outs["ins5"], thrs["ins5"])
        )
    ))

    # 5) Ultra-prec smoke: AND(1,2) then rescue by 3A (как раньше), fire AND(4,5)
    C.append(dict(
        name="V5 strict smoke ((1&2)|3A), strict fire (4&5)",
        uses=["ins1","ins2","ins3A","ins4","ins5"],
        rule=lambda outs, thrs: (
            (pos_from_out(outs["ins1"], thrs["ins1"]) & pos_from_out(outs["ins2"], thrs["ins2"])) |
            (pos_from_out(outs["ins3A"], thrs["ins3A"])),
            pos_from_out(outs["ins4"], thrs["ins4"]) & pos_from_out(outs["ins5"], thrs["ins5"])
        )
    ))

    # 6) Smoke only 1&2 majority (k=2 => AND), fire OR(4,5)
    C.append(dict(
        name="V6 smoke AND(1,2), fire OR(4,5)",
        uses=["ins1","ins2","ins4","ins5"],
        rule=lambda outs, thrs: (
            majority_vote(smoke_votes_12(outs,thrs), k=2),
            pos_from_out(outs["ins4"], thrs["ins4"]) | pos_from_out(outs["ins5"], thrs["ins5"])
        )
    ))

    return C

# Назначение: определяет или применяет логику комитета классификаторов.
def run_voting_committees_and_save(
    h5_test_path: str,
    out_save_path: str,
    outs: dict,   # {"ins1":out_ins1, ...} all GPU (N,1)
    thrs: dict,   # {"ins1":THR_INS1, ...}
    y_fire_pm1: torch.Tensor, # добавление y_fire_pm1
    y_smoke_pm1: torch.Tensor, # добавление y_smoke_pm1
):
    """
    Определяет или применяет логику комитета классификаторов.

    Параметры
    ---------
    h5_test_path :
        Параметр используется в соответствии с назначением функции.
    out_save_path :
        Параметр используется в соответствии с назначением функции.
    outs :
        Параметр используется в соответствии с назначением функции.
    thrs :
        Параметр используется в соответствии с назначением функции.
    y_fire_pm1 :
        Параметр используется в соответствии с назначением функции.
    y_smoke_pm1 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `run_voting_committees_and_save`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = run_voting_committees_and_save(h5_test_path='путь/к/ресурсу', out_save_path='путь/к/ресурсу', outs=...)
    """
    os.makedirs(os.path.dirname(out_save_path), exist_ok=True)

    with h5py.File(h5_test_path, "r") as h5src:
        starts = h5src["meta/sample_start"][:].astype(np.int64)
        counts = h5src["meta/sample_count"][:].astype(np.int64)

    committees = define_committees_voting()

    with h5py.File(out_save_path, "w") as h5:
        h5.attrs["source_test_h5"] = str(h5_test_path)
        h5.create_dataset("meta/sample_start", data=starts, dtype=np.int64)
        h5.create_dataset("meta/sample_count", data=counts, dtype=np.int64)

        # Calculate and save y_true3 (ground truth)
        y_true3 = ytrue_3class_from_pm1(y_fire_pm1, y_smoke_pm1)
        h5.create_dataset("y_true3", data=y_true3.detach().cpu().to(torch.int8).numpy(), compression="lzf")

        for c in committees:
            name = c["name"]
            key  = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace(":", "")

            outs_c = {k: outs[k] for k in c["uses"]}
            thrs_c = {k: thrs[k] for k in c["uses"]}

            smoke_pos, fire_pos = c["rule"](outs_c, thrs_c)

            # y3 pred with fire priority
            y3 = torch.zeros_like(smoke_pos, dtype=torch.int64)
            y3[smoke_pos] = 1
            y3[fire_pos]  = 2

            h5.create_dataset(
                f"committees/{key}/y3_pred",
                data=y3.detach().cpu().to(torch.uint8).numpy(),
                compression="lzf"
            )
            # save thresholds as attrs
            grp = h5[f"committees/{key}"]
            grp.attrs["committee_name"] = name
            for kk, vv in thrs_c.items():
                grp.attrs[f"thr_{kk}"] = float(vv)

    print("Saved voting committee preds to:", out_save_path)
    
# Назначение: определяет или применяет логику комитета классификаторов.
def define_committees_voting():
    """
    Определяет или применяет логику комитета классификаторов.

    Параметры
    ---------
    Функция не принимает обязательных пользовательских параметров.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `define_committees_voting`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = define_committees_voting()
    """
    C = []

    # ---------- helpers ----------
    def b(outs, thrs, k):  # bool from out>=thr
        return pos_from_out(outs[k], thrs[k])

    # Smoke voters
    def smoke_votes_12(outs, thrs):
        return [b(outs, thrs, "ins1"), b(outs, thrs, "ins2")]

    def smoke_votes_123A(outs, thrs):
        return [b(outs, thrs, "ins1"), b(outs, thrs, "ins2"), b(outs, thrs, "ins3A")]

    def smoke_votes_123B(outs, thrs):
        return [b(outs, thrs, "ins1"), b(outs, thrs, "ins2"), b(outs, thrs, "ins3B")]

    # Fire voters
    def fire_votes_45(outs, thrs):
        return [b(outs, thrs, "ins4"), b(outs, thrs, "ins5")]

    # =========================================================
    # 0) базовые 
    # =========================================================
    C.append(dict(
        name="V1 smoke maj(1,2,3A), fire ins4",
        uses=["ins1","ins2","ins3A","ins4"],
        rule=lambda outs, thrs: (
            majority_vote(smoke_votes_123A(outs,thrs), k=2),  # 2/3
            b(outs, thrs, "ins4")
        )
    ))
    C.append(dict(
        name="V3 smoke maj(1,2,3B), fire OR(4,5)",
        uses=["ins1","ins2","ins3B","ins4","ins5"],
        rule=lambda outs, thrs: (
            majority_vote(smoke_votes_123B(outs,thrs), k=2),
            b(outs, thrs, "ins4") | b(outs, thrs, "ins5")
        )
    ))
    C.append(dict(
        name="V4 FIRE-prec-first: smoke maj(1,2,3A), fire AND(4,5)",
        uses=["ins1","ins2","ins3A","ins4","ins5"],
        rule=lambda outs, thrs: (
            majority_vote(smoke_votes_123A(outs,thrs), k=2),
            b(outs, thrs, "ins4") & b(outs, thrs, "ins5")
        )
    ))

    # =========================================================
    # 1) Добавляем аналоги A->B для тех же логик
    # =========================================================
    C.append(dict(
        name="V4B FIRE-prec-first: smoke maj(1,2,3B), fire AND(4,5)",
        uses=["ins1","ins2","ins3B","ins4","ins5"],
        rule=lambda outs, thrs: (
            majority_vote(smoke_votes_123B(outs,thrs), k=2),
            b(outs, thrs, "ins4") & b(outs, thrs, "ins5")
        )
    ))

    C.append(dict(
        name="V2B smoke wvote(1,2,3B w=[1,1,0.7]), fire OR(4,5)",
        uses=["ins1","ins2","ins3B","ins4","ins5"],
        rule=lambda outs, thrs: (
            weighted_vote(smoke_votes_123B(outs,thrs), weights=[1.0,1.0,0.7]),
            b(outs, thrs, "ins4") | b(outs, thrs, "ins5")
        )
    ))

    # =========================================================
    # 2) “CONFIRMATION” режимы (не voting, не rescue)
    # =========================================================
    # Идея:
    #   базовая модель даёт кандидатов (где потенциально smoke/fire),
    #   подтверждаюзая подтверждает их. Это снижает FP.
    #
    # для дыма:
    #   вариант 1: (ins1 или ins2) подтверждение от ins3B
    #   вариант 2: ins2 подтверждение от ins3B (если ins2 сильнее / меньше FP)
    # для огня:
    #   ins4 подтверждение от  ins5  (или наоборот)
    # =========================================================

    # C1: smoke = (1|2) & 3B ; fire = 4 & 5
    C.append(dict(
        name="C1 CONFIRM: smoke=(1 OR 2) AND 3B, fire=4 AND 5",
        uses=["ins1","ins2","ins3B","ins4","ins5"],
        rule=lambda outs, thrs: (
            (b(outs, thrs, "ins1") | b(outs, thrs, "ins2")) & b(outs, thrs, "ins3B"),
            b(outs, thrs, "ins4") & b(outs, thrs, "ins5")
        )
    ))

    # C2: smoke = ins2 & 3B ; fire = 4 & 5
    C.append(dict(
        name="C2 CONFIRM: smoke=2 AND 3B, fire=4 AND 5",
        uses=["ins2","ins3B","ins4","ins5"],
        rule=lambda outs, thrs: (
            b(outs, thrs, "ins2") & b(outs, thrs, "ins3B"),
            b(outs, thrs, "ins4") & b(outs, thrs, "ins5")
        )
    ))

    # C3: smoke = ins1 & 3B ; fire = 4 & 5
    C.append(dict(
        name="C3 CONFIRM: smoke=1 AND 3B, fire=4 AND 5",
        uses=["ins1","ins3B","ins4","ins5"],
        rule=lambda outs, thrs: (
            b(outs, thrs, "ins1") & b(outs, thrs, "ins3B"),
            b(outs, thrs, "ins4") & b(outs, thrs, "ins5")
        )
    ))

    # C4: fire confirm asymmetry (если хочешь)
    # fire = 5 & 4 (то же самое, но полезно если у них разные thr и ты хочешь "base->confirm" семантику)
    C.append(dict(
        name="C4 CONFIRM: smoke maj(1,2,3B), fire=5 AND 4",
        uses=["ins1","ins2","ins3B","ins4","ins5"],
        rule=lambda outs, thrs: (
            majority_vote(smoke_votes_123B(outs,thrs), k=2),
            b(outs, thrs, "ins5") & b(outs, thrs, "ins4")
        )
    ))

    # =========================================================
    # 3) Ещё строгие “anti-FP” варианты со smoke AND + confirm
    # =========================================================
    # S1: smoke=(1&2) AND 3B  (очень строго)
    C.append(dict(
        name="S1 STRICT+CONFIRM: smoke=(1 AND 2) AND 3B, fire=4 AND 5",
        uses=["ins1","ins2","ins3B","ins4","ins5"],
        rule=lambda outs, thrs: (
            (b(outs, thrs, "ins1") & b(outs, thrs, "ins2")) & b(outs, thrs, "ins3B"),
            b(outs, thrs, "ins4") & b(outs, thrs, "ins5")
        )
    ))

    # S2: smoke=maj(1,2,3B) AND 3B  (почти то же, но чтобы явная "подтверждалка" оставалась)
    C.append(dict(
        name="S2 VOTE+CONFIRM: smoke=maj(1,2,3B) AND 3B, fire=4 AND 5",
        uses=["ins1","ins2","ins3B","ins4","ins5"],
        rule=lambda outs, thrs: (
            majority_vote(smoke_votes_123B(outs,thrs), k=2) & b(outs, thrs, "ins3B"),
            b(outs, thrs, "ins4") & b(outs, thrs, "ins5")
        )
    ))

    return C

# Назначение: считывает данные из внешнего источника.
def _h5_read_str(ds, i: int) -> str:
    """
    Считывает данные из внешнего источника.

    Параметры
    ---------
    ds :
        Параметр используется в соответствии с назначением функции.
    i :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_h5_read_str`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _h5_read_str(ds=..., i=...)
    """
    v = ds[i]
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8")
    return str(v)

@torch.no_grad()
# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def make_y3_from_gt_pointwise(y_fire01: torch.Tensor, y_smoke01: torch.Tensor) -> torch.Tensor:
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    y_fire01 :
        Параметр используется в соответствии с назначением функции.
    y_smoke01 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `make_y3_from_gt_pointwise`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = make_y3_from_gt_pointwise(y_fire01=..., y_smoke01=...)
    """
    if y_fire01.ndim == 2:  y_fire01 = y_fire01[:,0]
    if y_smoke01.ndim == 2: y_smoke01 = y_smoke01[:,0]
    fire = y_fire01 > 0
    smoke = y_smoke01 > 0
    y3 = torch.zeros_like(fire, dtype=torch.int64)
    y3[smoke] = 1
    y3[fire]  = 2
    return y3

# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def _infer_grid_shape_from_centers(cy_np: np.ndarray, cx_np: np.ndarray, Hr: int, Wr: int, R: int, stride: int):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    cy_np :
        Параметр используется в соответствии с назначением функции.
    cx_np :
        Параметр используется в соответствии с назначением функции.
    Hr :
        Параметр используется в соответствии с назначением функции.
    Wr :
        Параметр используется в соответствии с назначением функции.
    R :
        Параметр используется в соответствии с назначением функции.
    stride :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_infer_grid_shape_from_centers`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _infer_grid_shape_from_centers(cy_np=..., cx_np=..., Hr=...)
    """
    # типичный диапазон: [R, Hr-R) step=stride
    # ожидаемые размеры:
    ny = max(1, int(np.floor((Hr - 2*R - 1) / stride) + 1))
    nx = max(1, int(np.floor((Wr - 2*R - 1) / stride) + 1))
    # sanity: count should match
    return ny, nx

# Назначение: формирует или обрабатывает бинарные маски объектов.
def _dense_mask_from_grid(y3_flat: np.ndarray, Hr: int, Wr: int, R: int, stride: int):
    """
    Формирует или обрабатывает бинарные маски объектов.

    Параметры
    ---------
    y3_flat :
        Параметр используется в соответствии с назначением функции.
    Hr :
        Параметр используется в соответствии с назначением функции.
    Wr :
        Параметр используется в соответствии с назначением функции.
    R :
        Параметр используется в соответствии с назначением функции.
    stride :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_dense_mask_from_grid`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _dense_mask_from_grid(y3_flat=..., Hr=..., Wr=...)
    """
    N = int(y3_flat.shape[0])
    ny, nx = _infer_grid_shape_from_centers(None, None, Hr, Wr, R, stride)
    if ny * nx != N:
        return None

    grid = y3_flat.reshape(ny, nx).astype(np.uint8)

    out_h = Hr - 2*R
    out_w = Wr - 2*R
    up = cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    full = np.zeros((Hr, Wr), dtype=np.uint8)
    full[R:Hr-R, R:Wr-R] = up
    return full

# Назначение: выполняет вспомогательную операцию в вычислительном конвейере.
def _overlay_rgb(img_rgb: np.ndarray, mask_y3: np.ndarray, alpha=0.55):
    """
    Выполняет вспомогательную операцию в вычислительном конвейере.

    Параметры
    ---------
    img_rgb :
        Параметр используется в соответствии с назначением функции.
    mask_y3 :
        Параметр используется в соответствии с назначением функции.
    alpha :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_overlay_rgb`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _overlay_rgb(img_rgb=img_rgb, mask_y3=mask, alpha=...)
    """
    vis = img_rgb.copy().astype(np.float32)

    fire = (mask_y3 == 2)
    smoke = (mask_y3 == 1)

    overlay = np.zeros_like(vis)
    overlay[fire]  = np.array([255, 0, 0], dtype=np.float32)     # fire red
    overlay[smoke] = np.array([255, 255, 255], dtype=np.float32) # smoke white

    vis = (1 - alpha) * vis + alpha * overlay
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    return vis

# Назначение: формирует или обрабатывает бинарные маски объектов.
def visualize_committee_masks_from_saved(
    h5_path: str,
    y3_pred_all,               # torch.Tensor или np.ndarray длиной N (samples)
    img_idx: int | None = None,
    max_tries: int = 50,
    alpha: float = 0.55,
):
    """
    Формирует или обрабатывает бинарные маски объектов.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.
    y3_pred_all :
        Параметр используется в соответствии с назначением функции.
    img_idx :
        Параметр используется в соответствии с назначением функции.
    max_tries :
        Параметр используется в соответствии с назначением функции.
    alpha :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `visualize_committee_masks_from_saved`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = visualize_committee_masks_from_saved(h5_path='путь/к/ресурсу', y3_pred_all=..., img_idx=img_rgb)
    """
    if isinstance(y3_pred_all, torch.Tensor):
        y3_pred_all_t = y3_pred_all.detach().to("cpu")
        y3_pred_all_np = y3_pred_all_t.numpy()
    else:
        y3_pred_all_np = np.asarray(y3_pred_all)

    with h5py.File(h5_path, "r") as h5:
        stride = int(h5.attrs["stride"])
        R_local = int(globals().get("R", 12))  # если R определён; иначе 12  для WINDOW=25

        starts = h5["meta/sample_start"][:].astype(np.int64)
        counts = h5["meta/sample_count"][:].astype(np.int64)

        n_imgs = len(starts)

        # выбор изображения
        if img_idx is None:
            rng = np.random.default_rng(0)
            for _ in range(max_tries):
                cand = int(rng.integers(0, n_imgs))
                if counts[cand] > 0:
                    img_idx = cand
                    break
            if img_idx is None:
                raise RuntimeError("Не нашла ни одного изображения с sample_count>0")

        s = int(starts[img_idx]); c = int(counts[img_idx])
        if c <= 0:
            raise RuntimeError(f"img_idx={img_idx} has sample_count=0")

        sl = slice(s, s+c)

        # данные
        fn = _h5_read_str(h5["meta/file_name"], img_idx)
        crop_y = int(h5["meta/crop_y"][img_idx])
        Hr = int(h5["meta/resize_h"][img_idx])
        Wr = int(h5["meta/resize_w"][img_idx])

        cy = h5["samples/cy"][sl].astype(np.int32)
        cx = h5["samples/cx"][sl].astype(np.int32)
        y_fire01  = torch.from_numpy(h5["samples/y_fire"][sl].astype(np.uint8))
        y_smoke01 = torch.from_numpy(h5["samples/y_smoke"][sl].astype(np.uint8))

        y3_gt_pts = make_y3_from_gt_pointwise(y_fire01, y_smoke01).numpy().astype(np.uint8)
        y3_pr_pts = y3_pred_all_np[sl].astype(np.uint8)

        # восстановить изображение то, что реально использовалось для признаков
        # Нужны твои функции/константы проекта:
        # ROOT, get_img_path, load_coco_split уже не требуется — мы читаем изображение напрямую.
        split = "test"  # для test h5
        split_dir = os.path.join(ROOT, split)
        img_path = get_img_path(split_dir, fn)
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        if crop_y > 0:
            img_rgb = img_rgb[crop_y:, :, :]

        # если было resize — повторим
        was_resized = int(h5["meta/was_resized"][img_idx])
        if was_resized == 1:
            img_rgb = cv2.resize(img_rgb, (Wr, Hr), interpolation=cv2.INTER_AREA)
        else:
            # на всякий случай приводим к ожидаемым Hr,Wr если вдруг
            img_rgb = img_rgb[:Hr, :Wr, :]

        # плотные маски (через grid->upsample)
        gt_dense = _dense_mask_from_grid(y3_gt_pts, Hr, Wr, R_local, stride)
        pr_dense = _dense_mask_from_grid(y3_pr_pts, Hr, Wr, R_local, stride)

        # fallback:  по точкам
        if gt_dense is None or pr_dense is None:
            gt_dense = np.zeros((Hr, Wr), dtype=np.uint8)
            pr_dense = np.zeros((Hr, Wr), dtype=np.uint8)
            gt_dense[cy, cx] = y3_gt_pts
            pr_dense[cy, cx] = y3_pr_pts
            # чуть “раздуть” точки, чтобы видеть
            k = max(1, stride // 2)
            gt_dense = cv2.dilate(gt_dense, np.ones((2*k+1,2*k+1), np.uint8), iterations=1)
            pr_dense = cv2.dilate(pr_dense, np.ones((2*k+1,2*k+1), np.uint8), iterations=1)

        # наложения
        gt_overlay = _overlay_rgb(img_rgb, gt_dense, alpha=alpha)
        pr_overlay = _overlay_rgb(img_rgb, pr_dense, alpha=alpha)

        # отдельные бинарные метки пожар/не пожар
        gt_haz = (gt_dense > 0).astype(np.uint8)
        pr_haz = (pr_dense > 0).astype(np.uint8)

        # графики
        plt.figure(figsize=(16,10))
        plt.suptitle(f"img_idx={img_idx} | {fn} | crop_y={crop_y} | stride={stride} | R={R_local} | samples={c}", fontsize=12)

        ax = plt.subplot(2,3,1); ax.set_title("Image (processed)"); ax.imshow(img_rgb); ax.axis("off")
        ax = plt.subplot(2,3,2); ax.set_title("GT overlay"); ax.imshow(gt_overlay); ax.axis("off")
        ax = plt.subplot(2,3,3); ax.set_title("PRED overlay"); ax.imshow(pr_overlay); ax.axis("off")

        ax = plt.subplot(2,3,4); ax.set_title("GT y3 (0/1/2)"); ax.imshow(gt_dense, vmin=0, vmax=2); ax.axis("off")
        ax = plt.subplot(2,3,5); ax.set_title("PRED y3 (0/1/2)"); ax.imshow(pr_dense, vmin=0, vmax=2); ax.axis("off")
        ax = plt.subplot(2,3,6); ax.set_title("Hazard (GT vs PRED)"); ax.imshow(gt_haz*255, cmap="gray"); ax.imshow(pr_haz*255, cmap="inferno", alpha=0.35); ax.axis("off")

        plt.tight_layout()
        plt.show()

        return {
            "img_idx": img_idx,
            "file_name": fn,
            "crop_y": crop_y,
            "Hr": Hr, "Wr": Wr,
            "stride": stride,
            "R": R_local,
            "n_samples_img": c,
        }
# Назначение: считывает данные из внешнего источника.
def _h5_read_str(x):
    """
    Считывает данные из внешнего источника.

    Параметры
    ---------
    x :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `_h5_read_str`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = _h5_read_str(x=...)
    """
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    return str(x)

# Назначение: формирует или обрабатывает бинарные маски объектов.
def colorize_mask(mask3):
    """
    Формирует или обрабатывает бинарные маски объектов.

    Параметры
    ---------
    mask3 :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `colorize_mask`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = colorize_mask(mask3=mask)
    """
    H,W = mask3.shape
    rgb = np.zeros((H,W,3), dtype=np.uint8)
    rgb[mask3==1] = (255,255,255)  # дым
    rgb[mask3==2] = (255,0,0)      # огонь
    rgb[mask3==3] = (0,0,255)      # небо
    return rgb

# Назначение: формирует или обрабатывает бинарные маски объектов.
def visualize_committee_masks_one(
    h5_path: str,
    split_dir: str,          # ROOT + "/test"
    npz_path: str,           # сохранённый файл с y3_pred
    img_idx: int = 0,
    alpha: float = 1,
):
    """
    Формирует или обрабатывает бинарные маски объектов.

    Параметры
    ---------
    h5_path :
        Параметр используется в соответствии с назначением функции.
    split_dir :
        Параметр используется в соответствии с назначением функции.
    npz_path :
        Параметр используется в соответствии с назначением функции.
    img_idx :
        Параметр используется в соответствии с назначением функции.
    alpha :
        Параметр используется в соответствии с назначением функции.

    Возвращает
    ----------
    Результат вычислений, формируемый объектом `visualize_committee_masks_one`.

    Примечания
    ----------
    Документационная строка добавлена для упрощения сопровождения, повторного
    использования кода и включения файла в состав отчётной документации.

    Пример использования
    -------------------
    >>> result = visualize_committee_masks_one(h5_path='путь/к/ресурсу', split_dir='путь/к/ресурсу', npz_path='путь/к/ресурсу')
    """
    data = np.load(npz_path)
    y3_pred_all = data["y3_pred"]          # (N,) uint8
    y3_pred_all = torch.from_numpy(y3_pred_all.astype(np.int64))  # for slicing logic below

    with h5py.File(h5_path, "r") as h5:
        fn = _h5_read_str(h5["meta/file_name"][img_idx])
        crop_y = int(h5["meta/crop_y"][img_idx])
        was_resized = int(h5["meta/was_resized"][img_idx])
        rh = int(h5["meta/resize_h"][img_idx])
        rw = int(h5["meta/resize_w"][img_idx])

        start = int(h5["meta/sample_start"][img_idx])
        count = int(h5["meta/sample_count"][img_idx])
        if count <= 0:
            raise RuntimeError(f"img_idx={img_idx} has 0 samples in h5")

        sl = slice(start, start+count)

        cy = h5["samples/cy"][sl].astype(np.int32)
        cx = h5["samples/cx"][sl].astype(np.int32)
        y_fire = h5["samples/y_fire"][sl].astype(np.uint8)
        y_smoke = h5["samples/y_smoke"][sl].astype(np.uint8)

        # предсказания
        y3_pred = y3_pred_all[sl].cpu().numpy().astype(np.uint8)  # {0,1,2}

        y3_gt = np.zeros_like(y3_pred, dtype=np.uint8)
        y3_gt[y_smoke > 0] = 1
        y3_gt[y_fire  > 0] = 2

        # перестрока обрезанногое по размеру изображения в соответствии с координатами cy/cx
        img_path = os.path.join(split_dir, fn)
        img_rgb0 = np.array(Image.open(img_path).convert("RGB"))
        H0,W0 = img_rgb0.shape[:2]

        # обрезка неба
        if crop_y > 0:
            img_c = img_rgb0[crop_y:, :, :]
        else:
            img_c = img_rgb0

        # изменение размера
        if was_resized == 1:
            img_c = cv2.resize(img_c, (rw, rh), interpolation=cv2.INTER_AREA)

        Hr, Wr = img_c.shape[:2]

        # растрирование маски
        gt_mask = np.zeros((Hr,Wr), dtype=np.uint8)
        pr_mask = np.zeros((Hr,Wr), dtype=np.uint8)

        gt_mask[cy, cx] = y3_gt
        pr_mask[cy, cx] = y3_pred

            # добавляем "небо" в визуализацию (синее) в качестве обрезанной части, но только для отображения:
            # две версии:
            # - обрезанный вид (Hr x Wr), где неба не существует
            # - исходный вид (H0 x W0) с окрашенной областью неба.

        # build original-sized masks (for nicer view)
        gt_full = np.zeros((H0,W0), dtype=np.uint8)
        pr_full = np.zeros((H0,W0), dtype=np.uint8)

        # отмечаем небо
        if crop_y > 0:
            gt_full[:crop_y, :] = 3
            pr_full[:crop_y, :] = 3

        # возвращаем обрезанные маски на место
        if was_resized == 0:
            gt_full[crop_y:crop_y+Hr, :Wr] = np.maximum(gt_full[crop_y:crop_y+Hr, :Wr], gt_mask)
            pr_full[crop_y:crop_y+Hr, :Wr] = np.maximum(pr_full[crop_y:crop_y+Hr, :Wr], pr_mask)
            img_show = img_rgb0
        else:
            # при изменении размера по-прежнему накладываем изображение в координатах с измененным размером и обрезкой; для просмотра:
            # показываем само изображение с измененным размером и обрезкой, чтобы избежать несоответствия геометрии исходному W0/H0
            img_show = img_c
            gt_full = gt_mask
            pr_full = pr_mask

        gt_rgb = colorize_mask(gt_full)
        pr_rgb = colorize_mask(pr_full)

        # наложение
        overlay_gt = (img_show*(1-alpha) + gt_rgb*alpha).astype(np.uint8)
        overlay_pr = (img_show*(1-alpha) + pr_rgb*alpha).astype(np.uint8)

    # демонстрация
    plt.figure(figsize=(18,6))
    plt.subplot(1,3,1); plt.title(f"Image ({fn})"); plt.imshow(img_show); plt.axis("off")
    plt.subplot(1,3,2); plt.title("GT mask overlay"); plt.imshow(overlay_gt); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Pred mask overlay (committee C4)"); plt.imshow(overlay_pr); plt.axis("off")
    plt.show()
