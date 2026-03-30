from __future__ import annotations

"""
Модуль System.py
================

Назначение модуля
-----------------
Модуль реализует логику работы комитета классификаторов для распознавания
пожароопасных ситуаций на изображениях. В составе конвейера выполняются:

1. загрузка входного изображения;
2. отсечение области выше линии горизонта;
3. при необходимости изменение размера изображения;
4. извлечение признаков;
5. нормализация признаков;
6. запуск выбранных моделей;
7. объединение их предсказаний по правилам комитета;
8. восстановление итоговой карты классов в размер исходного изображения;
9. измерение скорости работы комитетов.

Используемые обозначения классов
--------------------------------
- 0 — фон / отсутствие признаков пожара;
- 1 — дым;
- 2 — огонь;
- 3 — область неба, отсечённая по линии горизонта.

Примечание
----------
В данном файле предполагается, что функции и модели, импортируемые из модуля
Model, уже реализованы и корректно инициализированы.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
from collections import defaultdict
from pycocotools import mask as maskUtils
import zipfile
import io
from Model import (
    compute_crop_y_original,
    make_3class_pred,
    infer_out_chunked,
    features_B_278_gpu_for_image_stride,
    apply_standardizer_gpu,
)
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

# ============================================================
# Унифицированная логика комитетов классификаторов
# ============================================================


def smoke_committee_pred(
    mode: str,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3b: torch.Tensor | None,
):
    """
    Формирует бинарное решение по классу «дым» на основе набора предсказаний.

    Параметры
    ---------
    mode : str
        Режим объединения предсказаний. Поддерживаются значения:
        - ``"v12"``    — логическое ИЛИ для ИНС-1 и ИНС-2;
        - ``"c12"``    — логическое И для ИНС-1 и ИНС-2;
        - ``"v12_v3"`` — логическое ИЛИ для ИНС-1, ИНС-2 и ИНС-3B;
        - ``"v12_c3"`` — (ИНС-1 ИЛИ ИНС-2) И ИНС-3B.
    p1 : torch.Tensor
        Бинарные предсказания ИНС-1.
    p2 : torch.Tensor
        Бинарные предсказания ИНС-2.
    p3b : torch.Tensor | None
        Бинарные предсказания ИНС-3B. Используются только в режимах,
        где предусмотрено участие третьей модели.

    Возвращает
    ----------
    torch.Tensor
        Булев тензор, где ``True`` означает наличие дыма.

    Пример
    -------
    >>> p1 = torch.tensor([True, False, True])
    >>> p2 = torch.tensor([False, False, True])
    >>> p3 = torch.tensor([True, True, False])
    >>> smoke_committee_pred("v12", p1, p2, p3)
    tensor([ True, False,  True])
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
    raise ValueError(f"Неизвестный режим объединения для дыма: {mode}")



def fire_committee_pred(mode: str, p4: torch.Tensor, p5: torch.Tensor):
    """
    Формирует бинарное решение по классу «огонь» на основе набора предсказаний.

    Параметры
    ---------
    mode : str
        Режим объединения предсказаний. Поддерживаются значения:
        - ``"4"``   — использовать только ИНС-4;
        - ``"5"``   — использовать только ИНС-5;
        - ``"v45"`` — логическое ИЛИ для ИНС-4 и ИНС-5;
        - ``"c45"`` — логическое И для ИНС-4 и ИНС-5.
    p4 : torch.Tensor
        Бинарные предсказания ИНС-4.
    p5 : torch.Tensor
        Бинарные предсказания ИНС-5.

    Возвращает
    ----------
    torch.Tensor
        Булев тензор, где ``True`` означает наличие огня.

    Пример
    -------
    >>> p4 = torch.tensor([True, False])
    >>> p5 = torch.tensor([False, True])
    >>> fire_committee_pred("v45", p4, p5)
    tensor([True, True])
    """
    if mode == "4":
        return p4
    if mode == "5":
        return p5
    if mode == "v45":
        return p4 | p5
    if mode == "c45":
        return p4 & p5
    raise ValueError(f"Неизвестный режим объединения для огня: {mode}")



def committee_from_cfg(outs: dict, thr: dict, cfg: dict):
    """
    Вычисляет итоговые бинарные решения комитета по конфигурации.

    Функция преобразует численные выходы моделей в булевы предсказания на
    основании порогов, после чего объединяет их по правилам, указанным в
    конфигурации комитета.

    Параметры
    ---------
    outs : dict
        Словарь выходов моделей. Значения должны быть тензорами формы ``(N, 1)``
        или ``(N,)``.
    thr : dict
        Словарь порогов. Ожидаются ключи ``ins1``, ``ins2``, ``ins3B``, ``ins4``
        и ``ins5``.
    cfg : dict
        Словарь конфигурации комитета. Обычно содержит ключи:
        ``sm``, ``fi``, ``use_1``, ``use_2``, ``use_3``, ``use_4``, ``use_5``.

    Возвращает
    ----------
    tuple[torch.Tensor, torch.Tensor]
        Кортеж из двух булевых тензоров:
        ``(smoke_pos, fire_pos)``.

    Пример
    -------
    >>> outs = {
    ...     "ins1": torch.tensor([[0.8], [0.2]]),
    ...     "ins2": torch.tensor([[0.6], [0.9]]),
    ...     "ins4": torch.tensor([[0.1], [0.7]]),
    ... }
    >>> thr = {"ins1": 0.5, "ins2": 0.5, "ins3B": 0.5, "ins4": 0.5, "ins5": 0.5}
    >>> cfg = {"sm": "v12", "fi": "4", "use_1": True, "use_2": True, "use_3": False, "use_4": True, "use_5": False}
    >>> smoke_pos, fire_pos = committee_from_cfg(outs, thr, cfg)
    """
    any_key = next(iter(outs.keys()))
    any_out = outs[any_key]
    if any_out.ndim == 2:
        n_samples = any_out.shape[0]
        dev = any_out.device
    else:
        n_samples = any_out.numel()
        dev = any_out.device

    def _get_1d(key: str) -> torch.Tensor:
        """Возвращает одномерный тензор предсказаний модели по имени ключа."""
        if key not in outs:
            raise KeyError(f"В словаре outs отсутствует ключ '{key}'. Доступно: {list(outs.keys())}")
        x = outs[key]
        if x.ndim == 2:
            x = x[:, 0]
        return x

    def _zeros() -> torch.Tensor:
        """Создаёт булев вектор нулей для неиспользуемой модели."""
        return torch.zeros((n_samples,), device=dev, dtype=torch.bool)

    p1 = (_get_1d("ins1") >= float(thr["ins1"])) if cfg.get("use_1", False) else _zeros()
    p2 = (_get_1d("ins2") >= float(thr["ins2"])) if cfg.get("use_2", False) else _zeros()
    p4 = (_get_1d("ins4") >= float(thr["ins4"])) if cfg.get("use_4", False) else _zeros()
    p5 = (_get_1d("ins5") >= float(thr["ins5"])) if cfg.get("use_5", False) else _zeros()

    p3b = None
    if cfg.get("use_3", False):
        p3b = _get_1d("ins3B") >= float(thr["ins3B"])

    smoke_pos = smoke_committee_pred(cfg["sm"], p1, p2, p3b)
    fire_pos = fire_committee_pred(cfg["fi"], p4, p5)
    return smoke_pos, fire_pos


# Конфигурации доступных комитетов K1...K18.
COMMITTEES_CFG = {
    "K1":  dict(sm="v12",     fi="4",   use_1=True, use_2=False, use_3=False, use_4=True,  use_5=False),
    "K2":  dict(sm="v12",     fi="4",   use_1=False, use_2=True, use_3=False, use_4=True,  use_5=False),
    "K3":  dict(sm="v12",     fi="4",   use_1=True, use_2=True,  use_3=False, use_4=True,  use_5=False),
    "K4":  dict(sm="c12",     fi="4",   use_1=True, use_2=True,  use_3=False, use_4=True,  use_5=False),
    "K5":  dict(sm="v12",     fi="5",   use_1=True, use_2=False, use_3=False, use_4=False, use_5=True),
    "K6":  dict(sm="v12",     fi="5",   use_1=False, use_2=True, use_3=False, use_4=False, use_5=True),
    "K7":  dict(sm="v12",     fi="5",   use_1=True, use_2=True,  use_3=False, use_4=False, use_5=True),
    "K8":  dict(sm="c12",     fi="5",   use_1=True, use_2=True,  use_3=False, use_4=False, use_5=True),
    "K9":  dict(sm="v12_v3",  fi="4",   use_1=True, use_2=True,  use_3=True,  use_4=True,  use_5=False),
    "K10": dict(sm="v12_c3",  fi="4",   use_1=True, use_2=True,  use_3=True,  use_4=True,  use_5=False),
    "K11": dict(sm="v12_v3",  fi="5",   use_1=True, use_2=True,  use_3=True,  use_4=False, use_5=True),
    "K12": dict(sm="v12_c3",  fi="5",   use_1=True, use_2=True,  use_3=True,  use_4=False, use_5=True),
    "K13": dict(sm="v12",     fi="v45", use_1=True, use_2=False, use_3=False, use_4=True,  use_5=True),
    "K14": dict(sm="v12",     fi="v45", use_1=False, use_2=True, use_3=False, use_4=True,  use_5=True),
    "K15": dict(sm="v12",     fi="v45", use_1=True, use_2=True,  use_3=False, use_4=True,  use_5=True),
    "K16": dict(sm="c12",     fi="v45", use_1=True, use_2=True,  use_3=False, use_4=True,  use_5=True),
    "K17": dict(sm="v12_v3",  fi="v45", use_1=True, use_2=True,  use_3=True,  use_4=True,  use_5=True),
    "K18": dict(sm="v12_c3",  fi="v45", use_1=True, use_2=True,  use_3=True,  use_4=True,  use_5=True),
}



def _unwrap_out(x):
    """
    Извлекает тензор предсказаний из различных форматов ответа.

    Функция предназначена для унификации результата работы ``infer_out_chunked``.
    Она поддерживает прямой тензор, кортеж, список или словарь.

    Параметры
    ---------
    x : Any
        Объект, содержащий тензор предсказаний.

    Возвращает
    ----------
    torch.Tensor
        Извлечённый тензор.

    Исключения
    ----------
    TypeError
        Вызывается, если тензор не удалось найти.

    Пример
    -------
    >>> _unwrap_out(torch.tensor([[0.1], [0.9]])).shape
    torch.Size([2, 1])
    """
    if isinstance(x, torch.Tensor):
        return x

    if isinstance(x, (list, tuple)):
        for value in reversed(x):
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError("Функция infer_out_chunked вернула список или кортеж без тензора внутри")

    if isinstance(x, dict):
        for key in ["out", "outs", "logits", "y", "pred", "output"]:
            if key in x and isinstance(x[key], torch.Tensor):
                return x[key]
        for value in x.values():
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError("Функция infer_out_chunked вернула словарь без тензора в значениях")

    raise TypeError(f"Неподдерживаемый тип результата infer_out_chunked: {type(x)}")


# ============================================================
# Вспомогательные функции для конвейера комитета
# ============================================================


def _now():
    """
    Возвращает текущее значение высокоточного таймера.

    Функция используется для замера времени выполнения отдельных этапов
    конвейера обработки изображения.

    Возвращает
    ----------
    float
        Текущее значение таймера в секундах.

    Пример
    -------
    >>> start = _now()
    >>> _ = sum(range(10))
    >>> elapsed = _now() - start
    """
    return time.perf_counter()



def _sync(device):
    """
    Синхронизирует выполнение операций на графическом устройстве.

    Для CPU функция не выполняет специальных действий. Для CUDA-устройства
    вызывается ``torch.cuda.synchronize()``, что необходимо для корректного
    измерения времени выполнения GPU-операций.

    Параметры
    ---------
    device : str | torch.device
        Устройство вычислений.

    Пример
    -------
    >>> _sync("cuda")  # при наличии CUDA
    """
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()



def _to_u8_rgb(img):
    """
    Приводит входное изображение к формату ``np.uint8`` в пространстве RGB.

    Поддерживаются изображения в формате ``PIL.Image`` и ``numpy.ndarray``.
    Одноканальные изображения автоматически преобразуются к трёхканальному виду.

    Параметры
    ---------
    img : PIL.Image.Image | numpy.ndarray
        Входное изображение.

    Возвращает
    ----------
    numpy.ndarray
        Массив формы ``(H, W, 3)`` и типа ``np.uint8``.

    Пример
    -------
    >>> img_rgb = _to_u8_rgb(Image.open("image.png"))
    >>> img_rgb.dtype
    dtype('uint8')
    """
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img



def _resize_rgb(img_rgb_u8, target_wh):
    """
    Изменяет размер RGB-изображения до заданного значения.

    Параметры
    ---------
    img_rgb_u8 : numpy.ndarray
        Цветное изображение формата RGB и типа ``np.uint8``.
    target_wh : tuple[int, int]
        Целевой размер в формате ``(ширина, высота)``.

    Возвращает
    ----------
    numpy.ndarray
        Изображение изменённого размера.

    Пример
    -------
    >>> resized = _resize_rgb(img_rgb, (224, 224))
    """
    return cv2.resize(img_rgb_u8, target_wh, interpolation=cv2.INTER_AREA)



def _slice_stats_from_278(X278):
    """
    Извлекает подмножество статистических признаков из 278-мерного вектора.

    Предполагается, что признаки организованы следующим образом:
    - первые 147 значений — признаки Laws;
    - следующие 35 значений — статистические признаки;
    - оставшиеся значения — признаки Харалика.

    Параметры
    ---------
    X278 : torch.Tensor
        Матрица признаков формы ``(N, 278)``.

    Возвращает
    ----------
    torch.Tensor
        Матрица статистических признаков формы ``(N, 35)``.

    Пример
    -------
    >>> X = torch.randn(10, 278)
    >>> _slice_stats_from_278(X).shape
    torch.Size([10, 35])
    """
    return X278[:, 147:182]



def _rasterize_mask_proc_from_points(y3_pts_u8, cy_np, cx_np, Hr, Wr):
    """
    Формирует дискретную карту классов в пространстве обработанного изображения.

    Каждому центру окна сопоставляется метка класса, после чего эти значения
    записываются в соответствующие координаты результирующей маски.

    Параметры
    ---------
    y3_pts_u8 : numpy.ndarray
        Одномерный массив длины ``N`` с метками классов ``{0, 1, 2}``.
    cy_np : numpy.ndarray
        Массив координат строк центров окон.
    cx_np : numpy.ndarray
        Массив координат столбцов центров окон.
    Hr : int
        Высота обработанного изображения.
    Wr : int
        Ширина обработанного изображения.

    Возвращает
    ----------
    numpy.ndarray
        Маска формы ``(Hr, Wr)`` и типа ``np.uint8``.

    Пример
    -------
    >>> mask = _rasterize_mask_proc_from_points(y, cy, cx, 224, 224)
    """
    mask = np.zeros((Hr, Wr), dtype=np.uint8)
    mask[cy_np, cx_np] = y3_pts_u8
    return mask



def _restore_full_mask(mask_proc_u8, H0, W0, crop_y, was_resized, Hr, Wr):
    """
    Восстанавливает маску классов в размере исходного изображения.

    Если изображение было обрезано по горизонту, верхняя часть помечается
    как небо. Если после обрезки выполнялось изменение размера, маска сначала
    масштабируется обратно до размеров обрезанной области исходного кадра.

    Параметры
    ---------
    mask_proc_u8 : numpy.ndarray
        Маска обработанного изображения.
    H0 : int
        Высота исходного изображения.
    W0 : int
        Ширина исходного изображения.
    crop_y : int
        Координата строки, выше которой кадр был отсечён.
    was_resized : bool
        Признак изменения размера изображения после обрезки.
    Hr : int
        Высота обработанного изображения.
    Wr : int
        Ширина обработанного изображения.

    Возвращает
    ----------
    numpy.ndarray
        Итоговая маска формы ``(H0, W0)`` с метками ``{0, 1, 2, 3}``.

    Пример
    -------
    >>> full_mask = _restore_full_mask(mask_proc, 1080, 1920, 120, False, 960, 1920)
    """
    mask_full = np.zeros((H0, W0), dtype=np.uint8)
    if crop_y > 0:
        mask_full[:crop_y, :] = 3

    if not was_resized:
        h_crop = H0 - crop_y
        h = min(h_crop, mask_proc_u8.shape[0])
        w = min(W0, mask_proc_u8.shape[1])
        region = mask_full[crop_y:crop_y + h, :w]
        region = np.maximum(region, mask_proc_u8[:h, :w])
        mask_full[crop_y:crop_y + h, :w] = region
        return mask_full

    h_crop = H0 - crop_y
    mask_unres = cv2.resize(mask_proc_u8, (W0, h_crop), interpolation=cv2.INTER_NEAREST)
    mask_full[crop_y:, :] = np.maximum(mask_full[crop_y:, :], mask_unres.astype(np.uint8))
    return mask_full


# ============================================================
# Реестр отдельных правил комитетов
# ============================================================


def committee_smoke_C1_voting(outs, thr):
    """
    Реализует голосование двух моделей дыма по правилу логического ИЛИ.

    Параметры
    ---------
    outs : dict
        Словарь выходов моделей.
    thr : dict
        Словарь порогов принятия решения.

    Возвращает
    ----------
    torch.Tensor
        Булев тензор наличия дыма.

    Пример
    -------
    >>> smoke_pos = committee_smoke_C1_voting(outs, thr)
    """
    o1 = outs["ins1"].squeeze(1)
    o2 = outs["ins2"].squeeze(1)
    smoke_pos = (o1 >= thr["ins1"]) | (o2 >= thr["ins2"])
    return smoke_pos



def committee_smoke_C2_confirmation(outs, thr):
    """
    Реализует подтверждение дыма двумя моделями по правилу логического И.

    Параметры и возвращаемое значение аналогичны функции
    :func:`committee_smoke_C1_voting`.

    Пример
    -------
    >>> smoke_pos = committee_smoke_C2_confirmation(outs, thr)
    """
    o1 = outs["ins1"].squeeze(1)
    o2 = outs["ins2"].squeeze(1)
    smoke_pos = (o1 >= thr["ins1"]) & (o2 >= thr["ins2"])
    return smoke_pos



def committee_smoke_C4_voting_with_ins3B(outs, thr):
    """
    Реализует голосование трёх моделей дыма по правилу логического ИЛИ.

    В данной реализации ИНС-3B включается в общее голосование наравне с ИНС-1
    и ИНС-2.

    Пример
    -------
    >>> smoke_pos = committee_smoke_C4_voting_with_ins3B(outs, thr)
    """
    o1 = outs["ins1"].squeeze(1)
    o2 = outs["ins2"].squeeze(1)
    o3 = outs["ins3B"].squeeze(1)
    smoke_pos = (o1 >= thr["ins1"]) | (o2 >= thr["ins2"]) | (o3 >= thr["ins3B"])
    return smoke_pos



def committee_smoke_C6_strict_confirm_by_ins3B(outs, thr):
    """
    Выполняет двухэтапное решение по дыму:
    сначала голосование ИНС-1 и ИНС-2, затем подтверждение ИНС-3B.

    Пример
    -------
    >>> smoke_pos = committee_smoke_C6_strict_confirm_by_ins3B(outs, thr)
    """
    o1 = outs["ins1"].squeeze(1)
    o2 = outs["ins2"].squeeze(1)
    o3 = outs["ins3B"].squeeze(1)
    smoke12 = (o1 >= thr["ins1"]) | (o2 >= thr["ins2"])
    smoke_pos = smoke12 & (o3 >= thr["ins3B"])
    return smoke_pos



def committee_fire_F1_voting(outs, thr):
    """
    Реализует голосование моделей огня ИНС-4 и ИНС-5 по правилу ИЛИ.

    Пример
    -------
    >>> fire_pos = committee_fire_F1_voting(outs, thr)
    """
    o4 = outs["ins4"].squeeze(1)
    o5 = outs["ins5"].squeeze(1)
    fire_pos = (o4 >= thr["ins4"]) | (o5 >= thr["ins5"])
    return fire_pos



def committee_fire_F2_confirmation(outs, thr):
    """
    Реализует подтверждение огня моделями ИНС-4 и ИНС-5 по правилу И.

    Пример
    -------
    >>> fire_pos = committee_fire_F2_confirmation(outs, thr)
    """
    o4 = outs["ins4"].squeeze(1)
    o5 = outs["ins5"].squeeze(1)
    fire_pos = (o4 >= thr["ins4"]) & (o5 >= thr["ins5"])
    return fire_pos



def committee_K18(outs, thr):
    """
    Реализует комитет K18.

    Логика:
    - дым: (ИНС-1 ИЛИ ИНС-2) И ИНС-3B;
    - огонь: ИНС-4 ИЛИ ИНС-5.

    Возвращает
    ----------
    tuple[torch.Tensor, torch.Tensor]
        Булевы тензоры ``(smoke_pos, fire_pos)``.

    Пример
    -------
    >>> smoke_pos, fire_pos = committee_K18(outs, thr)
    """
    smoke_pos = committee_smoke_C6_strict_confirm_by_ins3B(outs, thr)
    fire_pos = committee_fire_F1_voting(outs, thr)
    return smoke_pos, fire_pos



def committee_K15(outs, thr):
    """
    Реализует комитет K15.

    Логика:
    - дым: ИНС-1 ИЛИ ИНС-2;
    - огонь: ИНС-4 ИЛИ ИНС-5.

    Пример
    -------
    >>> smoke_pos, fire_pos = committee_K15(outs, thr)
    """
    o1 = outs["ins1"].squeeze(1)
    o2 = outs["ins2"].squeeze(1)
    smoke_pos = (o1 >= thr["ins1"]) | (o2 >= thr["ins2"])
    fire_pos = committee_fire_F1_voting(outs, thr)
    return smoke_pos, fire_pos



def committee_K16(outs, thr):
    """
    Реализует комитет K16.

    Логика:
    - дым: ИНС-1 И ИНС-2;
    - огонь: ИНС-4 ИЛИ ИНС-5.

    Пример
    -------
    >>> smoke_pos, fire_pos = committee_K16(outs, thr)
    """
    o1 = outs["ins1"].squeeze(1)
    o2 = outs["ins2"].squeeze(1)
    smoke_pos = (o1 >= thr["ins1"]) & (o2 >= thr["ins2"])
    fire_pos = committee_fire_F1_voting(outs, thr)
    return smoke_pos, fire_pos



def _get_needed_models_for_committee(committee_id: str) -> set[str]:
    """
    Возвращает множество моделей, необходимых для выбранного комитета.

    Параметры
    ---------
    committee_id : str
        Идентификатор комитета, например ``"K18"``.

    Возвращает
    ----------
    set[str]
        Множество имён моделей, которые необходимо запустить.

    Пример
    -------
    >>> _get_needed_models_for_committee("K18")
    {'ins1', 'ins2', 'ins3B', 'ins4', 'ins5'}
    """
    cfg = COMMITTEES_CFG[committee_id]
    need = set()
    if cfg.get("use_1", False):
        need.add("ins1")
    if cfg.get("use_2", False):
        need.add("ins2")
    if cfg.get("use_3", False):
        need.add("ins3B")
    if cfg.get("use_4", False):
        need.add("ins4")
    if cfg.get("use_5", False):
        need.add("ins5")
    return need


# ============================================================
# Основной конвейер работы комитета
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
    resize_to: tuple[int, int] = (224, 224),
    stride: int = 1,
    window: int = 25,
    levels: int = 32,
    chunk_center_rows: int = 64,
    mu278: torch.Tensor = None,
    sigma278: torch.Tensor = None,
    out_dtype: torch.dtype = torch.float16,
    bs_infer: int = 32768,
    return_outs: bool = False,
):
    """
    Запускает полный цикл работы комитета классификаторов для одного изображения.

    Этапы выполнения:
    1. преобразование входа к RGB ``uint8``;
    2. отсечение области выше линии горизонта;
    3. необязательное изменение размера;
    4. извлечение признаков;
    5. нормализация признаков;
    6. запуск нужных моделей;
    7. вычисление решения комитета;
    8. восстановление итоговой маски в размер исходного изображения.

    Параметры
    ---------
    img_rgb_or_pil : numpy.ndarray | PIL.Image.Image
        Входное изображение.
    models : dict
        Словарь моделей. Для комитета K18 ожидаются ключи ``ins1``, ``ins2``,
        ``ins3B``, ``ins4``, ``ins5``.
    thr : dict
        Словарь порогов классификации для каждой модели.
    committee_id : str, optional
        Идентификатор комитета, по умолчанию ``"K18"``.
    device : str, optional
        Устройство вычислений, по умолчанию ``"cuda"``.
    use_resize : bool, optional
        Признак использования изменения размера изображения.
    resize_to : tuple[int, int], optional
        Размер ``(ширина, высота)`` после масштабирования.
    stride : int, optional
        Шаг окон при извлечении признаков.
    window : int, optional
        Размер окна.
    levels : int, optional
        Число уровней квантования признаков Харалика.
    chunk_center_rows : int, optional
        Размер блока строк центров, обрабатываемых за один раз.
    mu278 : torch.Tensor, optional
        Вектор средних значений для нормализации 278-мерных признаков.
    sigma278 : torch.Tensor, optional
        Вектор стандартных отклонений для нормализации 278-мерных признаков.
    out_dtype : torch.dtype, optional
        Тип данных после нормализации.
    bs_infer : int, optional
        Размер пакета при инференсе моделей.
    return_outs : bool, optional
        Если ``True``, дополнительно возвращаются сырые выходы моделей.

    Возвращает
    ----------
    tuple
        Если ``return_outs=False``:
        ``(mask_full, info)``.

        Если ``return_outs=True``:
        ``(mask_full, info, outs)``.

        Здесь:
        - ``mask_full`` — итоговая маска классов формы ``(H0, W0)``;
        - ``info`` — словарь с геометрией, временами выполнения и скоростью;
        - ``outs`` — словарь выходов моделей.

    Пример
    -------
    >>> img = Image.open("example.png").convert("RGB")
    >>> mask_full, info = run_committee_on_image(
    ...     img,
    ...     models=models_dict,
    ...     thr=thr_dict,
    ...     committee_id="K18",
    ...     device="cuda",
    ...     use_resize=True,
    ...     resize_to=(224, 224),
    ... )
    """
    assert committee_id in COMMITTEES_CFG, (
        f"Неизвестный идентификатор комитета: {committee_id}. "
        f"Доступные варианты: {list(COMMITTEES_CFG.keys())}"
    )
    device = torch.device(device)

    # Этап 0. Загрузка входного изображения.
    t0 = _now()
    img0 = _to_u8_rgb(img_rgb_or_pil)
    H0, W0 = img0.shape[:2]
    _sync(device)
    t_load = _now() - t0

    # Этап 1. Отсечение области выше линии горизонта.
    t1 = _now()
    crop_y = int(compute_crop_y_original(img0))
    crop_y = int(np.clip(crop_y, 0, H0))
    img_c = img0[crop_y:, :, :] if crop_y > 0 else img0
    Hc, Wc = img_c.shape[:2]
    _sync(device)
    t_crop = _now() - t1

    # Этап 2. При необходимости выполняется изменение размера изображения.
    t2 = _now()
    was_resized = False
    if use_resize and ((Hc != resize_to[1]) or (Wc != resize_to[0])):
        img_proc = _resize_rgb(img_c, resize_to)
        was_resized = True
    else:
        img_proc = img_c
    Hr, Wr = img_proc.shape[:2]
    _sync(device)
    t_resize = _now() - t2

    # Этап 3. Извлечение признаков.
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

    n_windows = int(X278_np.shape[0])
    if n_windows == 0:
        mask_full = np.zeros((H0, W0), dtype=np.uint8)
        if crop_y > 0:
            mask_full[:crop_y, :] = 3
        info = dict(
            committee_id=committee_id,
            H0=H0,
            W0=W0,
            crop_y=crop_y,
            Hr=Hr,
            Wr=Wr,
            was_resized=int(was_resized),
            N=0,
            timings=dict(
                load=t_load,
                crop=t_crop,
                resize=t_resize,
                feat=t_feat,
                norm=0.0,
                infer=0.0,
                committee=0.0,
                restore=0.0,
            ),
            px_per_sec=0.0,
        )
        return (mask_full, info, {}) if return_outs else (mask_full, info)

    # Этап 4. Перенос признаков в torch и нормализация.
    t4 = _now()
    X278 = torch.from_numpy(X278_np).to(device=device, dtype=torch.float16, non_blocking=True)

    if (mu278 is not None) and (sigma278 is not None):
        X278_n = apply_standardizer_gpu(X278, mu278.to(device), sigma278.to(device), out_dtype=out_dtype)
    else:
        X278_n = X278.to(dtype=out_dtype)

    X182_n = X278_n[:, :182]
    X35_n = _slice_stats_from_278(X278_n)
    _sync(device)
    t_norm = _now() - t4

    # Этап 5. Инференс только тех моделей, которые требуются для выбранного комитета.
    t5 = _now()
    outs = {}
    needed_models = _get_needed_models_for_committee(committee_id)
    for key in sorted(list(needed_models)):
        model = models[key]

        if key in ["ins1"]:
            raw = infer_out_chunked(model, X182_n, bs=bs_infer)
            out = _unwrap_out(raw)
            outs[key] = out.to(torch.float32)

        elif key in ["ins2", "ins3B", "ins5"]:
            raw = infer_out_chunked(model, X278_n, bs=bs_infer)
            out = _unwrap_out(raw)
            outs[key] = out.to(torch.float32)

        elif key in ["ins4"]:
            raw = infer_out_chunked(model, X35_n, bs=bs_infer)
            out = _unwrap_out(raw)
            outs[key] = out.to(torch.float32)

        else:
            raise ValueError(f"Неизвестный ключ модели '{key}'. Необходимо добавить соответствующую маршрутизацию.")
    _sync(device)
    t_infer = _now() - t5

    # Этап 6. Применение логики комитета и формирование предсказания по точкам.
    t6 = _now()
    cfg = COMMITTEES_CFG[committee_id]
    smoke_pos, fire_pos = committee_from_cfg(outs, thr, cfg)

    # Приоритет огня над дымом задаётся функцией make_3class_pred.
    y3 = make_3class_pred(fire_pos, smoke_pos)
    y3_u8 = y3.detach().to("cpu").numpy().astype(np.uint8)

    mask_proc = _rasterize_mask_proc_from_points(y3_u8, cy_np, cx_np, Hr, Wr)
    _sync(device)
    t_committee = _now() - t6

    # Этап 7. Восстановление маски в размер исходного изображения.
    t7 = _now()
    mask_full = _restore_full_mask(mask_proc, H0, W0, crop_y, was_resized, Hr, Wr)
    t_restore = _now() - t7

    total_core = t_feat + t_norm + t_infer + t_committee
    px_per_sec = float(n_windows / max(total_core, 1e-12))

    info = dict(
        committee_id=committee_id,
        H0=H0,
        W0=W0,
        crop_y=crop_y,
        Hr=Hr,
        Wr=Wr,
        was_resized=int(was_resized),
        N=n_windows,
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
            total=float(t_load + t_crop + t_resize + total_core + t_restore),
        ),
        px_per_sec=px_per_sec,
        pred_fire_ratio=float((mask_proc == 2).mean()),
        pred_smoke_ratio=float((mask_proc == 1).mean()),
        pred_hazard_ratio=float((mask_proc > 0).mean()),
    )

    if return_outs:
        return mask_full, info, outs
    return mask_full, info


# Пример вызова:
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
    resize_to: tuple[int, int] = (224, 224),
    mu278=None,
    sigma278=None,
    bs_infer: int = 32768,
    warmup: int = 1,
    repeats: int = 5,
    out_xlsx: str = "/content/committee_speed.xlsx",
):
    """
    Выполняет замер скорости работы нескольких комитетов на наборе изображений.

    Для каждого изображения и каждого комитета выполняется несколько запусков,
    после чего сохраняются медианные значения времени и производительности.
    Итоговые результаты записываются в Excel-файл с двумя листами:

    - ``summary_by_committee`` — сводная таблица по комитетам;
    - ``per_image`` — детальные результаты по каждому изображению.

    Параметры
    ---------
    image_paths : list[str]
        Список путей к изображениям.
    committee_ids : list[str]
        Список идентификаторов комитетов, например ``["K15", "K18"]``.
    models : dict
        Словарь загруженных моделей.
    thr : dict
        Словарь порогов классификации.
    device : str, optional
        Устройство вычислений.
    use_resize : bool, optional
        Использовать ли масштабирование изображения перед извлечением признаков.
    resize_to : tuple[int, int], optional
        Целевой размер изображения в формате ``(ширина, высота)``.
    mu278, sigma278 : optional
        Параметры стандартизации признаков.
    bs_infer : int, optional
        Размер пакета при инференсе.
    warmup : int, optional
        Количество прогревочных запусков без записи результатов.
    repeats : int, optional
        Количество повторов замера для каждой пары «изображение — комитет».
    out_xlsx : str, optional
        Путь к выходному Excel-файлу.

    Возвращает
    ----------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Подробная таблица ``df`` и агрегированная таблица ``agg``.

    Пример
    -------
    >>> df, agg = benchmark_committees_speed(
    ...     image_paths=test_images,
    ...     committee_ids=["K15", "K18"],
    ...     models=models_dict,
    ...     thr=thr_dict,
    ...     device="cuda",
    ...     repeats=3,
    ...     out_xlsx="/content/committee_speed.xlsx",
    ... )
    """
    rows = []

    # Прогрев устройства и внутренних кешей перед основными измерениями.
    if warmup > 0 and len(image_paths) > 0 and len(committee_ids) > 0:
        img = Image.open(image_paths[0]).convert("RGB")
        for _ in range(warmup):
            _ = run_committee_on_image(
                img,
                models=models,
                thr=thr,
                committee_id=committee_ids[0],
                device=device,
                use_resize=use_resize,
                resize_to=resize_to,
                mu278=mu278,
                sigma278=sigma278,
                bs_infer=bs_infer,
            )

    for img_i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")

        for committee_id in committee_ids:
            pxs = []
            cores = []
            n_windows_list = []
            totals = []

            for _ in range(repeats):
                mask_full, info = run_committee_on_image(
                    img,
                    models=models,
                    thr=thr,
                    committee_id=committee_id,
                    device=device,
                    use_resize=use_resize,
                    resize_to=resize_to,
                    mu278=mu278,
                    sigma278=sigma278,
                    bs_infer=bs_infer,
                )
                pxs.append(float(info.get("px_per_sec", 0.0)))
                cores.append(float(info["timings"].get("core", 0.0)))
                totals.append(float(info["timings"].get("total", 0.0)))
                n_windows_list.append(int(info.get("N", 0)))

            row = {
                "image_idx": img_i,
                "image_path": img_path,
                "committee_id": committee_id,
                "N_windows": int(np.median(n_windows_list)),
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
            print(
                f"[{img_i}] {os.path.basename(img_path)} | {committee_id}: "
                f"px/s={row['px_per_sec_median']:.1f}  "
                f"core={row['core_sec_median']:.3f}s  "
                f"total={row['total_sec_median']:.3f}s"
            )

    df = pd.DataFrame(rows)

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

    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        agg.to_excel(writer, index=False, sheet_name="summary_by_committee")
        df.to_excel(writer, index=False, sheet_name="per_image")

    print("Файл сохранён:", out_xlsx)
    return df, agg
