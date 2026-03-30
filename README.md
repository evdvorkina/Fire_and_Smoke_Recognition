# Применение комитета классификаторов к решению задачи распознания пожароопасных ситуаций

В данной работе рассматривается задача распознавания и сегментации пожароопасных ситуаций на изображениях по признакам дыма и огня. Актуальность направления исследования обусловлена необходимостью оперативного выявления возгораний при анализе с камер наблюдения, в том числе в условиях ограниченных вычислительных ресурсов. Для решения задачи предлагается использовать метод, основанный на применении комитета классификаторов, голосующими элементами которого являются быстрые искусственные нейронные сети, снабженные функцией активации нового типа s-parabola. Обработка изображений выполняется путем отсечения области неба по линии горизонта для снижения влияния атмосферных явлений на результат распознавания. Полученное изображение просматривается сканирующим окном, центральному пикселю которого присваивается один из следующих классов “дым”, “огонь”, “норма” на основе решения голосующих элементов комитета. Предложенный подход характеризуется низкими требованиями к вычислительным ресурсам и простотой интеграции в существующие системы видеонаблюдения.
Приводятся результаты экспериментальных исследований на изображениях, включающих сцены с наличием пожара, дыма и штатных ситуаций без признаков возгорания. Качество решения оценивалось по стандартным метрикам Precision, Recall, которые вычисляются на основании корректности классификации отдельных пикселей, а также по проценту найденных пожаров. Результаты показывают, что эффективность обнаружения пожаров напрямую зависит от выбранной архитектуры и функции активации классификаторов. Полученные результаты подтверждают эффективность применения комитета классификаторов из быстрых искусственных нейронных сетей с функцией активации s-parabola к решению задач обнаружения пожароопасных ситуаций.

## Используемый набор данных:

https://universe.roboflow.com/khoi-v0jxf/fire-smoke-segmentation-3ym8x

---

## Введение

Методы компьютерного зрения активно применяются в задачах мониторинга окружающей среды, промышленной безопасности и интеллектуального видеонаблюдения. Одной из практически значимых задач в данной области является обнаружение пожароопасных ситуаций по изображениям, получаемым с камер наблюдения, беспилотных летательных аппаратов и других устройств регистрации. Своевременное выявление дыма и огня позволяет сократить время реагирования на возгорание и уменьшить возможный ущерб.

В последние годы для решения задач обнаружения пожаров широко применяются глубокие сверточные нейронные сети, в том числе модели сегментации и детектирования объектов. Такие методы обеспечивают высокое качество распознавания, однако нередко требуют значительных вычислительных ресурсов, большого объема обучающих данных и специализированного аппаратного обеспечения. В связи с этим сохраняет актуальность разработка более легких и вычислительно эффективных подходов, пригодных для практического использования в условиях ограниченной производительности.

В данном проекте исследуется подход, основанный на построении комитета классификаторов из нескольких быстрых искусственных нейронных сетей. Особенностью реализованного подхода является то, что анализ изображения выполняется не целиком, а методом скользящего окна. Для каждого положения окна вычисляется вектор признаков, описывающий текстурные и статистические свойства локального фрагмента изображения. Далее центральный пиксель окна относится к одному из классов: «огонь», «дым» или «норма». Такой способ обработки позволяет использовать неглубокие нейронные сети и при этом учитывать локальную структуру изображения. Для уменьшения числа ложноположительных срабатываний на этапе предобработки выполняется отсечение области неба по линии горизонта.

Целью проекта является исследование возможностей применения комитета классификаторов из быстрых искусственных нейронных сетей с функцией активации s-parabola к задаче распознавания пожароопасных ситуаций на изображениях.

## Необходимое ПО

### Основные зависимости

Для запуска потребуется установить библиотеки:
```bash
pip install numpy opencv-python matplotlib pillow pandas tqdm h5py pycocotools
pip install torch torchvision
pip install segmentation-models-pytorch
```

Используемые модули Python:

```Python
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
import segmentation_models_pytorch as smp
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
```

### Рекомендуемая среда

* Python 3.10+
* GPU 
* Google Colab / Linux / Windows

---

## Описание набора данных

Используется открытый набор данных, предназначенный для многоклассовой сегментации объектов https://universe.roboflow.com/khoi-v0jxf/fire-smoke-segmentation-3ym8x

Формат масок:

* COCO segmentation (`_annotations.coco.json`)
* классы:
  * `fire`
  * `smoke`
 
Разделение изображений:

* Тренировочная выборка -- 1070 изображений
* Валидационная выборка -- 111 изображений
* Тестовая выборка -- 125 изображений

## Предобработка данных

### Подготовка данных
После подготовки проекта используются следующие каталоги:
```bash
code/
crop_cache/
best ins weights/
best UNet weights/
standartization params/
test output/
```
Исходный код расположен в папке code/, где находятся основные файлы:

* `Model.py` -- предобработка данных, обучение ИНС и сбор комитетов
* `System.py` -- общая система с возможностью подачи на вход изображения и получения на выходи маски сегментации по результатам работы комитета
* `Unet Comparison.py` -- обучение U-Net с весами ResNet.

Файлы с заранее вычисленными значениями отсечения неба по линии горизонта расположены в папке: `crop_cache/`:

```bash
crop_y_train_orig_K11_t0.06.json
crop_y_valid_orig_K11_t0.06.json
crop_y_test_orig_K11_t0.06.json
```

Необходимо распаковать набор данных:

```bash
fire-smoke-segmentation/
    train/
    valid/
    test/
    _cache/
```

### Отсечение неба по линии горизонта

Реализована функция `compute_crop_y_original` в файле `code/Model.py`, вычисляющая значение y, по которому будет происходить обрезка изображения. Функция `build_crop_cache` для ускорения повторных запусков сохраняет результаты вычисления границы отсечения в JSON-файлы в папке `crop_cache/`

Пример:

```python
ROOT = "/content/fire-smoke-segmentation.v4i.coco-segmentation"
SPLITS = ["train", "valid", "test"]

crop_cache_paths = {s: build_crop_cache(s) for s in SPLITS}
```

Пример поэтапного результата работы алгоритма поиска линии горизонта:

<p align="center">
  <img src="https://github.com/user-attachments/assets/2f30c659-5e55-446f-96d9-3c2489768655" width="180"/>
  <img src="https://github.com/user-attachments/assets/b83929b4-b848-48bc-9ffa-15d31fd817e2" width="180"/>
  <img src="https://github.com/user-attachments/assets/844a4f5b-ea2c-4706-b29a-9b0075eb76d9" width="180"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f2c4ff2b-32dd-4a4c-844d-e912c03587d0" width="180"/>
  <img src="https://github.com/user-attachments/assets/603b1101-5474-48ab-b446-c3111ad51261" width="180"/>
</p>

### Приведение изображения к рабочему размеру

Изображение при необходимости приводится к фиксированному размеру 224×224 пикселя с помощью функции. Если размер изображения после обрезки меньше целевого, применяется дополнение до требуемого формата. Такое приведение необходимо для унификации дальнейшей обработки и сопоставимости результатов комитетов классификаторов и U-Net.

### Извлечение признаков

Обработка изображения выполняется методом скользящего окна размером 25×25 пикселей. Для каждого положения окна формируется вектор признаков, описывающий локальный фрагмент изображения. Базовым объектом классификации является центральный пиксель окна. При последовательном перемещении окна по изображению формируется множество локальных описаний, по которым далее принимается решение о принадлежности пикселя к одному из классов:

* `fire`;
* `smoke`;
* `non-fire`.

Вектор признаков содержит 278 признаков:

* 147 -- признаки Лавса, характеризующие текстурные свойства изображения, вычисляются функцией `laws_maps_gpu` из `Model.py`;
* 96 -- признаки Харалика, вычисляемые по матрицам совместной встречаемости, вычисляются функцией `stats_maps_gpu` из `Model.py`;
* 35 -- статистические признаки, описывающие распределение значений цветовых компонент, вычисляются функцией `lharalick96_patches_gpu` из `Model.py`.

### Стандартизация признаков

Перед подачей признаков в нейронные сети выполняется стандартизация. Параметры стандартизации сохраняются отдельно и используются повторно на этапах тестирования и сравнения моделей: `standartization params/ins2_standardization_params.pt`

Пример применения:

```Python
from Model import apply_standardizer_gpu

# загрузка параметров стандартизации
params = torch.load("standartization params/ins2_standardization_params.pt")

mu = params["mu"]
sigma = params["sigma"]

# X278 — матрица признаков (N, 278)
X278 = torch.from_numpy(X278_np).to("cuda")

# стандартизация
X278_norm = apply_standardizer_gpu(
    X278,
    mu.to("cuda"),
    sigma.to("cuda")
)
```

## Подбор гиперпараметров классификаторов

Подбор гиперпараметров выполняется на валидационной выборке.

Настраиваются:

* параметры функции активации `s-parabola` (`p`, `beta`);
* пороги классификации (`threshold`);

Подбор гиперпараметров на примере ИНС-1:

```Python
# --- INS-1 uses first 182 features: [LAWS(147) + STATS(35)] ---
INS1_SLICE = slice(0, 182)

# choose target for INS-1
Y_KEY = "y_smoke"   # INS-1 = smoke / non-smoke

train_h5 = paths["train"]
valid_h5 = paths["valid"]

# 1) load full X(278) + y(-1/+1) to GPU
Xtr278, ytr = load_h5_to_cuda(train_h5, y_key=Y_KEY, device="cuda", max_rows=None, as_pm1=True)
Xva278, yva = load_h5_to_cuda(valid_h5, y_key=Y_KEY, device="cuda", max_rows=None, as_pm1=True)

ins2_params_path = "/content/ins2_standardization_params.pt"
ins2_params = torch.load(ins2_params_path, map_location="cpu")
mu_ins2 = ins2_params['mu'].to(Xtr278.device)
sigma_ins2 = ins2_params['sigma'].to(Xtr278.device)

Xtr278_n = apply_standardizer_gpu(Xtr278, mu_ins2, sigma_ins2, out_dtype=torch.float16)
Xva278_n = apply_standardizer_gpu(Xva278, mu_ins2, sigma_ins2, out_dtype=torch.float16)

# free big tensors
del Xtr278, Xva278
torch.cuda.empty_cache()

# 2) slice X -> 182 dims for INS-1
Xtr_n = Xtr278_n[:, INS1_SLICE].contiguous()
Xva_n = Xva278_n[:, INS1_SLICE].contiguous()

# free big tensors
del Xtr278_n, Xva278_n
torch.cuda.empty_cache()

print("INS-1 tensors on GPU:", Xtr_n.shape, ytr.shape, Xva_n.shape, yva.shape, "pos_rate(train)=", float((ytr>0).float().mean()))
# ------------------ GRID SEARCH CELL ------------------
p_grid    = [0.25, 0.5, 1.0, 2.0]
beta_grid = [0.0, 0.1, 0.3, 0.5]
layer_sizes = [182, 90, 45, 10]
epochs = 5
batch_size = 32768
lr = 3e-4

min_recall = 0.50   # твое ограничение (можно 0.10 если хочешь сильнее давить FP)

results = []
best = None

for p, beta in product(p_grid, beta_grid):
    rep = train_one_cfg_mae_thr(
        Xtr_n, ytr, Xva_n, yva,
        p=p, beta=beta,
        epochs=epochs, batch_size=batch_size, lr=lr,
        min_recall=min_recall,
        thr_grid=None,
        seed=42,
        layer_sizes = layer_sizes
    )
    results.append({k: v for k,v in rep.items() if k != "model_state"})

    key = (rep["P"], rep["F1"], rep["R"])
    if (best is None) or (key > (best["P"], best["F1"], best["R"])):
        best = rep

    print(f"p={p:<4} beta={beta:<3} -> best_thr={rep['thr_out']:.2f} | P={rep['P']:.4f} R={rep['R']:.4f} F1={rep['F1']:.4f} | fp={rep['fp']} fn={rep['fn']}")
df = pd.DataFrame(results).sort_values(["P","F1","R"], ascending=False).reset_index(drop=True)
display(df.head(12))

print("\nBEST CONFIG:")
print({k: best[k] for k in ["p","beta","thr_out","P","R","F1","fp","fn","epochs","batch_size","lr"]})

# keep best state to reuse for final training / saving
best_ins1_cfg = best
```

Результат подбора гиперпараметров для каждой ИНС:

| №   | Архитектура | Назначение | Параметры функции активации | Порог принятия решения |
|-----|------------|------------|-----------------------------|------------------------|
| 1   | 182 входа, 4 скрытых слоя: 182, 90, 45, 10, 1 выход | Обнаружение дыма по признакам Лавса и статистическим признакам | p = 1.0, β = 0.3 | 0.975 |
| 2   | 278 входов, 4 скрытых слоя: 278, 140, 70, 10, 1 выход | Обнаружение дыма на полном наборе признаков | p = 1.0, β = 0.3 | 0.975 |
| 3A  | 182 входа, 3 скрытых слоя: 182, 64, 16, 1 выход | Уточнение распознавания дыма по признакам Лавса и статистическим признакам | p = 0.5, β = 0.5 | 1.025 |
| 3B  | 278 входов, 3 скрытых слоя: 278, 96, 24, 1 выход | Уточнение распознавания дыма на полном наборе признаков | p = 2.0, β = 0.0 | 1.0 |
| 4   | 35 входов, 3 скрытых слоя: 35, 20, 5, 1 выход | Обнаружение огня по статистическим признакам | p = 0.5, β = 0.0 | 0.25 |
| 5   | 278 входов, 3 скрытых слоя: 278, 96, 24, 1 выход | Обнаружение огня на полном наборе признаков | p = 0.25, β = 0.0 | 0.7 |

## Обучение классификаторов

Запуск обучения ИНС в ноутбуке на примере ИНС-1 (В файле `Model.py` находятся все необходимые классы и функции: `FastMLP`, `S_Parabola`):

```python
# =========================
# FINAL TRAIN CELL (INS-1)
# =========================

# ---------- CONFIG (подставь лучшие) ----------
BEST_P    = 1
BEST_BETA = 0.3
BEST_THR  = 0.975          # порог по out (не по sigmoid)
LR        = 3e-4
EPOCHS    = 30
BS        = 32768
CLIP_NORM = 5.0
# архитектура INS-1: вход 182
INPUT_DIM = Xtr_n.shape[1]
HIDDEN    = [182, 90, 45, 10]   # INS-1 hidden (как в статье/у тебя)

# куда сохранить
SAVE_PATH = "/content/ins1_fastmlp_sparabola_best.pt"
device = Xtr_n.device

# ---------- model / loss / opt ----------
model = FastMLP(
    input_dim=INPUT_DIM,
    layer_sizes=HIDDEN,
    activation="sparabola",
    p=BEST_P,
    beta=BEST_BETA
).to(device)

# MAE по выходу (как ты хотела "похоже на статью": |y* - y_out|)
crit = nn.L1Loss(reduction="mean")
opt  = torch.optim.AdamW(model.parameters(), lr=LR)

# =========================
# CELL 2: train loop (FIX SHAPES)
# =========================
best = None

N  = Xtr_n.shape[0]
Nv = Xva_n.shape[0]

for ep in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(N, device=device)
    loss_sum = 0.0
    n_sum = 0

    for i in range(0, N, BS):
        idx = perm[i:i+BS]
        xb = Xtr_n[idx].float()

        # FIX: y must be (B,1) to match out (B,1)
        yb = _as_col(ytr[idx]).float()

        opt.zero_grad(set_to_none=True)
        out = model(xb)          # (B,1)
        loss = crit(out, yb)     # SAFE: both (B,1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()

        loss_sum += loss.item() * xb.size(0)
        n_sum += xb.size(0)

    tr_loss = loss_sum / max(n_sum, 1)

    # ---- eval (каждую эпоху) ----
    with torch.no_grad():
        out_tr = eval_out_chunked(model, Xtr_n, bs=BS)  # (N,1)
        out_va = eval_out_chunked(model, Xva_n, bs=BS)  # (Nv,1)

        Ptr, Rtr, F1tr, tp, fp, fn, tn, mtr, str_ = metrics_pm1_from_out(out_tr, ytr, thr_out=BEST_THR)
        Pva, Rva, F1va, tpv, fpv, fnv, tnv, mva, sva = metrics_pm1_from_out(out_va, yva, thr_out=BEST_THR)

        # FIX: (Nv,1) - (Nv,1)
        yva2 = _as_col(yva).float()
        va_err = (out_va.float() - yva2).abs().mean().item()

    print(
        f"ep {ep:02d} | loss_tr={tr_loss:.4f} | val_err={va_err:.4f} | "
        f"out_mean={mva:.4f} out_std={sva:.4f} | "
        f"Pva={Pva:.4f} Rva={Rva:.4f} F1va={F1va:.4f} | fp={fpv} fn={fnv}"
    )

    # сохраняем лучшую по минимальной val_err (MAE)
    key = (-va_err,)
    if (best is None) or (key > best["key"]):
        best = {
            "key": key, "epoch": ep,
            "val_err": float(va_err),
            "Pva": Pva, "Rva": Rva, "F1va": F1va,
            "thr_out": float(BEST_THR),
            "p": float(BEST_P), "beta": float(BEST_BETA),
            "lr": float(LR), "bs": int(BS),
        }
        save_checkpoint(SAVE_PATH, model, best)

print("\nBEST:", best)
print("Saved:", SAVE_PATH)
```

Результат обучения моделей по эпохам в виде графиков:

<p align="center">
  <img src="https://github.com/user-attachments/assets/e69b4c7b-df98-46e8-9eb7-93725d6893e3" width="250"/>
  <img src="https://github.com/user-attachments/assets/2e30a4ca-6adb-4615-bfe1-9d4bf52e4600" width="250"/>
  <img src="https://github.com/user-attachments/assets/f5b5b237-1b85-4aa9-bc29-6a4b239d8711" width="250"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/2734dc23-dec7-45cc-8667-09a155ab0716" width="250"/>
  <img src="https://github.com/user-attachments/assets/dc620b7c-16e3-45b3-9537-3cf7fcfbc0c4" width="250"/>
  <img src="https://github.com/user-attachments/assets/207346a6-5c03-4743-8a9e-a614a9133fcc" width="250"/>
</p>

## Реализация комитетов

Комитеты реализованы функциями `smoke_committe_pred` и `fire_committe_pred` из файла `Model.py` 

Пример запуска комитетов:

```python
IMG_THRESHOLDS = (0.0001, 0.0003, 0.0005, 0.001, 0.01)

df_K = run_K_committees(
    h5_test_path=h5_test_path,
    y_smoke_pm1=y_smoke.cpu(),    # {-1,+1}
    y_fire_pm1=y_fire.cpu(),      # {-1,+1}
    out_ins1_path=OUT_INS1,
    out_ins2_path=OUT_INS2,
    out_ins3B_path=OUT_INS3B,
    out_ins4_path=OUT_INS4,
    out_ins5_path=OUT_INS5,
    thr_ins1=THR_INS1,
    thr_ins2=THR_INS2,
    thr_ins3B=THR_INS3B,
    thr_ins4=THR_INS4,
    thr_ins5=THR_INS5,
    img_thresholds=IMG_THRESHOLDS,
    save_dir="/content/committee_preds_K",
)
```

Определена логика комитетов функцией `define_committees` из `Model.py`

### Оценка качества

Для оценки качества классификации использовались метрики глобальной и средней точности, глобальной и средней полноты, а также процент найденных пожаров, определяющий, на скольких изображениях из всех был правильно обнаружен пожар.

Глобальная точность и глобальная полнота вычисляются следующим образом:

$$
Precision_{global} = \frac{TP_{global}}{TP_{global} + FP_{global}}
$$

$$
Recall_{global} = \frac{TP_{global}}{TP_{global} + FN_{global}}
$$

где:
- $TP_{global}$ — общее количество пикселей во всех тестовых изображениях, распознанных истинно положительно;
- $FP_{global}$ — общее количество пикселей во всех тестовых изображениях, распознанных ложно положительно;
- $FN_{global}$ — общее количество пикселей во всех тестовых изображениях, распознанных ложно отрицательно.

Средняя точность и средняя полнота вычисляются по формулам:

$$
Precision_{mean} = \frac{1}{N} \sum_{i=1}^{N} Precision_i
$$

$$
Recall_{mean} = \frac{1}{N} \sum_{i=1}^{N} Recall_i
$$

где:
- $Precision_i$ — точность распознавания $i$-го изображения;
- $Recall_i$ — полнота распознавания $i$-го изображения;
- $N$ — количество обработанных изображений.

Пример вывода оценки качества

```python
metrics = binary_metrics_from_pred(pred, target)
```

или:

```python
cm = confusion_3x3(y_true, y_pred)
```

### Оценка скорости работы системы

```python
res = benchmark_unet_like_committee(paths)
print(res)
```

## Результаты и сравнение с U-Net

Следует отметить, что в процессе экспериментальных исследований была рассмотрена модель ИНС-3A, предназначенная для уточнения распознавания дыма на основе признаков Лавса и статистических признаков. Однако полученные результаты показали её низкую эффективность по сравнению с моделью ИНС-3B, использующей полный набор признаков. В связи с этим в итоговой реализации комитетов классификаторов используется только модель ИНС-3B.

###  Обучение U-Net (baseline)

Используется нейросеть архитектуры U-Net с предобученными весами ResNet. Реализация находится в файле `code\Unet Comparison.py`. Пример:

```python
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2
)
```

###  Оценка качества работы комитетов

| № | Состав комитета классификаторов | Глобальная точность | Глобальная полнота | Средняя точность | Средняя полнота | Процент найденных пожаров, % |
|---|--------------------------------|--------------------|--------------------|------------------|------------------|-------------------------------|
| 1 | ИНС 1, ИНС 4 | 0.926 | 0.693 | 0.862 | 0.615 | 0.968 |
| 2 | ИНС 2, ИНС 4 | 0.925 | 0.752 | 0.862 | 0.655 | 0.976 |
| 3 | ИНС 1, ИНС 2, ИНС 4 | 0.919 | 0.801 | 0.853 | 0.692 | 0.976 |
| 4 | ИНС 1, ИНС 5 | 0.941 | 0.690 | 0.873 | 0.614 | 0.960 |
| 5 | ИНС 2, ИНС 5 | 0.939 | 0.748 | 0.873 | 0.654 | 0.968 |
| 6 | ИНС 1, ИНС 2, ИНС 5 | 0.932 | 0.798 | 0.863 | 0.691 | 0.968 |
| 7 | ИНС 1, ИНС 2, ИНС 3B, ИНС 4 | 0.925 | 0.633 | 0.866 | 0.572 | 0.976 |
| 8 | ИНС 1, ИНС 2, ИНС 4, ИНС 5 | 0.917 | 0.815 | 0.851 | 0.719 | 0.984 |
| 9 | ИНС 1, ИНС 2, ИНС 3B, ИНС 4, ИНС 5 | 0.921 | 0.647 | 0.863 | 0.598 | 0.984 |

###  Сравнение работы комитетов с U-Net

Возьмем два комитета, показавших наилучший результат по нахождению пожара, сравним их с U-Net. Результаты показывают, что качество решения поставленной задачи с помощью комитетов сопоставимо по основным показателям с СНС.

| № | Классификатор | Глобальная точность | Глобальная полнота | Средняя точность | Средняя полнота | Процент найденных пожаров, % |
|---|--------------|--------------------|--------------------|------------------|------------------|-------------------------------|
| 1 | U-Net | 0.945 | 0.936 | 0.882 | 0.878 | 0.984 |
| 2 | ИНС 1, ИНС 2, ИНС 4, ИНС 5 | 0.917 | 0.815 | 0.851 | 0.719 | 0.984 |
| 3 | ИНС 1, ИНС 2, ИНС 3B, ИНС 4, ИНС 5 | 0.921 | 0.647 | 0.863 | 0.598 | 0.984 |

### Скорость работы комитетов

Скорость работы комитета имеет обратную зависимость от количества классификаторов в нем, однако разница в скорости работы при количестве классификаторов от 2 до 5, в целом, незаметна и мала.

| № | Состав комитета классификаторов | Скорость (пиксели/с) |
|---|--------------------------------|----------------------|
| 1 | ИНС 1, ИНС 4 | 54002.60 |
| 2 | ИНС 2, ИНС 4 | 53885.25 |
| 3 | ИНС 1, ИНС 2, ИНС 4 | 53665.60 |
| 4 | ИНС 1, ИНС 5 | 53876.56 |
| 5 | ИНС 2, ИНС 5 | 53738.67 |
| 6 | ИНС 1, ИНС 2, ИНС 5 | 53340.04 |
| 7 | ИНС 1, ИНС 2, ИНС 3B, ИНС 4 | 53237.08 |
| 8 | ИНС 1, ИНС 2, ИНС 4, ИНС 5 | 53366.81 |
| 9 | ИНС 1, ИНС 2, ИНС 3B, ИНС 4, ИНС 5 | 52999.04 |

### Реализованный функционал

* Предобработка изображений:
  * отсечение горизонта;
  * нормализация;
  * изменение рамзерности и паддинг;

* Извлечение признаков в вектор:
  * Лавса;
  * Харалика;
  * Статистических признаков;

* Классификаторы:

  * ИНС-1: определяет дым / не дым по статистическим признакам и признакам Лавса;
  * ИНС-2: определяет дым / не дым по полному набору признаков;
  * ИНС-3А: дым / не дым, уточняющая результаты ИНС-1 и ИНС-2 (обучалась на ошибках), по статистическим признакам и признакам Лавса;
  * ИНС-3B: дым / не дым, уточняющая результаты ИНС-1 и ИНС-2 (обучалась на ошибках), по полному набору признаков;
  * ИНС-4: огонь / не огонь по статистическим признакам;
  * ИНС-5: огонь / не огонь по полному набору признаков;

* Комитеты классификаторов с различным составом классификаторов

* Метрики качества:

  * Precision / Recall / F1;
  * confusion matrix;
  * detection rate;

* Сравнение с U-Net:

---

## Визуализация результатов

Примеры:
<table align="center">
  <tr>
    <th>№</th>
    <th>Исходное</th>
    <th>Эталонная маска</th>
    <th>K15</th>
    <th>K18</th>
    <th>U-Net</th>
  </tr>

  <!-- Строка 1 -->
  <tr>
    <td>1</td>
    <td><img src="https://github.com/user-attachments/assets/c12a518b-ef23-4bb4-a9b9-e4d41625865c" width="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/4c3fbf72-5608-4825-86fb-5cc675608133" width="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/038cd147-66cf-4954-9776-6166f60fc6b7" width="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/0f13077b-af64-4416-ac53-ab032201ad03" width="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/52212041-0f63-4d30-b0c9-2854e2e11fd1" width="120"/></td>
  </tr>

  <!-- Строка 2 -->
  <tr>
    <td>2</td>
    <td><img src="https://github.com/user-attachments/assets/0271e6c2-a323-43c8-b1be-c8bfafb36b01" width="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/8f6c0be0-1595-48d7-aefd-9d3e4baa7129" width="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/06aba9fa-e094-44a1-9ebd-8ae320d1b0e1" width="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/f79729e4-d7a7-49df-a27f-7938ce0f76c6" width="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/67742233-6dc3-4284-996e-5a11489e27e1" width="120"/></td>
  </tr>

  <!-- Строка 3 -->
  <tr>
    <td>3</td>
    <td><img src="https://github.com/user-attachments/assets/4d12c965-ad9b-4f54-b91b-ed19da94c6fd" width="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/77d2f59a-49a2-43a6-8535-e629e12188d3" width="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/7f773f33-86d5-4aa8-9769-6f4a14e377ba" width="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/d65cea60-3eeb-4eda-8579-ad1d86c21f70" width="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/d9c34c86-46d7-4a29-b06b-493951835999" width="120"/></td>
  </tr>
</table>

---

## Примеры использования

В файле `code\System.py` реализован метод `run_committee_on_image`, позволяющий запускать систему на новом изображении и строить для него маску сегментации с использованием выбранного комитета классификаторов.  Аналогичный метод `unet_process_frame` с использованием дообученной U-Net реализован в `code\Unet Comparison.py`

### Пример 1: предсказание U-Net на новом изображении

```python
pred_fire, pred_smoke = unet_process_frame(img, crop_y=10)
```

### Пример 2: запуск работы комитета для обработки изображения

```python
mask, info = run_committee_on_image(img, committee_id="K18")
```

---

## Новизна

* использование функции активации **s-parabola**;
* применение **комитета из лёгких моделей** вместо одной тяжелой нейросетти;

---

## Структура проекта

```text
.
├── best UNet weights/          # сохраненные веса модели U-Net
├── best ins weights/           # сохраненные веса искусственных нейронных сетей
├── code/                       # исходный код проекта
│   ├── Model.py                # извлечение признаков, предобработка, построение признаковых описаний
│   ├── System.py               # логика комитетов классификаторов и запуск инференса
│   └── Unet Comparison.py      # обучение, тестирование и сравнение с U-Net
├── crop_cache/                 # кэш значений crop_y для отсечения неба по линии горизонта
├── standartization params/     # параметры стандартизации признаков
├── test output/                # примеры результатов работы моделей и комитетов
└── README.md
```

---

## Примечания

Проект выполнен в рамках научно-исследовательской работы студента.
Результаты носят исследовательский характер.

