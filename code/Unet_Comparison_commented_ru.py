"""
Модуль для обучения, оценки, визуализации и сравнения модели U-Net
в задаче сегментации дыма и огня на изображениях.

В модуле реализованы:
- классы датасетов для работы с COCO-разметкой;
- функции предобработки изображений в стиле комитета классификаторов;
- цикл обучения и валидации U-Net;
- вычисление бинарных и многоклассовых метрик;
- визуализация предсказаний;
- бенчмарки скорости работы модели U-Net.

Примечание:
В исходном коде присутствуют повторные определения некоторых функций
(load_coco_split, _ann_to_binary_mask, build_fire_smoke_masks, get_img_path,
load_crop_map). Это сохранено для совместимости с исходной логикой файла.
При последовательном запуске интерпретатор Python использует последнюю
встреченную реализацию функции с данным именем.

Пример использования модуля:
    # Формирование обучающей, валидационной и тестовой выборок
train_ds = FireSmokeCocoCommitteeLike(train_dir, crop_train, keep_frac=KEEP, seed=SEED)
    val_ds = FireSmokeCocoCommitteeLike(valid_dir, crop_valid, keep_frac=KEEP, seed=SEED)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    for epoch in range(5):
        train_loss = train_one_epoch()
        val_loss = eval_one_epoch()
        print(epoch, train_loss, val_loss)
"""

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
    """
    Класс датасета для чтения изображений и масок огня/дыма
        из COCO-разметки с последующим изменением размера.

        Назначение:
            Используется для базового обучения U-Net на полном изображении без
            специальной предобработки в стиле комитета классификаторов.

        Параметры:
            root_dir (str): путь к каталогу со split-набором данных.
            resize (tuple[int, int], optional): целевой размер изображения.
            keep_frac (float, optional): доля изображений, используемая из набора.
            seed (int, optional): зерно генератора случайных чисел.

        Возвращает:
            tuple[torch.Tensor, torch.Tensor]:
                - изображение размерности (3, H, W);
                - маска размерности (2, H, W), где канал 0 соответствует огню,
                  а канал 1 — дыму.

        Пример использования:
            dataset = CocoFireSmokeDataset("/content/data/train", resize=(224, 224))
            image, mask = dataset[0]

    """
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

        # Категории классов
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

        # Формирование пустых масок для классов «огонь» и «дым»
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

        # Приведение изображения и масок к заданному размеру
        image = cv2.resize(image, self.resize)
        fire_mask = cv2.resize(fire_mask, self.resize, interpolation=cv2.INTER_NEAREST)
        smoke_mask = cv2.resize(smoke_mask, self.resize, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0

        mask = np.stack([fire_mask, smoke_mask], axis=0).astype(np.float32)

        image = torch.from_numpy(image.transpose(2,0,1))
        mask = torch.from_numpy(mask)

        return image, mask


def get_device():
    """
    Возвращает строковый идентификатор доступного вычислительного устройства.

        Назначение:
            Используется для автоматического выбора между графическим и центральным
            процессором.

        Возвращает:
            str: значение "cuda", если CUDA доступна, иначе "cpu".

        Пример использования:
            device = get_device()

    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def _select_subset(file_names, frac, rng):
    """
    Формирует случайное подмножество имён файлов.

        Назначение:
            Позволяет ограничить объём обучающей или валидационной выборки при
            проведении экспериментов.

        Параметры:
            file_names (list[str]): список имён файлов.
            frac (float): доля элементов, которую требуется оставить.
            rng: генератор случайных чисел NumPy.

        Возвращает:
            list[str]: выбранное подмножество имён файлов.

        Пример использования:
            rng = np.random.default_rng(42)
            subset = _select_subset(file_names, 0.3, rng)

    """
    if frac >= 1.0:
        return file_names
    k = int(np.ceil(len(file_names) * frac))
    if k <= 0:
        return []
    idx = rng.choice(len(file_names), size=k, replace=False)
    return [file_names[i] for i in idx]

def load_coco_split(split_dir):
    """
    Загружает COCO-аннотации для выбранного разбиения набора данных.

        Назначение:
            Преобразует JSON-файл аннотаций в удобные словари с метаданными
            изображений, категориями и списками аннотаций по каждому изображению.

        Параметры:
            split_dir (str): путь к каталогу разбиения набора данных.

        Возвращает:
            tuple[dict, dict, dict]:
                - словарь изображений;
                - словарь категорий;
                - словарь аннотаций, сгруппированных по идентификатору изображения.

        Пример использования:
            images, cats, anns_by_img = load_coco_split("/content/data/test")

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

def _ann_to_binary_mask(ann, h, w):
    """
    Преобразует одну COCO-аннотацию в бинарную маску.

        Назначение:
            Поддерживает полигональную разметку, сжатый и несжатый формат RLE.

        Параметры:
            ann (dict): объект аннотации COCO.
            h (int): высота маски.
            w (int): ширина маски.

        Возвращает:
            numpy.ndarray: бинарная маска формы (h, w) типа uint8.

        Пример использования:
            mask = _ann_to_binary_mask(annotation, 224, 224)

    """
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
    """
    Строит итоговые бинарные маски классов «огонь» и «дым» для изображения.

        Назначение:
            Объединяет все объектные аннотации изображения в две совокупные маски.

        Параметры:
            img_meta (dict): метаданные изображения.
            anns (list[dict]): список аннотаций данного изображения.
            cats_map (dict): отображение идентификаторов категорий в их имена.

        Возвращает:
            tuple[numpy.ndarray, numpy.ndarray]:
                - маска огня;
                - маска дыма.

        Пример использования:
            fire_mask, smoke_mask = build_fire_smoke_masks(img_meta, anns, cats_map)

    """
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
    """
    Возвращает полный путь к изображению в каталоге split-набора данных.

        Назначение:
            Учитывает два варианта структуры каталога: непосредственное размещение
            файлов и размещение внутри подпапки images.

        Параметры:
            split_dir (str): путь к каталогу разбиения.
            file_name (str): имя файла изображения.

        Возвращает:
            str: полный путь к изображению.

        Исключения:
            FileNotFoundError: если файл не найден.

        Пример использования:
            path = get_img_path(test_dir, "image_001.jpg")

    """
    p1 = os.path.join(split_dir, file_name)
    if os.path.exists(p1): return p1
    p2 = os.path.join(split_dir, "images", file_name)
    if os.path.exists(p2): return p2
    raise FileNotFoundError(file_name)

def load_crop_map(crop_json_path):
    """
    Загружает словарь отсечения горизонта из JSON-файла.

        Назначение:
            Используется для применения одинаковой предобработки при сравнении
            U-Net и комитетов классификаторов.

        Параметры:
            crop_json_path (str): путь к JSON-файлу с координатами crop_y.

        Возвращает:
            dict: отображение вида {имя_файла: crop_y}.

        Пример использования:
            crop_map = load_crop_map(crop_test)

    """
    with open(crop_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["crop_y_by_file"]  # Словарь вида {имя_файла: crop_y}

def pad_to_224(img, fire, smoke, out_hw=(224,224)):
    """
    Дополняет изображение и маски до размера 224×224.

        Назначение:
            Приводит данные к фиксированному размеру, добавляя нулевые значения
            справа и снизу, если исходное изображение меньше целевого.

        Параметры:
            img (numpy.ndarray): изображение RGB.
            fire (numpy.ndarray): маска огня.
            smoke (numpy.ndarray): маска дыма.
            out_hw (tuple[int, int], optional): целевая высота и ширина.

        Возвращает:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                дополненные изображение и маски.

        Пример использования:
            img2, fire2, smoke2 = pad_to_224(img, fire, smoke)

    """
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
    """
    Класс датасета, имитирующего предобработку комитетов классификаторов.

        Назначение:
            Используется для справедливого сравнения U-Net с комитетами. Включает
            отсечение горизонта, условное уменьшение размера и дополнение до 224×224.

        Параметры:
            split_dir (str): путь к каталогу разбиения набора данных.
            crop_json_path (str): путь к JSON-файлу с координатами отсечения горизонта.
            keep_frac (float, optional): доля изображений для использования.
            seed (int, optional): зерно генератора случайных чисел.
            enable_resize (bool, optional): разрешить условное изменение размера.
            resize_to (tuple[int, int], optional): целевой размер (W, H).

        Возвращает:
            tuple[torch.Tensor, torch.Tensor]:
                подготовленное изображение и двухканальную маску.

        Пример использования:
            dataset = FireSmokeCocoCommitteeLike(test_dir, crop_test, keep_frac=1.0)
            image, mask = dataset[0]

    """
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

        # 1) Список имён файлов формируется в том же порядке, что и в экспериментах с комитетами
        file_names = [im["file_name"] for im in images_meta.values()]
        file_names = [fn for fn in file_names if fn in self.crop_map]

        # 2) Используется тот же тип генератора случайных чисел и та же логика выбора подмножества
        rng = np.random.default_rng(self.seed)
        file_names = _select_subset(file_names, self.keep_frac, rng)

        # 3) Метаданные сохраняются в фиксированном порядке
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

        # Отсечение области выше горизонта выполняется до изменения размера, как и в комитетах
        if crop_y > 0:
            img_rgb = img_rgb[crop_y:, :, :]
            fire = fire[crop_y:, :]
            smoke = smoke[crop_y:, :]

        # Условное уменьшение размера в стиле комитетов выполняется только для изображений, превышающих размер 224×224
        if self.enable_resize:
            Wt, Ht = self.resize_to
            H, W = img_rgb.shape[:2]
            if H > Ht or W > Wt:
                img_rgb = cv2.resize(img_rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)
                fire    = cv2.resize(fire.astype(np.uint8),  (Wt, Ht), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                smoke   = cv2.resize(smoke.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # Дополнение изображения и масок до размера 224×224, если они меньше
        img_rgb, fire, smoke = pad_to_224(img_rgb, fire, smoke, out_hw=(224,224))

        img = (img_rgb.astype(np.float32) / 255.0)
        img = torch.from_numpy(img.transpose(2,0,1))  # C,H,W

        # Двухканальная маска: первый канал — огонь, второй канал — дым
        mask = np.stack([fire, smoke], axis=0).astype(np.float32)
        mask = torch.from_numpy(mask)

        return img, mask

 
# Формирование обучающей, валидационной и тестовой выборок
train_ds = FireSmokeCocoCommitteeLike(train_dir, crop_train, keep_frac=KEEP, seed=SEED)
val_ds   = FireSmokeCocoCommitteeLike(valid_dir, crop_valid, keep_frac=KEEP, seed=SEED)
# Тестовая выборка используется полностью
test_ds  = FireSmokeCocoCommitteeLike(test_dir,  crop_test,  keep_frac=1.0, seed=SEED)  # если crop_test есть

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

device = get_device()

# Инициализация модели U-Net
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
    activation=None  # Возвращаются логиты без встроенной активации
).to(device)

bce  = nn.BCEWithLogitsLoss()
dice = smp.losses.DiceLoss(mode="multilabel")  # Два независимых выходных канала: огонь и дым

def loss_fn(logits, targets):
    """
    Вычисляет совмещённую функцию потерь для сегментации.

        Назначение:
            Комбинирует BCEWithLogitsLoss и DiceLoss для повышения устойчивости
            обучения на несбалансированных масках.

        Параметры:
            logits (torch.Tensor): сырые выходы модели.
            targets (torch.Tensor): целевые маски.

        Возвращает:
            torch.Tensor: значение функции потерь.

        Пример использования:
            loss = loss_fn(logits, targets)

    """
    return bce(logits, targets) + dice(logits, targets)

# Оптимизатор обучения модели
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def train_one_epoch():
    """
    Выполняет одну эпоху обучения модели U-Net.

        Назначение:
            Проходит по всем батчам обучающего загрузчика, вычисляет функцию потерь,
            обновляет параметры модели и возвращает среднее значение потерь.

        Возвращает:
            float: среднее значение функции потерь за эпоху.

        Пример использования:
            train_loss = train_one_epoch()

    """
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
    """
    Выполняет одну эпоху валидации модели U-Net.

        Назначение:
            Оценивает модель на валидационной выборке без обновления параметров.

        Возвращает:
            float: среднее значение функции потерь на валидации.

        Пример использования:
            val_loss = eval_one_epoch()

    """
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
    """
    Загружает COCO-аннотации для выбранного разбиения набора данных.

        Назначение:
            Преобразует JSON-файл аннотаций в удобные словари с метаданными
            изображений, категориями и списками аннотаций по каждому изображению.

        Параметры:
            split_dir (str): путь к каталогу разбиения набора данных.

        Возвращает:
            tuple[dict, dict, dict]:
                - словарь изображений;
                - словарь категорий;
                - словарь аннотаций, сгруппированных по идентификатору изображения.

        Пример использования:
            images, cats, anns_by_img = load_coco_split("/content/data/test")

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

def _ann_to_binary_mask(ann, h, w):
    """
    Преобразует одну COCO-аннотацию в бинарную маску.

        Назначение:
            Поддерживает полигональную разметку, сжатый и несжатый формат RLE.

        Параметры:
            ann (dict): объект аннотации COCO.
            h (int): высота маски.
            w (int): ширина маски.

        Возвращает:
            numpy.ndarray: бинарная маска формы (h, w) типа uint8.

        Пример использования:
            mask = _ann_to_binary_mask(annotation, 224, 224)

    """
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
    """
    Строит итоговые бинарные маски классов «огонь» и «дым» для изображения.

        Назначение:
            Объединяет все объектные аннотации изображения в две совокупные маски.

        Параметры:
            img_meta (dict): метаданные изображения.
            anns (list[dict]): список аннотаций данного изображения.
            cats_map (dict): отображение идентификаторов категорий в их имена.

        Возвращает:
            tuple[numpy.ndarray, numpy.ndarray]:
                - маска огня;
                - маска дыма.

        Пример использования:
            fire_mask, smoke_mask = build_fire_smoke_masks(img_meta, anns, cats_map)

    """
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
    """
    Возвращает полный путь к изображению в каталоге split-набора данных.

        Назначение:
            Учитывает два варианта структуры каталога: непосредственное размещение
            файлов и размещение внутри подпапки images.

        Параметры:
            split_dir (str): путь к каталогу разбиения.
            file_name (str): имя файла изображения.

        Возвращает:
            str: полный путь к изображению.

        Исключения:
            FileNotFoundError: если файл не найден.

        Пример использования:
            path = get_img_path(test_dir, "image_001.jpg")

    """
    p1 = os.path.join(split_dir, file_name)
    if os.path.exists(p1): return p1
    p2 = os.path.join(split_dir, "images", file_name)
    if os.path.exists(p2): return p2
    raise FileNotFoundError(file_name)

def load_crop_map(path):
    """
    Загружает словарь отсечения горизонта из JSON-файла.

        Назначение:
            Используется для применения одинаковой предобработки при сравнении
            U-Net и комитетов классификаторов.

        Параметры:
            crop_json_path (str): путь к JSON-файлу с координатами crop_y.

        Возвращает:
            dict: отображение вида {имя_файла: crop_y}.

        Пример использования:
            crop_map = load_crop_map(crop_test)

    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["crop_y_by_file"]

def preprocess_committee_like(img_rgb, fire, smoke, crop_y, resize_to=(224,224), enable_resize=True):
    """
    Применяет к изображению и маскам предобработку в стиле комитетов.

        Назначение:
            Выполняет отсечение горизонта, условное изменение размера и дополнение
            до 224×224 для согласованного сравнения разных подходов.

        Параметры:
            img_rgb (numpy.ndarray): исходное изображение RGB.
            fire (numpy.ndarray): маска огня.
            smoke (numpy.ndarray): маска дыма.
            crop_y (int): координата отсечения горизонта.
            resize_to (tuple[int, int], optional): целевой размер (W, H).
            enable_resize (bool, optional): включить условное изменение размера.

        Возвращает:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                обработанные изображение и маски.

        Пример использования:
            img_p, fire_p, smoke_p = preprocess_committee_like(img, fire, smoke, crop_y=12)

    """
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
    """
    Вычисляет бинарные метрики качества классификации.

        Назначение:
            Позволяет получить precision, recall, F1, accuracy и balanced accuracy
            по булевым маскам предсказаний и истинных меток.

        Параметры:
            pred_pos (torch.Tensor): предсказанные положительные объекты.
            true_pos (torch.Tensor): истинные положительные объекты.

        Возвращает:
            dict: словарь с метриками и значениями TP, FP, FN, TN.

        Пример использования:
            metrics = binary_metrics_from_pred(pred_fire, true_fire)

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

@torch.no_grad()
def make_3class_pred(fire_pos: torch.Tensor, smoke_pos: torch.Tensor):
    """
    Формирует трёхклассовое предсказание на основе двух бинарных масок.

        Назначение:
            Кодирует классы следующим образом:
            0 — норма, 1 — дым, 2 — огонь.
            При наличии огня он имеет приоритет над дымом.

        Параметры:
            fire_pos (torch.Tensor): бинарная маска класса «огонь».
            smoke_pos (torch.Tensor): бинарная маска класса «дым».

        Возвращает:
            torch.Tensor: тензор целых меток классов.

        Пример использования:
            y3 = make_3class_pred(pred_fire, pred_smoke)

    """
    y3 = torch.zeros_like(fire_pos, dtype=torch.int64)
    y3[smoke_pos] = 1
    y3[fire_pos]  = 2
    return y3

@torch.no_grad()
def confusion_3x3(y_true3, y_pred3):
    """
    Строит матрицу ошибок для трёхклассовой задачи.

        Параметры:
            y_true3 (torch.Tensor): истинные метки классов.
            y_pred3 (torch.Tensor): предсказанные метки классов.

        Возвращает:
            torch.Tensor: матрица ошибок размера 3×3.

        Пример использования:
            cm = confusion_3x3(y_true3, y_pred3)

    """
    cm = torch.zeros((3,3), dtype=torch.int64)
    for t in range(3):
        for p in range(3):
            cm[t,p] = ((y_true3==t) & (y_pred3==p)).sum()
    return cm

@torch.no_grad()
def prf_from_cm_3class(cm):
    """
    Вычисляет precision, recall и F1-score по матрице ошибок 3×3.

        Назначение:
            Возвращает метрики для классов «дым», «огонь», «норма», а также
            микро- и макроусреднённые показатели для классов опасности.

        Параметры:
            cm (torch.Tensor): матрица ошибок 3×3.

        Возвращает:
            dict: словарь с вычисленными метриками.

        Пример использования:
            report = prf_from_cm_3class(cm)

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

def image_detection_rate_from_images(pred_ratio_list, true_ratio_list, thresholds=(0.02,0.05,0.07,0.10)):
    """
    Оценивает долю обнаруженных изображений при заданных порогах.

        Назначение:
            Сравнивает предсказанную и истинную долю опасных пикселей на изображении
            и вычисляет показатели обнаружения на уровне изображения.

        Параметры:
            pred_ratio_list (list[float]): доли предсказанной опасности по изображениям.
            true_ratio_list (list[float]): истинные доли опасности по изображениям.
            thresholds (tuple[float], optional): список порогов обнаружения.

        Возвращает:
            dict: словарь показателей обнаружения.

        Пример использования:
            rep = image_detection_rate_from_images(pred_ratios, true_ratios)

    """
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

    """
    Вычисляет средние метрики по изображениям и показатели детекции.

        Назначение:
            Используется для агрегирования результатов на уровне изображений,
            включая средние значения precision, recall, F1 и показатели FPR/TPR.

        Параметры:
            tp_list, fp_list, fn_list (list[int]): значения TP, FP, FN по изображениям.
            pred_ratio_list (list[float]): предсказанная доля опасных пикселей.
            true_ratio_list (list[float]): истинная доля опасных пикселей.
            thresholds (tuple[float], optional): пороги срабатывания.

        Возвращает:
            dict: агрегированный отчёт по метрикам.

        Пример использования:
            rep = mean_per_image_and_detection_from_lists(tp, fp, fn, pred_ratios, true_ratios)

    """
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

# Если размер окна WINDOW равен 25
R = 12   # рамка (с каждой стороны) которую исключаем из метрик
images_meta, cats_map, anns_by_img = load_coco_split(TEST_DIR)
crop_map = load_crop_map(CROP_TEST_JSON)

# Порядок элементов соответствует JSON-файлу и экспериментам с комитетами
test_metas = list(images_meta.values())

def build_color_mask(fire_mask, smoke_mask, horizon_mask=None):
    """
    Строит цветную маску для визуализации предсказаний и разметки.

        Назначение:
            Отображает огонь красным цветом, дым — белым, а линию горизонта —
            синим, если соответствующая маска передана.

        Параметры:
            fire_mask (numpy.ndarray): бинарная маска огня.
            smoke_mask (numpy.ndarray): бинарная маска дыма.
            horizon_mask (numpy.ndarray | None): маска горизонта.

        Возвращает:
            numpy.ndarray: цветное изображение-маска формата RGB.

        Пример использования:
            color_mask = build_color_mask(fire_mask, smoke_mask, horizon_mask)

    """
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
    """
    Визуализирует одно изображение, истинную маску и предсказание U-Net.

        Назначение:
            Используется для качественного анализа результата сегментации.

        Параметры:
            idx (int): индекс изображения в тестовом наборе.
            thr_pixel (float, optional): порог бинаризации вероятностей.

        Возвращает:
            None

        Пример использования:
            visualize_one_sample(0, thr_pixel=0.5)

    """
    im = test_metas[idx]
    fn = im["file_name"]
    img_id = im["id"]

    img_path = get_img_path(TEST_DIR, fn)
    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    anns = anns_by_img.get(img_id, [])
    fire, smoke = build_fire_smoke_masks(im, anns, cats_map)

    crop_y = int(crop_map.get(fn, 0))

    # Маска горизонта до операции отсечения
    horizon_mask_full = np.zeros_like(fire, dtype=np.uint8)
    if crop_y > 0:
        horizon_mask_full[:crop_y, :] = 1

    # Предобработка изображения и масок
    img_p, fire_p, smoke_p = preprocess_committee_like(
        img_rgb, fire, smoke, crop_y=crop_y, resize_to=(224,224)
    )

    # Маска горизонта после отсечения и изменения размера
    horizon_mask = None
    if crop_y > 0:
        # создаём mask той же формы, что и fire_p
        horizon_mask = np.zeros_like(fire_p, dtype=np.uint8)
        # Верхняя строка после отсечения соответствует линии горизонта
        # Для наглядности линия горизонта выделяется тонкой полосой
        horizon_mask[0:2, :] = 1  # тонкая синяя линия

    # Получение предсказания модели
    x = torch.from_numpy((img_p.astype(np.float32)/255.0).transpose(2,0,1)).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x))[0].detach().cpu()

    pred_fire  = (prob[0] >= thr_pixel).numpy()
    pred_smoke = (prob[1] >= thr_pixel).numpy()

    # Формирование цветовых масок для отображения
    gt_color  = build_color_mask(fire_p, smoke_p, horizon_mask=horizon_mask)
    pred_color = build_color_mask(pred_fire, pred_smoke, horizon_mask=horizon_mask)

    # Визуализация результата
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
    """
    Применяет предобработку в стиле комитетов только к изображению.

        Назначение:
            Используется в инференсе U-Net, когда истинные маски отсутствуют.

        Параметры:
            img_rgb (numpy.ndarray): исходное RGB-изображение.
            crop_y (int): координата отсечения горизонта.
            resize_to (tuple[int, int], optional): целевой размер.
            enable_resize (bool, optional): включить условное изменение размера.

        Возвращает:
            numpy.ndarray: подготовленное изображение.

        Пример использования:
            img_p = preprocess_committee_like_image_only(img_rgb, crop_y=10)

    """
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
    Выполняет инференс U-Net для одного кадра.

        Назначение:
            Применяет предобработку, запускает модель и возвращает бинарные маски
            классов «огонь» и «дым» на изображении размера 224×224.

        Параметры:
            img_rgb (numpy.ndarray): исходное RGB-изображение.
            crop_y (int): координата отсечения горизонта.
            thr_pixel (float, optional): порог бинаризации вероятностей.

        Возвращает:
            tuple[torch.Tensor, torch.Tensor]:
                предсказанные маски огня и дыма.

        Пример использования:
            pred_fire, pred_smoke = unet_process_frame(img, crop_y=15, thr_pixel=0.5)

    """
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
    """
    Считывает изображение с диска и возвращает его в формате RGB.

        Параметры:
            path (str): путь к файлу изображения.

        Возвращает:
            numpy.ndarray: изображение RGB.

        Пример использования:
            img = read_rgb("/content/image.jpg")

    """
    img = np.array(Image.open(path).convert("RGB"))
    return img



def benchmark_with_disk_io(file_list, crop_map, thr_pixel=0.5, warmup=20, n_measure=200):
    """
    Измеряет скорость работы U-Net с учётом загрузки изображений с диска.

        Назначение:
            Возвращает показатели производительности в изображениях в секунду и
            пикселях в секунду при полном конвейере обработки.

        Параметры:
            file_list (list[str]): список путей к изображениям.
            crop_map (dict): словарь crop_y по имени файла.
            thr_pixel (float, optional): порог бинаризации.
            warmup (int, optional): число прогревочных запусков.
            n_measure (int, optional): число измеряемых изображений.

        Возвращает:
            dict: словарь с результатами бенчмарка.

        Пример использования:
            stats = benchmark_with_disk_io(paths, crop_map, thr_pixel=0.5)

    """
    # Разогрев графического процессора перед измерением скорости
    for i in range(min(warmup, len(file_list))):
        fn = file_list[i]
        img = read_rgb(fn)
        crop_y = int(compute_crop_y_original(img))
        _ = unet_process_frame(img, crop_y, thr_pixel=thr_pixel)
    if device == "cuda":
        torch.cuda.synchronize()

    # Основной этап измерения производительности
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
    """
    Измеряет скорость работы U-Net при заранее загруженных изображениях.

        Назначение:
            Позволяет исключить вклад дисковой подсистемы и измерить чистую
            производительность инференса и предобработки.

        Параметры:
            images_rgb (list[numpy.ndarray]): список изображений в памяти.
            crop_y_list (list[int]): список координат отсечения горизонта.
            thr_pixel (float, optional): порог бинаризации.
            warmup (int, optional): число прогревочных запусков.
            n_measure (int, optional): число измерений.

        Возвращает:
            dict: словарь с результатами бенчмарка.

        Пример использования:
            stats = benchmark_in_memory(images_rgb, crop_list, thr_pixel=0.5)

    """
    # warmup
    for i in range(min(warmup, len(images_rgb))):
        _ = unet_process_frame(images_rgb[i], crop_y_list[i], thr_pixel=thr_pixel)
    if device == "cuda":
        torch.cuda.synchronize()

    # Основной этап измерения производительности
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
# Бенчмарк скорости U-Net в формате, сопоставимом с метрикой скорости комитетов
# Используется та же идея: px_per_sec = N / core_time, где core_time не включает загрузку, отсечение горизонта и изменение размера
#
# Для комитетов core-время включает извлечение признаков, нормализацию, инференс и логику комитета
# Для U-Net core-время включает инференс, вычисление сигмоиды и пороговую бинаризацию
#
# Дополнительно вычисляется полное время TOTAL, включающее загрузку, отсечение горизонта, изменение размера/дополнение и core-время
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Пути к данным
TEST_DIR = "/content/fire-smoke-segmentation.v4i.coco-segmentation/test"

# Настройки эксперимента
thr_pixel = 0.5
WINDOW = 25
R = WINDOW // 2  # 12

# Важно: модель U-Net ожидает вход размером 224×224
resize_to = (224, 224)  # (W,H)

# Сопоставимое количество пикселей для сравнения с комитетами
# В комитетах используется N — число центральных точек; при stride=1 это соответствует внутренним пикселям
N_PIX = (resize_to[1] - 2*R) * (resize_to[0] - 2*R)  # (H-2R)*(W-2R)

WARMUP = 20
N_MEASURE = 125  # set bigger if you want stable mean

def _now():
    """
    Возвращает текущее значение высокоточного таймера.

        Назначение:
            Используется для измерения временных интервалов при бенчмарке.

        Возвращает:
            float: значение таймера.

        Пример использования:
            t0 = _now()

    """
    return time.perf_counter()

def _sync():
    """
    Синхронизирует очередь операций графического процессора.

        Назначение:
            Необходима для корректного измерения времени на CUDA.

        Возвращает:
            None

        Пример использования:
            _sync()

    """
    if device.type == "cuda":
        torch.cuda.synchronize()

def _to_u8_rgb(img):
    """
    Преобразует входное изображение к типу uint8 и формату RGB.

        Назначение:
            Унифицирует входные данные перед предобработкой и инференсом.

        Параметры:
            img: изображение в формате PIL или NumPy.

        Возвращает:
            numpy.ndarray: изображение формата RGB типа uint8.

        Пример использования:
            img_u8 = _to_u8_rgb(img)

    """
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def pad_to_224_rgb(img_rgb, out_hw=(224,224)):
    """
    Дополняет RGB-изображение до размера 224×224.

        Параметры:
            img_rgb (numpy.ndarray): исходное изображение.
            out_hw (tuple[int, int], optional): целевой размер.

        Возвращает:
            numpy.ndarray: дополненное изображение.

        Пример использования:
            img224 = pad_to_224_rgb(img_rgb)

    """
    Ht, Wt = out_hw[1], out_hw[0]
    H, W = img_rgb.shape[:2]
    pad_h = max(0, Ht - H)
    pad_w = max(0, Wt - W)
    if pad_h == 0 and pad_w == 0:
        return img_rgb
    return np.pad(img_rgb, ((0,pad_h),(0,pad_w),(0,0)), mode="constant", constant_values=0)

def preprocess_unet_like_committee(img0_u8):
    """
    Подготавливает изображение для U-Net по схеме, близкой к комитетам.

        Назначение:
            Выполняет отсечение горизонта, изменение размера до 224×224 и
            дополнительное выравнивание размера при необходимости.

        Параметры:
            img0_u8 (numpy.ndarray): исходное изображение uint8.

        Возвращает:
            tuple[numpy.ndarray, int]:
                подготовленное изображение 224×224 и значение crop_y.

        Пример использования:
            img224, crop_y = preprocess_unet_like_committee(img0)

    """
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

    # Приведение изображения и масок к заданному размеру to 224x224 (UNet input)
    img_r = cv2.resize(img_c, resize_to, interpolation=cv2.INTER_AREA)

    # pad (usually not needed after resize, but keep consistent)
    img_p = pad_to_224_rgb(img_r, out_hw=resize_to)
    return img_p, crop_y

@torch.no_grad()
def unet_core_infer(img_224_u8, thr_pixel=0.5):
    """
    Выполняет основную часть инференса U-Net.

        Назначение:
            Включает перенос данных на устройство, прямой проход модели,
            вычисление сигмоиды и пороговую бинаризацию.

        Параметры:
            img_224_u8 (numpy.ndarray): изображение размера 224×224.
            thr_pixel (float, optional): порог бинаризации вероятностей.

        Возвращает:
            tuple[torch.Tensor, torch.Tensor]:
                бинарные маски классов «огонь» и «дым».

        Пример использования:
            pred_fire, pred_smoke = unet_core_infer(img224, thr_pixel=0.5)

    """
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
    Выполняет бенчмарк скорости U-Net в формате, сопоставимом с комитетами.

        Назначение:
            Отдельно учитывает core-время инференса и полное время конвейера,
            включая загрузку, отсечение горизонта и изменение размера.

        Параметры:
            paths (list[str]): список путей к изображениям.
            thr_pixel (float, optional): порог бинаризации.
            warmup (int, optional): количество прогревочных запусков.
            n_measure (int, optional): число измеряемых изображений.

        Возвращает:
            dict: словарь с усреднёнными временами и производительностью.

        Пример использования:
            res = benchmark_unet_like_committee(paths, thr_pixel=0.5, n_measure=100)

    """
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

    # Разогрев графического процессора перед основными измерениями
    img0 = _to_u8_rgb(Image.open(paths[0]).convert("RGB"))
    img224, _ = preprocess_unet_like_committee(img0)
    for _ in range(min(warmup, 50)):
        _ = unet_core_infer(img224, thr_pixel=thr_pixel)
    _sync()

    # Основной цикл измерений
    t_load_sum = 0.0
    t_crop_sum = 0.0
    t_resize_sum = 0.0
    t_core_sum = 0.0
    t_total_sum = 0.0

    for i in tqdm(range(n), desc="UNet speed", dynamic_ncols=True):
        p = paths[i]

        t0_total = _now()

        # Загрузка изображения
        t0 = _now()
        img0 = _to_u8_rgb(Image.open(p).convert("RGB"))
        t_load = _now() - t0

        # Отсечение горизонта и изменение размера с раздельным учётом времени, как в комитетах
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

        # Основная часть инференса с синхронизацией графического процессора
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

    # Усреднённые значения времени
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

# Сбор путей к тестовым изображениям
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