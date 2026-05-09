# DL experiments: описание экспериментов и результатов

## Исходные ноутбуки

- [DL_Experiments_MURA_DINOv2_Adapters_v3_Sunnatilla.ipynb](./DL_Experiments_MURA_DINOv2_Adapters_v3_Sunnatilla.ipynb)
- [DL_Experiments_densenet_Leona.ipynb](./DL_Experiments_densenet_Leona.ipynb)

## Общий контекст

Оба ноутбука решают задачу бинарной классификации MURA: определить, является ли рентгеновское исследование/изображение нормальным или патологическим. Используются 7 анатомических категорий:

- `XR_WRIST`
- `XR_ELBOW`
- `XR_SHOULDER`
- `XR_FINGER`
- `XR_FOREARM`
- `XR_HUMERUS`
- `XR_HAND`

Основная метрика в экспериментах - Cohen's kappa. Также считаются accuracy, AUC и, в DINOv2-ноутбуке, F1.

Важное ограничение сравнения: DenseNet-ноутбук агрегирует предсказания по исследованию через среднюю вероятность изображений одного `study_id`, а DINOv2-ноутбук, судя по коду, считает метрики по строкам `VAL_CSV`, то есть по изображениям, если CSV не был заранее агрегирован. Поэтому абсолютные значения DenseNet и DINOv2 нужно сравнивать осторожно.

## Эксперимент 1: DINOv2-Large, SimpleMURA

Источник: [DL_Experiments_MURA_DINOv2_Adapters_v3_Sunnatilla.ipynb](./DL_Experiments_MURA_DINOv2_Adapters_v3_Sunnatilla.ipynb)

### Цель

Улучшить предыдущую версию модели `v1`, где общий kappa был `0.6348`. В начале ноутбука ожидался диапазон `0.76-0.82` overall kappa, с заметным ростом слабых категорий `XR_FINGER`, `XR_HAND` и `XR_SHOULDER`.

### Данные и окружение

- Среда: Google Colab.
- GPU: NVIDIA L4.
- VRAM: 23.7 GB.
- Размер изображения: `448x448`.
- Train: `36,808` изображений.
- Validation: `3,197` изображений.
- Batch size: `8`.
- Gradient accumulation: `16`.
- Эффективный batch size: `128`.
- Количество эпох: `15`.

### Архитектура

Несмотря на название файла с `Adapters`, в фактическом коде реализована модель `SimpleMURA` без отдельного adapter-модуля:

- backbone: `facebook/dinov2-large`;
- DINOv2-Large изначально полностью заморожен;
- используется CLS-токен `last_hidden_state[:, 0, :]`;
- поверх CLS-токена стоит общий head:
  - `Linear(1024, 256)`;
  - `LayerNorm`;
  - `GELU`;
  - `Dropout(0.3)`;
  - `Linear(256, 64)`;
  - `GELU`;
  - `Dropout(0.2)`;
- для каждой анатомии отдельный бинарный классификатор `Linear(64, 2)`.

### Аугментации и балансировка

В ноутбуке выделены слабые категории:

- `XR_FINGER`
- `XR_HAND`
- `XR_SHOULDER`
- `XR_FOREARM`

Для них используется более сильный train-transform:

- horizontal flip;
- rotation до 20 градусов;
- affine scale/translate;
- elastic transform;
- CLAHE;
- random gamma;
- brightness/contrast;
- noise;
- blur.

Для остальных категорий применяется более мягкая аугментация:

- horizontal flip;
- rotation до 10 градусов;
- CLAHE;
- random gamma;
- brightness/contrast.

Также используется `WeightedRandomSampler` с повышенными весами для слабых категорий:

| Категория | Вес |
|---|---:|
| XR_FINGER | 1.5 |
| XR_HAND | 1.5 |
| XR_SHOULDER | 1.3 |
| XR_FOREARM | 1.2 |
| XR_ELBOW | 1.0 |
| XR_HUMERUS | 1.0 |
| XR_WRIST | 1.0 |

`POS_WEIGHT` для положительного класса составил `1.475`.

### Обучение

Использовалась функция потерь `ProgressiveLoss`:

- эпохи 1-5: BCE с label smoothing `0.1`;
- эпохи 6-9: смесь BCE и focal loss;
- после 9-й эпохи: focal loss с `gamma=1.5`.

Оптимизация:

- `AdamW`;
- learning rate `1e-3` для head и классификаторов;
- weight decay `0.01`;
- cosine annealing scheduler;
- mixed precision через AMP;
- gradient clipping `1.0`;
- early stopping с patience `5`.

План постепенной разморозки DINOv2:

| Эпоха | Размораживаемые блоки |
|---:|---|
| 5 | последние 4 блока |
| 8 | последние 8 блоков |
| 11 | последние 12 блоков |

Для размороженных параметров добавлялся learning rate `1e-6`.

### Оценка

Для каждой анатомии подбирался лучший threshold по kappa на диапазоне `0.05-0.95` с шагом `0.01`. В коде реализован TTA, но запуск TTA был прерван `KeyboardInterrupt`, поэтому финальная сохранённая модель использует метод `No TTA`.

В конце также добавлен код для CV-подбора threshold через `StratifiedKFold`, но сохранённый output показывает только начало выполнения: `Загрузка лучшей модели...`. Завершённых CV-результатов в ноутбуке нет.

### Результаты DINOv2 v3 без TTA

| Категория | v1 kappa | v3 kappa | AUC | F1 | Accuracy | Threshold | N | Delta kappa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XR_WRIST | 0.7118 | 0.7380 | 0.9226 | 0.8444 | 0.8725 | 0.56 | 659 | +0.0262 |
| XR_ELBOW | 0.7202 | 0.7457 | 0.9284 | 0.8599 | 0.8731 | 0.57 | 465 | +0.0255 |
| XR_SHOULDER | 0.5664 | 0.6092 | 0.8655 | 0.8036 | 0.8046 | 0.49 | 563 | +0.0428 |
| XR_FINGER | 0.5371 | 0.6442 | 0.8812 | 0.8285 | 0.8221 | 0.48 | 461 | +0.1071 |
| XR_FOREARM | 0.6813 | 0.7675 | 0.9311 | 0.8763 | 0.8837 | 0.49 | 301 | +0.0862 |
| XR_HUMERUS | 0.7156 | 0.7846 | 0.9257 | 0.8897 | 0.8924 | 0.58 | 288 | +0.0690 |
| XR_HAND | 0.5646 | 0.6164 | 0.8686 | 0.7500 | 0.8217 | 0.46 | 460 | +0.0518 |
| OVERALL | 0.6348 | 0.6874 | 0.9040 | 0.8304 | 0.8445 | 0.49 | 3197 | +0.0526 |

### Вывод по DINOv2

Модель улучшила общий kappa с `0.6348` до `0.6874`, то есть на `+0.0526`. Все категории улучшились относительно `v1`, самый большой рост получился на:

- `XR_FINGER`: `+0.1071`;
- `XR_FOREARM`: `+0.0862`;
- `XR_HUMERUS`: `+0.0690`.

Цель `0.80` overall kappa не достигнута: до неё осталось `0.1126`. Из ожидаемых диапазонов лучше всего совпал `XR_HUMERUS`, а слабые категории `XR_SHOULDER`, `XR_FINGER` и `XR_HAND` всё ещё заметно ниже целевых значений.

## Эксперимент 2: DenseNet121

Источник: [DL_Experiments_densenet_Leona.ipynb](./DL_Experiments_densenet_Leona.ipynb)

### Данные и общий pipeline

Ноутбук использует предварительно подготовленный датасет после resize и улучшения качества. Через `build_exp_dataframe(DATA_ROOT, return_images=True)` было загружено `37,108` изображений.

После загрузки формируется `unique_study_id`:

```python
df_images["unique_study_id"] = df_images["patient_id"] + "_" + df_images["study_id"]
```

Далее данные делятся на `train` и `valid`. Во всех экспериментах validation-предсказания агрегируются на уровень исследования:

- модель выдаёт вероятность для каждого изображения;
- для одного `study_id` берётся средняя вероятность;
- label и anatomy берутся как первые значения группы;
- метрики считаются уже по агрегированному `study_df`.

Это ближе к стандартной MURA-оценке на уровне исследования.

### Общие функции

В ноутбуке определены:

- `MURADataset` для чтения изображений через PIL и применения torchvision transforms;
- `train_one_epoch`;
- `validate`;
- `aggregate`;
- `compute_metrics`;
- `threshold`, которая перебирает `200` threshold-значений от `0` до `1` и выбирает threshold с максимальным kappa.

Во всех базовых DenseNet-экспериментах используется:

- `DenseNet121` с ImageNet-весами;
- бинарный classifier `Linear(in_features, 1)`;
- `BCEWithLogitsLoss`;
- `Adam`;
- batch size `64`;
- validation batch size `64`;
- нормализация `Normalize([0.485]*3, [0.229]*3)`.

### DenseNet: сводная таблица экспериментов

| Эксперимент | Основное изменение | Best threshold | Accuracy | AUC | Kappa |
|---|---|---:|---:|---:|---:|
| Baseline | Разморожен только `denseblock4` и classifier | 0.4774 | 0.8204 | 0.8817 | 0.6355 |
| Weighted loss | Baseline + `pos_weight=1.1297` | 0.4070 | 0.8122 | 0.8728 | 0.6232 |
| 2 blocks | Разморожены `denseblock3`, `denseblock4` и classifier | 0.4874 | 0.8184 | 0.8756 | 0.6316 |
| Augmented | 2 blocks + усиленные аугментации для слабых анатомий | 0.4121 | 0.7936 | 0.8628 | 0.5818 |
| Sampler | 2 blocks + oversampling слабых анатомий | 0.4171 | 0.8142 | 0.8698 | 0.6246 |
| Adaptive threshold | Загружен `blocks2_DenseNet.pth`, threshold подобран отдельно для каждой анатомии | per anatomy | - | - | 0.6622 |

### Baseline DenseNet121

Настройка:

- `DenseNet121_Weights.IMAGENET1K_V1`;
- все параметры заморожены;
- разморожен только `denseblock4`;
- classifier заменён на `Linear(..., 1)`;
- optimizer: `Adam`, learning rate `1e-5`;
- loss: обычный `BCEWithLogitsLoss`;
- сохранение модели: `base_DenseNet.pth`.

Лучший threshold:

- `0.47738693467336685`.

Итоговые метрики:

| Accuracy | AUC | Kappa |
|---:|---:|---:|
| 0.8204 | 0.8817 | 0.6355 |

Метрики по анатомиям:

| Категория | Accuracy | AUC | Kappa |
|---|---:|---:|---:|
| XR_WRIST | 0.8696 | 0.9112 | 0.7265 |
| XR_SHOULDER | 0.8395 | 0.8995 | 0.6780 |
| XR_FOREARM | 0.8058 | 0.9042 | 0.6139 |
| XR_HAND | 0.8030 | 0.8500 | 0.5627 |
| XR_FINGER | 0.7636 | 0.8312 | 0.5232 |
| XR_HUMERUS | 0.8584 | 0.9066 | 0.7169 |
| XR_ELBOW | 0.7909 | 0.8886 | 0.5796 |

Вывод из ноутбука: уже на baseline DenseNet заметно превзошла предыдущие ML-эксперименты.

### DenseNet с учётом дисбаланса классов

Настройка:

- архитектура как в baseline;
- `pos_weight = neg / pos = 1.1296703296703297`;
- loss: `BCEWithLogitsLoss(pos_weight=...)`;
- сохранение модели: `weighted_DenseNet.pth`.

Лучший threshold:

- `0.40703517587939697`.

Итоговые метрики:

| Accuracy | AUC | Kappa |
|---:|---:|---:|
| 0.8122 | 0.8728 | 0.6232 |

Метрики по анатомиям:

| Категория | Accuracy | AUC | Kappa |
|---|---:|---:|---:|
| XR_WRIST | 0.8478 | 0.9093 | 0.6916 |
| XR_SHOULDER | 0.7963 | 0.8783 | 0.5942 |
| XR_FOREARM | 0.8544 | 0.8996 | 0.7093 |
| XR_HAND | 0.8030 | 0.8369 | 0.5806 |
| XR_FINGER | 0.7515 | 0.8098 | 0.5024 |
| XR_HUMERUS | 0.8142 | 0.8893 | 0.6275 |
| XR_ELBOW | 0.8364 | 0.9124 | 0.6722 |

Вывод: учёт дисбаланса улучшил `XR_FOREARM`, `XR_HAND` и `XR_ELBOW`, но ухудшил общий kappa относительно baseline.

### DenseNet с разморозкой двух последних блоков

Настройка:

- `DenseNet121`;
- разморожены `features.denseblock3` и `features.denseblock4`;
- classifier обучается отдельно;
- optimizer:
  - параметры feature-блоков: learning rate `1e-6`;
  - classifier: learning rate `1e-5`;
- сохранение модели: `blocks2_DenseNet.pth`.

Лучший threshold:

- `0.48743718592964824`.

Итоговые метрики:

| Accuracy | AUC | Kappa |
|---:|---:|---:|
| 0.8184 | 0.8756 | 0.6316 |

Метрики по анатомиям:

| Категория | Accuracy | AUC | Kappa |
|---|---:|---:|---:|
| XR_WRIST | 0.8804 | 0.9222 | 0.7510 |
| XR_SHOULDER | 0.7840 | 0.8778 | 0.5670 |
| XR_FOREARM | 0.8252 | 0.9008 | 0.6529 |
| XR_HAND | 0.7955 | 0.8445 | 0.5473 |
| XR_FINGER | 0.7697 | 0.8269 | 0.5357 |
| XR_HUMERUS | 0.8496 | 0.9094 | 0.6990 |
| XR_ELBOW | 0.8273 | 0.8803 | 0.6525 |

Вывод из ноутбука на этом этапе:

- baseline лучше для `XR_SHOULDER` и `XR_HUMERUS`;
- weighted loss лучше для `XR_FOREARM`, `XR_HAND`, `XR_ELBOW`;
- 2 blocks лучше для `XR_WRIST` и `XR_FINGER`.

### Аугментация для проседающих анатомий

Слабыми категориями были выбраны:

- `XR_SHOULDER`
- `XR_HAND`
- `XR_FINGER`

Для них добавлены более выраженные преобразования:

- horizontal flip;
- rotation до 10 градусов;
- random affine translate `(0.03, 0.03)`;
- color jitter по brightness и contrast `0.05`.

Для остальных категорий применялись обычные `RandomHorizontalFlip`, `RandomRotation`, `ToTensor`, `Normalize`.

Архитектура: DenseNet121 с размороженными `denseblock3` и `denseblock4`.

Сохранение модели: `augm_DenseNet.pth`.

Лучший threshold:

- `0.4120603015075377`.

Итоговые метрики:

| Accuracy | AUC | Kappa |
|---:|---:|---:|
| 0.7936 | 0.8628 | 0.5818 |

Метрики по анатомиям:

| Категория | Accuracy | AUC | Kappa |
|---|---:|---:|---:|
| XR_WRIST | 0.8207 | 0.8859 | 0.6284 |
| XR_SHOULDER | 0.7716 | 0.8699 | 0.5442 |
| XR_FOREARM | 0.8252 | 0.8992 | 0.6525 |
| XR_HAND | 0.7424 | 0.8162 | 0.4174 |
| XR_FINGER | 0.7758 | 0.8321 | 0.5477 |
| XR_HUMERUS | 0.8319 | 0.9185 | 0.6633 |
| XR_ELBOW | 0.8000 | 0.8879 | 0.5969 |

Вывод: аугментация дала небольшое улучшение только для `XR_FINGER`, но общий результат заметно просел.

### Oversampling слабых анатомий

Настройка:

- слабые категории: `XR_SHOULDER`, `XR_HAND`, `XR_FINGER`;
- `WeightedRandomSampler`;
- вес `2.0` для слабых категорий;
- вес `1.0` для остальных;
- архитектура DenseNet121 с размороженными `denseblock3` и `denseblock4`;
- сохранение модели: `sampler_DenseNet.pth`.

Лучший threshold:

- `0.4170854271356784`.

Итоговые метрики:

| Accuracy | AUC | Kappa |
|---:|---:|---:|
| 0.8142 | 0.8698 | 0.6246 |

Метрики по анатомиям:

| Категория | Accuracy | AUC | Kappa |
|---|---:|---:|---:|
| XR_WRIST | 0.8533 | 0.9172 | 0.6960 |
| XR_SHOULDER | 0.8210 | 0.8926 | 0.6425 |
| XR_FOREARM | 0.8350 | 0.8838 | 0.6713 |
| XR_HAND | 0.7803 | 0.8238 | 0.5107 |
| XR_FINGER | 0.7758 | 0.8437 | 0.5486 |
| XR_HUMERUS | 0.8230 | 0.8937 | 0.6454 |
| XR_ELBOW | 0.8091 | 0.8528 | 0.6167 |

Вывод: oversampling лучше аугментации по overall kappa, но всё ещё ниже baseline и 2-block модели с обычным threshold-подбором.

### Adaptive threshold по анатомиям

В финальном эксперименте загружается модель `blocks2_DenseNet.pth`, затем threshold подбирается отдельно для каждой анатомии.

Итог:

| Метрика | Значение |
|---|---:|
| Total kappa | 0.6622 |

Метрики по анатомиям после применения индивидуальных threshold:

| Категория | Accuracy | AUC | Kappa |
|---|---:|---:|---:|
| XR_WRIST | 0.8804 | 0.9222 | 0.7510 |
| XR_SHOULDER | 0.8086 | 0.8778 | 0.6186 |
| XR_FOREARM | 0.8641 | 0.9008 | 0.7279 |
| XR_HAND | 0.8106 | 0.8445 | 0.5883 |
| XR_FINGER | 0.7879 | 0.8269 | 0.5703 |
| XR_HUMERUS | 0.8584 | 0.9094 | 0.7169 |
| XR_ELBOW | 0.8273 | 0.8803 | 0.6527 |

Это лучший DenseNet-результат в ноутбуке по total kappa, но threshold подбирается и оценивается на одном и том же validation set. Поэтому результат может быть оптимистичным; для честной оценки нужен отдельный hold-out или cross-validation threshold-подбор.

## DenseNet: сравнение kappa по анатомиям

| Категория | Baseline | Weighted | 2 blocks | Augmented | Sampler | Adaptive threshold |
|---|---:|---:|---:|---:|---:|---:|
| XR_WRIST | 0.7265 | 0.6916 | 0.7510 | 0.6284 | 0.6960 | 0.7510 |
| XR_SHOULDER | 0.6780 | 0.5942 | 0.5670 | 0.5442 | 0.6425 | 0.6186 |
| XR_FOREARM | 0.6139 | 0.7093 | 0.6529 | 0.6525 | 0.6713 | 0.7279 |
| XR_HAND | 0.5627 | 0.5806 | 0.5473 | 0.4174 | 0.5107 | 0.5883 |
| XR_FINGER | 0.5232 | 0.5024 | 0.5357 | 0.5477 | 0.5486 | 0.5703 |
| XR_HUMERUS | 0.7169 | 0.6275 | 0.6990 | 0.6633 | 0.6454 | 0.7169 |
| XR_ELBOW | 0.5796 | 0.6722 | 0.6525 | 0.5969 | 0.6167 | 0.6527 |

## Итоговое сравнение

| Направление | Лучший результат | Kappa | Комментарий |
|---|---|---:|---|
| DINOv2 | SimpleMURA DINOv2-Large без TTA | 0.6874 | Лучший общий результат среди двух ноутбуков, но метрика считается по validation-строкам DINOv2 pipeline |
| DenseNet | Adaptive threshold поверх `blocks2_DenseNet.pth` | 0.6622 | Лучший DenseNet-результат, но threshold подобран на том же validation set |
| DenseNet без per-anatomy threshold | Baseline DenseNet121 | 0.6355 | Лучший честный DenseNet-вариант из основных экспериментов |

Общий вывод:

- DINOv2 v3 дал самый высокий reported kappa: `0.6874`.
- DenseNet baseline оказался сильной и простой отправной точкой: `0.6355`.
- Подбор индивидуальных thresholds для DenseNet поднял kappa до `0.6622`, но требует более строгой валидации.
- Аугментация слабых анатомий в текущем виде ухудшила overall качество.
- Oversampling слабых анатомий оказался лучше targeted augmentation, но не превзошёл baseline.
- Самые проблемные категории в обоих направлениях остаются `XR_HAND`, `XR_FINGER` и `XR_SHOULDER`.

