# MLflow checkpoint по MURA

> 🧭 [README](README.md) · **Чекпойнт 7 — MLflow** · ← [CP6 · Deep Learning](DL_Experiments.md) · [Анализ ошибок](docs/error_analysis.md) · [Устойчивость](docs/robustness_analysis.md) →

- [Общее описание checkpoint](#общее-описание-checkpoint)
- [Финальная модель](#финальная-модель)
- [Инфраструктура MLflow](#инфраструктура-mlflow)
- [Основной MLflow notebook](#основной-mlflow-notebook)
- [Логирование экспериментов в MLflow](#логирование-экспериментов-в-mlflow)
  - [Параметры](#параметры)
  - [Метрики](#метрики)
  - [Артефакты](#артефакты)
- [Воспроизводимость](#воспроизводимость)
- [Анализ ошибок](#анализ-ошибок)
- [Сравнение с baseline](#сравнение-с-baseline)
- [Проверка на устойчивость](#проверка-на-устойчивость)
- [Чистый PRD notebook](#чистый-prd-notebook)

## Общее описание checkpoint

Цель checkpoint — подготовить и проверить воспроизводимый MLflow для финальной DL-модели по задаче бинарной классификации MURA:

- выбрать лучшую модель
- переобучить выбранную модель с MLflow
- логгировать ключевые параметры и метрики
- сохранить артефакты в S3
- провести анализ ошибок модели
- сравнить с baseline
- проверить устойчивость (robustness)
- подготовить отдельный чистый notebook для загрузки PRD-модели и тестового predict

Основная задача модели: классификация рентгеновских изображений MURA на два класса:

- `normal`;
- `abnormal`.

Основная метрика проекта: **Cohen's kappa**. Дополнительно используются accuracy, F1, ROC-AUC и PR-AUC.
## Финальная модель

Финальная выбранная модель INOv2-large с threshold `0.480`. Она показала преимущество в оцениваемых метриках по сравнению с остальными экспериментами.

Порог выбран на internal validation по study-level предсказаниям.

Итоговые метрики:

| Уровень | Accuracy | F1 | Cohen's kappa | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Image-level | 0.841 | 0.824 | 0.680 | 0.903 | 0.912 |
| Study-level | 0.853 | 0.827 | 0.700 | 0.911 | 0.911 |


## Инфраструктура MLflow

Для версионирования экспериментов используется локальный MLflow stack в Docker. Инфраструктура находится в папке:

```text
infra/mlflow
```

Стек состоит из трёх основных сервисов:

| Сервис | Назначение |
|---|---:|
| PostgreSQL | Backend store для MLflow: хранит experiments, runs, параметры, метрики, tags и registry metadata |
| MinIO | Локальное S3-compatible object storage: хранит artifacts моделей, графики, CSV, checkpoints и другие файлы |
| MLflow Server | Tracking server и UI: принимает логи из notebook, показывает runs, метрики, параметры, artifacts и модели |

Дополнительно используется служебный контейнер minio-create-bucket, который автоматически создаёт bucket для MLflow artifacts при запуске инфраструктуры.

### Запуск инфраструктуры

1. Создать env-файл:

```bash
cd infra/mlflow
cp .env.example .env
```
Файл .env содержит локальные настройки PostgreSQL, MinIO, bucket name и портов.

2. Поднять Postgres, MinIO и MLflow:

```bash
docker compose up -d
```
3. Проверить состояние контейнеров:
```bash
docker compose ps
```
4. После запуска доступны:

- MLflow UI: `http://localhost:5050`
- MinIO Console: `http://localhost:9001`
- MinIO API: `http://localhost:9000`

MLflow UI используется для проверки experiments, runs, metrics, parameters, artifacts и registered models.

MinIO Console используется для проверки физического хранения artifacts в локальном S3-compatible bucket.

## Основной MLflow notebook

[CP_7_02_MURA_DINOv2_MLflow_Colab_with_enhanced.ipynb](https://github.com/Year-project-team-2-Topic-12/Project2025/blob/b83e3835fcabaf4e962fbac7ac4fd799e1909f30/notebooks/CP_7_02_MURA_DINOv2_MLflow_Colab_with_enhanced.ipynb)

Основной notebook выполняет полный pipeline:

1. подключение к MLflow Tracking Server;
2. подготовка MURA dataset;
3. обучение или дообучение финальной модели;
4. evaluation на test split;
5. логирование параметров и метрик;
6. сохранение артефактов;
7. анализ ошибок;
8. robustness check;
9. регистрация финальной модели как PRD.

## Логирование экспериментов в MLflow

### Параметры

В MLflow логируются ключевые параметры эксперимента:

| Группа | Некоторые из параметров                                                |
|---|-------------------------------------------------------------------------|
| Модель | `model_name`, `backbone`                                                |
| Данные | `dataset`, `image_size`, `split_strategy`, `study_leakage`, `bone_categories` |
| Обучение | `epochs`, `batch_size`, `learning_rate`, `unfreeze_schedule`,`seed`, `weight_decay`, |
| Loss | `pos_weight`, `label_smoothing`                  |
| Inference | `threshold`, `preprocessing`                           |


### Метрики

| Группа                       | Метрики                                                                                                 |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| Train                        | `accuracy`, `precision`, `recall`, `f1`, `kappa`, `roc_auc`,`pr_auc`,`composite`, `loss`, `trainable_params` |
| Validation total/ by anatomy | `accuracy`, `precision`, `recall`, `f1`, `kappa`, `roc_auc`,`pr_auc`,`composite`, `loss`                |
| Test image-level             | `test_study_kappa`, `test_study_accuracy` |

Финальные test-метрики:

```text
image_level_accuracy = 0.841
image_level_f1 = 0.824
image_level_kappa = 0.680
image_level_roc_auc = 0.903
image_level_pr_auc = 0.912

study_level_accuracy = 0.853
study_level_f1 = 0.827
study_level_kappa = 0.700
study_level_roc_auc = 0.911
study_level_pr_auc = 0.911
```

### Артефакты

В MLflow сохраняются артефакты финального эксперимента.

Ключевые артефакты:

| Artifact                                                                                            | Назначение |
|-----------------------------------------------------------------------------------------------------|---|
| [`models`](https://drive.google.com/drive/folders/1ZaawPxDR944TRLvoj2LI5V-VgdaFzwSy?usp=drive_link) | финальная модель |
| `confusion_matrix_test_image.png`, `confusion_matrix_test_study.png`                                | confusion matrix |
| `learning_curves.png`                                                                               | графики обучения |
| `all_test_errors.csv`                                                                               | полный список неверных image-level предсказаний |
| `error_counts_by_anatomy.csv`                                                                       | ошибки по анатомии |
| `error_examples_10_20.csv`                                                                          | 20 выбранных уверенных ошибок |
| `error_examples_grid.png`                                                                           | визуальная сетка ошибочных предсказаний |
| `robustness_report.csv`                                                                             | агрегированный robustness report |
| `robustness_predictions.csv`                                                                        | детальные вероятности robustness-теста |

## Воспроизводимость

Для воспроизводимости зафиксированы:

- seed;
- параметры preprocessing;
- параметры split;
- источник данных;
- threshold;
- параметры модели;
- параметры обучения.

## Анализ ошибок

Для финальной DINOv2-модели выполнен анализ ошибок.
Подробный анализ ошибок сохранен в отдельном файле [error_analysis.md](https://github.com/Year-project-team-2-Topic-12/Project2025/blob/b83e3835fcabaf4e962fbac7ac4fd799e1909f30/docs/error_analysis.md)

На test image-level:

```text
3197 изображений
509 ошибок
15.9% error rate
```

Распределение ошибок:

```text
335 false negative
174 false positive
```

False negative доминируют: модель чаще пропускает abnormal-класс, чем ошибочно относит normal-класс к abnormal.

Ошибки по анатомии:

| Anatomy | Images | False negative | False positive | Total errors | Error rate |
|---|---:|---:|---:|---:|---:|
| XR_SHOULDER | 563 | 56 | 66 | 122 | 21.7% |
| XR_FINGER | 461 | 61 | 26 | 87 | 18.9% |
| XR_HAND | 460 | 63 | 22 | 85 | 18.5% |
| XR_WRIST | 659 | 69 | 16 | 85 | 12.9% |
| XR_ELBOW | 465 | 41 | 17 | 58 | 12.5% |
| XR_FOREARM | 301 | 35 | 9 | 44 | 14.6% |
| XR_HUMERUS | 288 | 10 | 18 | 28 | 9.7% |

Типичные категории ошибок:

1. False negative на тонких костях.
2. False positive на металлических конструкциях.
3. False positive на гипсе, шинах, маркерах и нестандартной укладке.
4. Ошибки сложных анатомий с сильным наложением структур.
5. Уверенные ошибки, которые не исправляются простой сменой threshold.

По 20 выбранным ошибкам выполнен отдельный разбор. Эти примеры являются самыми уверенными ошибочными предсказаниями, а не случайной выборкой.

Основное ограничение анализа: MURA содержит лейблы на уровне исследований (`study1_positive` / `study1_negative`), а не на каждом изображении. Поэтому ошибка на уровне изображения не всегда означает, что конкретный снимок визуально содержит или не содержит патологию. Это подтверждается тем, что метрики на уровне study получились выше.

## Сравнение с baseline

В качестве baseline выбран лучший классический ML-подход из ранних GridSearchCV-экспериментов:

```text
hog_pca_poly_logreg_all
```

Он использует HOG/PCA-признаки и логистическую регрессию.

Результат baseline:

```text
Accuracy = 0.6572
AUC = 0.7165
Cohen's kappa = 0.3137
Training time = 2h
```

Сравнение с финальной DINOv2-моделью:

| Модель | Тип | Уровень оценки | Accuracy | AUC / ROC-AUC | Cohen's kappa | Комментарий |
|---|---|---|---:|--------------:|--------------:|---|
| `hog_pca_poly_logreg_all` | Classical ML baseline | overall valid | 0.6572 |0.7165 |        0.3137 | лучший ML baseline |
| `DINOv2-large` | Final DL model | image-level test | 0.841 |        0.903  |         0.680 | финальная image-level оценка |
| `DINOv2-large` | Final DL model | study-level test | 0.853 |         0.911 |         0.700 | финальная study-level оценка |

Финальная DINOv2-модель существенно превосходит лучший классический baseline:

```text
kappa: 0.3137 → 0.680 image-level
kappa: 0.3137 → 0.700 study-level
```

Разница объясняется тем, что DINOv2 использует предобученные визуальные представления и лучше извлекает признаки из рентгеновских изображений. Классические ML-подходы на HOG/PCA-признаках теряют часть пространственной и текстурной информации, важной для обнаружения патологий.

## Проверка на устойчивость

Подробный анализ устойчивости описан в [robustness_analysis.md](https://github.com/Year-project-team-2-Topic-12/Project2025/blob/mlflow-leo/docs/robustness_analysis.md)
Для проверки устойчивости использовались простые искажения входных изображений:

- увеличение яркости;
- уменьшение яркости;
- увеличение контраста;
- поворот на `+5°`;
- поворот на `-5°`;
- blur.

Результаты:

| Perturbation | Mean abs delta prob | Max abs delta prob | Flip rate |
|---|---:|---:|---:|
| brightness_up | 0.009 | 0.310 | 0.98% |
| brightness_down | 0.008 | 0.161 | 0.98% |
| contrast_up | 0.010 | 0.327 | 0.59% |
| rotate_pos5 | 0.041 | 0.627 | 3.91% |
| rotate_neg5 | 0.041 | 0.536 | 3.52% |
| blur | 0.033 | 0.745 | 2.34% |

Вывод:

- яркость и контраст почти не меняют решения модели;
- повороты и blur сильнее влияют на вероятность и чаще меняют класс;
- это согласуется с ошибками на нестандартных проекциях, crop/padding и снимках со сложным наложением структур.

## Чистый PRD notebook
[CP_7_03_MURA_DINOv2_PRD_Predict_Colab.ipynb](https://github.com/Year-project-team-2-Topic-12/Project2025/blob/b83e3835fcabaf4e962fbac7ac4fd799e1909f30/notebooks/CP_7_03_MURA_DINOv2_PRD_Predict_Colab.ipynb)

Чистый PRD notebook используется отдельно и не переобучает модель. Его задача — проверить, что финальная модель восстанавливается из MLflow и может делать предсказание в независимой среде.

1. подключение к MLflow Tracking Server;
2. загрузка модели;
3. загрузка тестового изображения MURA;
4. применение предобработки;
5. выполнение предсказания;
6. вывод вероятности abnormal-класса и итогового класса по threshold `0.480`.
