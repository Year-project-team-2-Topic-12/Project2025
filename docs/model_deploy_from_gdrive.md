# Деплой модели: Google Drive → локальный MLflow → бэкенд

> 🧭 [README](../README.md) · [Чекпойнт 7 — MLflow](../MLflow_Checkpoint_MURA.md) · **Деплой модели с Google Drive**

Гайд описывает, как забрать обученную модель с Google Drive, зарегистрировать её
в локальном MLflow и переключить бэкенд на эту версию.

Модель хранится как **папка MLflow** (`MLmodel`, `data/`, `conda.yaml`, ...),
а не как один файл.

## Порядок инициализации

1. Скачать модель с Google Drive.
2. Поднять MLflow infra (Postgres + MinIO + MLflow server).
3. Зарегистрировать модель в MLflow.
4. Прописать версию в `.env`.
5. (Пере)запустить бэкенд.

Важно: сначала инфра, потом регистрация — регистрировать некуда, пока MLflow не поднят.

## Шаг 1. Скачать модель с Google Drive

Модель на диске лежит либо как zip, либо как папка.

Вариант A — zip-архив (ссылка на файл `https://drive.google.com/file/d/<FILE_ID>/view`):

```bash
uvx gdown "https://drive.google.com/uc?id=<FILE_ID>" -O model.zip
unzip model.zip -d models/from_gdrive
```

Вариант B — папка (ссылка на folder):

```bash
uvx gdown --folder "https://drive.google.com/drive/folders/<FOLDER_ID>" -O models/from_gdrive
```

Убедиться, что внутри есть файл `MLmodel`, и запомнить путь к папке, где он лежит:

```bash
find models/from_gdrive -name MLmodel
# .../models/from_gdrive/<...>/MLmodel  -> нужен путь к этой папке (без /MLmodel)
```

Именно путь к папке с `MLmodel` пойдёт в шаг регистрации.

> Если ссылка «Anyone with the link» не отдаётся через gdown (большой файл / квота),
> скачать вручную через браузер и распаковать в `models/from_gdrive`.

## Шаг 2. Поднять локальный MLflow

```bash
./bootstrap.sh
```

В TUI-меню выбрать **Start MLflow infra (docker compose up -d)**.
Без меню:

```bash
cd infra/mlflow && docker compose up -d
```

После запуска доступны:

- MLflow UI — http://localhost:5050
- MinIO Console — http://localhost:9901 (S3 API — :9900)

Проверка контейнеров:

```bash
cd infra/mlflow && docker compose ps
```

## Шаг 3. Зарегистрировать модель в MLflow

В меню выбрать **Register model file in MLflow (env name+alias)** и указать путь
к папке с `MLmodel`. Без меню — задать путь через переменную окружения:

```bash
MODEL_UPLOAD_PATH="models/from_gdrive/<...>" ./bootstrap.sh
# затем выбрать пункт "Register model file in MLflow"
```

Что делает шаг:

- подхватывает S3-креды MinIO из `infra/mlflow/.env`;
- копирует артефакты в MinIO без переупаковки (обход бага cloudpickle на Python 3.13);
- создаёт следующую версию модели (инкремент автоматический);
- переносит алиас `prd` на новую версию.

В конце печатается строка вида:
`Registered model: mura_dinov2_transformer version: 3 alias: prd`.

Имя реестра и алиас берутся из `.env` (`MLFLOW_MODEL_NAME`, `MLFLOW_MODEL_ALIAS`).

## Шаг 4. Указать бэкенду версию

Бэкенд резолвит модель сначала по номеру версии. В `.env` выбрать один вариант:

- по номеру:

  ```dotenv
  MLFLOW_MODEL_VERSION='3'
  ```

- по алиасу (шаг уже передвинул `prd` на новую версию):

  ```dotenv
  MLFLOW_MODEL_VERSION=''
  ```

## Шаг 5. Запустить бэкенд

В меню — **Start backend**. Предиктор кэшируется в процессе, поэтому если бэкенд
уже был запущен, его нужно перезапустить.

Проверка в логах (меню **View logs** → backend):

```text
Loaded MURA DINO model models:/mura_dinov2_transformer/3 ...
```

## Быстрый чек-лист

| # | Действие | Пункт меню |
|---|---|---|
| 1 | Скачать с Google Drive → папка с `MLmodel` | `uvx gdown ...` |
| 2 | Поднять MLflow | Start MLflow infra |
| 3 | Зарегистрировать модель | Register model file in MLflow |
| 4 | Прописать `MLFLOW_MODEL_VERSION` в `.env` | — |
| 5 | (Пере)запустить бэкенд | Start backend |

## Траблшутинг

- `NoCredentialsError` — MLflow infra не поднята или порт S3 не 9900.
  Проверить Шаг 2 и `S3_API_PORT` в `infra/mlflow/.env`.
- `IndexError: tuple index out of range` при регистрации — подсунут «сырой» объект
  вместо папки MLflow. Указать путь именно к папке с `MLmodel`.
- Бэкенд отдаёт старые предсказания — процесс не перезапущен (кэш) или
  `MLFLOW_MODEL_VERSION` указывает на старую версию.
- Модель грузится на CPU / медленно — `MURA_DINO_DEVICE='cpu'` в `.env`;
  для GPU поставить `cuda`.
