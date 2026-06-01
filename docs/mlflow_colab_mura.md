# MLflow + Colab для MURA DINOv2

## Локальный MLflow stack

1. Создать env-файл:

```bash
cd infra/mlflow
cp .env.example .env
```

2. Поднять Postgres, MinIO и MLflow:

```bash
docker compose up -d
```

3. Проверить локально:

- MLflow UI: `http://localhost:5050`
- MinIO Console: `http://localhost:9001`
- MinIO API: `http://localhost:9000`

MLflow server запущен с `--serve-artifacts`, поэтому Colab должен обращаться только к MLflow Tracking URI. Артефакты будут проксироваться MLflow-сервером в MinIO.

В `docker-compose.yml` для MLflow выставлен увеличенный gunicorn timeout:

```text
--gunicorn-opts "--timeout 1200 --graceful-timeout 1200 --workers 2"
```

Это защищает от падения worker-ов при долгих artifact uploads через медленный tunnel. Для очень больших checkpoint-файлов всё равно надежнее использовать прямой multipart upload в MinIO/S3 и логировать в MLflow JSON-pointer на S3 URI.

## Проброс через VPS

В Colab notebook указывается:

```python
VPS_HOST = "YOUR_VPS_IP"
MLFLOW_TRACKING_URI = f"http://{VPS_HOST}:5050"
```

Минимально для Colab нужен порт `5050`. Порты `9000` и `9001` нужны только если хочешь открывать MinIO API/console через VPS.

Пример reverse tunnel с локальной машины на VPS:

```bash
ssh -N \
  -R 0.0.0.0:5050:localhost:5050 \
  -R 0.0.0.0:9000:localhost:9000 \
  -R 0.0.0.0:9001:localhost:9001 \
  user@YOUR_VPS_IP
```

На VPS в `sshd_config` должен быть разрешен remote bind наружу:

```text
GatewayPorts clientspecified
```

После изменения `sshd_config` перезапусти sshd на VPS.

## Colab notebooks

- `notebooks/MURA_DINOv2_MLflow_Colab.ipynb` - самодостаточный train/eval/log/register notebook.
- `notebooks/MURA_DINOv2_PRD_Predict_Colab.ipynb` - чистый notebook для загрузки PRD-модели из MLflow registry и тестового predict.

В Colab не нужны файлы из репозитория. Нужны только:

- сам `.ipynb`;
- `MURA-v1.1.zip` в Google Drive или `/content`;
- доступный `MLFLOW_TRACKING_URI`.

Raw MURA обрабатывается на лету: notebook сам извлекает архив, строит dataframe, делает split без study leakage и resize/pad в Dataset.

Для скорости и стабильности Colab notebook сначала копирует `MURA-v1.1.zip` из Google Drive в `/content/MURA-v1.1.zip`, а затем распаковывает локальную копию в `/content/mura_dinov2_work/mura_extracted`.
