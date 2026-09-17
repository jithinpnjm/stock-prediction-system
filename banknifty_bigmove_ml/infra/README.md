# Nebius infra — setup order

Blocking checkpoint: none of this runs until SSH access to the VM exists.
Once it does, run these in order.

## 1. Bootstrap the VM
```
scp bootstrap_node.sh <user>@<vm-host>:~
ssh <user>@<vm-host>
./bootstrap_node.sh /dev/disk/by-id/<attached-volume-id>   # find via `lsblk`
# if it says "REBOOT required", reboot and re-run the same command
```
Confirms: NVIDIA driver + `nvidia-smi`, Docker + nvidia-container-toolkit
(GPU visible inside a container), `/data/mlops` mounted and persisted via
`/etc/fstab`, firewall allowing only SSH.

## 2. MLflow + Postgres
```
cd banknifty_bigmove_ml/infra
cp .env.example .env        # set a real MLFLOW_DB_PASSWORD
docker compose up -d
docker compose ps           # both services healthy
```
Smoke test from your laptop:
```
ssh -L 5000:localhost:5000 <user>@<vm-host>   # keep open in a separate terminal
# then locally:
MLFLOW_TRACKING_URI=http://localhost:5000 python3 -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
with mlflow.start_run():
    mlflow.log_param('smoke_test', True)
    mlflow.log_metric('ok', 1.0)
print('logged a run — check http://localhost:5000 in your browser')
"
```
Restart the containers afterward and confirm the run is still there (backend
store is Postgres on the persistent volume, not container-local).

## 3. DVC remote
On the VM:
```
mkdir -p /data/mlops/dvc-store
```
From your laptop, inside the repo:
```
cd banknifty_bigmove_ml
dvc init --subdir           # only if not already run
dvc remote add -d nebius ssh://<user>@<vm-host>/data/mlops/dvc-store
dvc remote modify nebius keyfile ~/.ssh/<your-key>
```
Then `dvc add data/processed/<labeled>.parquet && dvc push` to confirm the
round-trip.

## 4. Training image
On the VM (repo cloned or rsynced there):
```
cd banknifty_bigmove_ml
docker build -f infra/Dockerfile.train -t banknifty-train:latest .
docker run --rm --gpus all banknifty-train:latest
# expect: "CUDA available: True" and the L40S device name printed
```

Only after all four smoke tests pass does real Phase 3/4 training work move
onto the VM.
