#!/usr/bin/env bash
# bootstrap_node.sh
# ------------------
# One-time setup for the Nebius L40S VM: NVIDIA driver, Docker,
# nvidia-container-toolkit, and the persistent data volume mount.
# Run as a user with sudo, once, right after first SSH login.
#
# Usage: ./bootstrap_node.sh /dev/disk/by-id/<your-attached-volume-id>
#
# Idempotent-ish: safe to re-run, each step checks before acting.
set -euo pipefail

DATA_DEVICE="${1:-}"
DATA_MOUNT="/data/mlops"

echo "== 1/5: system update =="
sudo apt-get update -y
sudo apt-get upgrade -y

echo "== 2/5: NVIDIA driver =="
if ! command -v nvidia-smi &>/dev/null; then
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    echo "!! Driver installed — REBOOT required before nvidia-smi works. Re-run this script after reboot to continue."
    exit 0
else
    echo "nvidia-smi already present:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

echo "== 3/5: Docker + nvidia-container-toolkit =="
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
fi

if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

echo "Verifying GPU visible inside a container..."
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

echo "== 4/5: persistent data volume =="
if [ -n "$DATA_DEVICE" ]; then
    if ! blkid "$DATA_DEVICE" &>/dev/null; then
        echo "Formatting $DATA_DEVICE as ext4 (only runs if it has no existing filesystem)..."
        sudo mkfs.ext4 "$DATA_DEVICE"
    fi
    sudo mkdir -p "$DATA_MOUNT"
    if ! mountpoint -q "$DATA_MOUNT"; then
        sudo mount "$DATA_DEVICE" "$DATA_MOUNT"
    fi
    if ! grep -q "$DATA_MOUNT" /etc/fstab; then
        UUID=$(sudo blkid -s UUID -o value "$DATA_DEVICE")
        echo "UUID=$UUID $DATA_MOUNT ext4 defaults 0 2" | sudo tee -a /etc/fstab
    fi
    sudo mkdir -p "$DATA_MOUNT"/{postgres,mlflow-artifacts,dvc-store}
    sudo chown -R "$USER":"$USER" "$DATA_MOUNT"
    echo "Data volume mounted at $DATA_MOUNT"
else
    echo "!! No data device passed — skipping volume mount. Pass the attached disk's /dev path as arg 1."
fi

echo "== 5/5: firewall =="
sudo ufw allow OpenSSH || true
sudo ufw --force enable || true
echo "Only SSH is open. MLflow UI (port 5000) stays private — reach it via SSH tunnel:"
echo "  ssh -L 5000:localhost:5000 <user>@<this-host>"

echo ""
echo "Bootstrap done. Next: cd into infra/ and run 'docker compose up -d' to start MLflow+Postgres."
