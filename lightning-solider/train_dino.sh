#!/bin/bash
#SBATCH --job-name=DINO-SOLIDER
#SBATCH --partition=hopper
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --mem=2000G
#SBATCH --comment="dataset_generation"
#SBATCH --output=model_%A.log

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date
hostname -I;

# SLURM 환경 변수 출력 (디버깅용)
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# NCCL 환경 변수 설정 (네트워크 연결 문제 해결)
# cubox 서버는 InfiniBand가 없고 IPv6 링크-로컬 주소로 연결 시도 시 실패하므로 설정 필요

# 1. InfiniBand 비활성화 (Ethernet만 사용) - 가장 중요!
export NCCL_IB_DISABLE=1

# 2. 네트워크 인터페이스 지정 (IPv4 주소를 가진 인터페이스 사용)
# hostname -I의 첫 번째 IP 주소를 가진 인터페이스 찾기
PRIMARY_IP=$(hostname -I | awk '{print $1}')
if [ -n "$PRIMARY_IP" ]; then
    # ip 명령어로 해당 IP를 가진 인터페이스 찾기
    IFACE=$(ip -4 addr show | grep -B 2 "$PRIMARY_IP" | grep -oP '^\d+:\s\K[^:]+' | head -1)
    if [ -z "$IFACE" ]; then
        # ifconfig 사용 (ip 명령어가 없는 경우)
        IFACE=$(ifconfig | grep -B 1 "$PRIMARY_IP" | grep -oP '^\S+' | head -1 | tr -d ':')
    fi
    if [ -n "$IFACE" ]; then
        echo "Using network interface: $IFACE (IP: $PRIMARY_IP)"
        export NCCL_SOCKET_IFNAME=$IFACE
    else
        echo "Warning: Could not detect network interface, using default"
        # 일반적인 인터페이스 이름 시도
        export NCCL_SOCKET_IFNAME=eth0,eth1,ens,eno,enp
    fi
else
    echo "Warning: Could not detect primary IP"
    export NCCL_SOCKET_IFNAME=eth0,eth1,ens,eno,enp
fi

# 3. 디버깅 레벨 설정 (문제 발생 시 INFO로 변경하여 상세 로그 확인 가능)
export NCCL_DEBUG=WARN

# 4. 추가 네트워크 정보 출력 (디버깅용)
echo "All IP addresses:"
hostname -I
echo "Network interface details:"
ip -4 addr show 2>/dev/null | head -20 || ifconfig 2>/dev/null | head -20

# srun을 사용하여 각 노드에서 프로세스 실행
# PyTorch Lightning이 SLURM 환경 변수를 자동으로 감지하여 분산 학습 설정
srun python -W ignore train_dino.py \
--arch swin_base \
--data_path /purestorage/AILAB/AI_2/datasets/PersonReID/solider_surv_pre_v2/images \
--output_dir ./log/solider_base_out_0.04_epoch15_dataset_v2_val-test \
--height 256 --width 128 \
--crop_height 128 --crop_width 64 \
--epochs 50 \
--batch_size_per_gpu 112 \
--num_workers 8 \
--global_crops_scale 0.8 1. \
--local_crops_scale 0.05 0.8 \
--devices 8 \
--num_nodes 4 \
--precision bf16-mixed \
--teacher_temp 0.04