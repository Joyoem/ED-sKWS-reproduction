#!/bin/bash
#SBATCH --job-name=edskws-100
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=18:00:00
#SBATCH --output=/home/zhaoyan/logs/edskws-100-%j.out

set -e

START=$(date +%s)
echo "JOB START: $(date)"

source /home/zhaoyan/miniforge3/etc/profile.d/conda.sh
conda activate edskws

cd /home/zhaoyan/projects/ED-sKWS

python -u run_exp.py \
  --model_type adLIF \
  --dataset_name sc \
  --data_folder /data/zhaoyan/datasets/speech_commands_v0.02 \
  --new_exp_folder /data/zhaoyan/checkpoints/edskws/gsc35_512_ep100 \
  --nb_layers 3 \
  --nb_hiddens 512 \
  --batch_size 128 \
  --nb_epochs 100 \
  --lr 0.01 \
  --pdrop 0.1 \
  --normalization batchnorm \
  --use_bias 0 \
  --bidirectional 0 \
  --use_augm 0 \
  --log_tofile 0

END=$(date +%s)
echo "JOB END: $(date)"
echo "TOTAL SECONDS: $((END-START))"
