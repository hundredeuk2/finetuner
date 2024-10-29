export WANDB_ENTITY="hundredeuk2"
export WANDB_PROJECT="judge"

# 그 후에 원래 명령어 실행
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 \
torchrun --nproc_per_node=4 \
./train.py \
--config ./configs/base.yaml