export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=disabled

deepspeed --master_port 29505 qwenvl_run.py ./qwen_vl/args/qwen.yaml \
  --deepspeed \
  --deepspeed_config ds_config.json \
  --collect_grad False \
  --use_data_flag full \
  --progressive False \
  --ratio 0.2 \
  --use_tokensr True \
  --pattern 32_patch \
  --model_version v_2
