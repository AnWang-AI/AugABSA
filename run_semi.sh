# run by command like: bash ./run_semi.sh device_num seed

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=$1

tasks=('asqp')
datasets=('rest15')

for task in ${tasks[*]}
do
  for dataset in ${datasets[*]}
  do
    python scripts/text2data/main.py \
      --task ${task} \
      --dataset ${dataset} \
      --model_name_or_path t5-base \
      --back_model_name_or_path t5-base \
      --seed $2 \
      --train_batch_size 8 \
      --gradient_accumulation_steps 1 \
      --eval_batch_size 32 \
      --learning_rate 3e-4 \
      --num_train_epochs 20 \
      --do_train \
      --forward \
      --template_version v3 \
      --debug \
#      --aug_load_file save/pesudo_parallel_data_ere_SCIERC_256000_filtered \
#      --aug \
#      --debug \

  done
done








