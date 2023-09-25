seed=(2023 111 222 333 444 555 666 777 888 999)
for round in 0;
do
  for class_order in 0 1 2 3 4;
  do
    for ft_task in $(seq 1 4);
      do
        CUDA_VISIBLE_DEVICES=2 python eval.py \
        --task ${ft_task} \
        --idrandom 0 \
        --baseline 'deit_C10_5T' \
        --seed ${seed[$round]} \
        --batch_size 64 \
        --sequence_file 'C10_5T' \
        --learning_rate 0.005 \
        --num_train_epochs 20 \
        --base_dir ./data \
        --class_order ${class_order} 
      done
  done
done