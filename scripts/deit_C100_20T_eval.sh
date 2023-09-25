seed=(2023 111 222 333 444 555 666 777 888 999)



for round in 0;
do
  for idrandom in 0 1 2 3 4;
  do
    for ft_task in $(seq 1 19);
      do
        CUDA_VISIBLE_DEVICES=6 python eval.py \
        --task ${ft_task} \
        --idrandom 0 \
        --class_order ${idrandom} \
        --baseline 'deit_C100_20T' \
        --seed ${seed[$round]} \
        --batch_size 64 \
        --sequence_file 'C100_20T' \
        --learning_rate 0.005 \
        --latent 128 \
        --replay_buffer_size 2000 \
        --num_train_epochs 40 \
        --base_dir ./data
      done
  done
done