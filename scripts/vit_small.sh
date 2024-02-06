seed=(2023 111 222 333 444 555 666 777 888 999)
cuda_id=0
ve="vit_small_patch16_224"
bs=('vit_small_patch16_224_C10_5T_hat' 'vit_small_patch16_224_C100_10T_hat' 'vit_small_patch16_224_C100_20T_hat' 'vit_small_patch16_224_T_5T_hat' 'vit_small_patch16_224_T_10T_hat')
seqfile=('C10_5T' 'C100_10T' 'C100_20T' 'T_5T' 'T_10T')
learning_rate=(0.005 0.001 0.005 0.005 0.005)
num_train_epochs=(20 40 40 15 10)
base_dir="ckpt"
final_task=(4 9 19 4 9)
latent=(64 128 128 128 128)
buffersize=(200 2000 2000 2000 2000)

for round in 0;
do
  for class_order in 0;
  do
    for i in "${!bs[@]}";
    do
        for ft_task in $(seq 0 ${final_task[$i]});
        do
            CUDA_VISIBLE_DEVICES=$cuda_id python main.py \
            --task ${ft_task} \
            --idrandom 0 \
            --visual_encoder $ve \
            --baseline "${bs[$i]}" \
            --seed ${seed[$round]} \
            --batch_size 64 \
            --sequence_file "${seqfile[$i]}" \
            --learning_rate ${learning_rate[$i]} \
            --num_train_epochs ${num_train_epochs[$i]} \
            --base_dir ckpt \
            --class_order ${class_order} \
            --latent  ${latent[$i]} \
            --replay_buffer_size ${buffersize[$i]} \
            --training
        done
        for ft_task in $(seq ${final_task[$i]} ${final_task[$i]});
        do
            CUDA_VISIBLE_DEVICES=$cuda_id python eval.py \
            --task ${ft_task} \
            --idrandom 0 \
            --visual_encoder $ve \
            --baseline "${bs[$i]}" \
            --seed ${seed[$round]} \
            --batch_size 64 \
            --sequence_file "${seqfile[$i]}" \
            --base_dir ckpt \
            --class_order ${class_order} \
            --latent  ${latent[$i]} \
            --replay_buffer_size ${buffersize[$i]}
        done
    done
  done
done