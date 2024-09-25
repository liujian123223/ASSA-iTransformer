export CUDA_VISIBLE_DEVICES=0

# 设置参数可选值
nums=(1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5)
intervals=(1 2 4 6 1 2 4 6 1 2 4 6 1 2 4 6 1 2 4 6)
d_model_values=(512 2048 1024 2048 1024 1024 2048 2048 1024 1024 2048 2048 1024 1024 2048 2048 512 1024 2048 2048)
d_ff_values=(1024 1024 1024 2048 1024 1024 2048 256 1024 1024 2048 512 256 512 2048 2048 1024 1024 512 1024)
e_layers_values=(3 3 4 3 3 3 3 3 6 5 3 4 6 3 3 3 3 3 3 3 )
learning_rate_values=(0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)
batch_size_values=(16 16 8 8 16 8 8 8 16 8 16 8 64 16 16 16 16 16 16 16)
subsequence_num_values=(15 19 12 18 9 5 12 12 29 25 20 20 21 17 16 17 19 19 17 16)

# 遍历参数组合
for ((i=0; i<${#nums[@]}; i++)) ;do
    num=${nums[i]}
    interval=${intervals[i]}
    e_layers=${e_layers_values[i]}
    d_model=${d_model_values[i]}
    d_ff=${d_ff_values[i]}
    learning_rate=${learning_rate_values[i]}
    batch_size=${batch_size_values[i]}
    subsequence_num=${subsequence_num_values[i]}
      for decomposition_method in "ssa" "vmd" "swt" "ewt" "emd" ; do
      # 运行代码
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/UK-DALE/ \
        --model iTransformer \
        --data UK-DALE \
        --decomposition_method ${decomposition_method} \
        --model_id house${num}_it${interval}_${decomposition_method} \
        --data_path house${num}_5min_KWh.csv \
        --interval ${interval} \
        --e_layers ${e_layers} \
        --d_model ${d_model} \
        --d_ff ${d_ff} \
        --learning_rate ${learning_rate} \
        --batch_size ${batch_size} \
        --subsequence_num ${subsequence_num} \
        --features M \
        --seq_len 8 \
        --label_len 1 \
        --pred_len 1 \
        --enc_in 4 \
        --dec_in 4 \
        --c_out 4 \
        --des 'Exp' \
        --itr 1
    done
done