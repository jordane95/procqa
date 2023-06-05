export CUDA_VISIBLE_DEVICES=3

MODEL_DIR="tmp/debug_c_bert"
EMBEDDINGS_DIR="tmp/embeddings_c_bert"


python run_biencoder.py \
    --model_name_or_path bert-base-uncased \
    --output_dir $MODEL_DIR \
    --data_file ../data/qa.en.c.json \
    --train_group_size 1 \
    --do_train \
    --per_device_train_batch_size 16 \
    --query_max_len 128 \
    --passage_max_len 128 \
    --num_train_epochs 3 \
    --do_eval \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --overwrite_output_dir \
    --logging_steps 10
