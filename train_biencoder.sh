export CUDA_VISIBLE_DEVICES=0

python run_biencoder.py \
    --model_name_or_path bert-base-uncased \
    --output_dir tmp \
    --data_file ../data/qa.c.json \
    --do_train
