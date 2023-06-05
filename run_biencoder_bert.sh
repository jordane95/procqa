export CUDA_VISIBLE_DEVICES=3

MODEL_DIR="tmp/bi_c_bert"
EMBEDDINGS_DIR="tmp/embeddings_c_bert"


python run_biencoder.py \
    --model_name_or_path bert-base-uncased \
    --output_dir $MODEL_DIR \
    --data_file ../data/qa.en.c.json \
    --train_group_size 1 \
    --do_train \
    --per_device_train_batch_size 16 \
    --num_train_epochs 3 \
    --do_eval \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --overwrite_output_dir \
    --logging_steps 100


python run_biencoder.py \
    --model_name_or_path $MODEL_DIR \
    --output_dir $MODEL_DIR \
    --data_file ../data/qa.en.c.json \
    --do_predict \
    --prediction_save_path $EMBEDDINGS_DIR \
    --encode_corpus \
    --encode_query \
    --overwrite_output_dir

python test.py \
    --query_reps_path $EMBEDDINGS_DIR/query_reps \
    --passage_reps_path $EMBEDDINGS_DIR/passage_reps \
    --qrels_file $EMBEDDINGS_DIR/qrels.test.tsv \
    --ranking_file  $EMBEDDINGS_DIR/test_ranking.txt
