export CUDA_VISIBLE_DEVICES=0

MODEL_DIR="tmp/debug_biencoder"
EMBEDDINGS_DIR="tmp/embeddings"

python run_biencoder.py \
    --model_name_or_path bert-base-uncased \
    --output_dir $MODEL_DIR \
    --data_file data/qa.c.clean.json \
    --do_train \
    --train_group_size 1 \
    --do_predict \
    --prediction_save_path $EMBEDDINGS_DIR \
    --encode_corpus \
    --encode_query \
    --overwrite_output_dir


python test.py \
    --query_reps_path $EMBEDDINGS_DIR/query_reps \
    --passage_reps_path $EMBEDDINGS_DIR/passage_reps \
    --qrels_file $EMBEDDINGS_DIR/qrels.test.tsv \
    --ranking_file  $EMBEDDINGS_DIR/dev_ranking.txt \
    --use_gpu 
