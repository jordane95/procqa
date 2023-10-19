# export CUDA_VISIBLE_DEVICES=2

pl=$1 # name of programming language

MODEL_DIR="tmp/bi_bm25neg_codebert_${pl}"
EMBEDDINGS_DIR="tmp/bi_bm25neg_codebert_embeddings_${pl}"
RESULT_DIR="tmp/bi_bm25neg_codebert_result_${pl}"

mkdir -p $RESULT_DIR

python run_biencoder.py \
    --model_name_or_path ../models/codebert-base \
    --output_dir $MODEL_DIR \
    --data_file data/qa.en.${pl}.bm25.list.json \
    --train_group_size 2 \
    --do_train \
    --per_device_train_batch_size 16 \
    --num_train_epochs 3 \
    --do_eval \
    --per_device_eval_batch_size 64 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --logging_steps 100 \
    --fp16


python run_biencoder.py \
    --model_name_or_path $MODEL_DIR \
    --output_dir $MODEL_DIR \
    --data_file ../pls/qa.en.${pl}.json \
    --do_predict \
    --prediction_save_path $EMBEDDINGS_DIR \
    --encode_corpus \
    --encode_query \
    --overwrite_output_dir \
    --fp16


python test.py \
    --query_reps_path $EMBEDDINGS_DIR/query_reps \
    --passage_reps_path $EMBEDDINGS_DIR/passage_reps \
    --qrels_file $EMBEDDINGS_DIR/qrels.test.tsv \
    --ranking_file  $EMBEDDINGS_DIR/test_ranking.txt \
    > ${RESULT_DIR}/result.log 2>&1

