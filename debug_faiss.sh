export CUDA_VISIBLE_DEVICES=0

EMBEDDINGS_DIR="tmp/embeddings"


python test.py \
    --query_reps_path $EMBEDDINGS_DIR/query_reps \
    --passage_reps_path $EMBEDDINGS_DIR/passage_reps \
    --qrels_file $EMBEDDINGS_DIR/qrels.test.tsv \
    --ranking_file  $EMBEDDINGS_DIR/test_ranking.txt 
