export CUDA_VISIBLE_DEIVCES=0

python seq2seq/run_seq2seq_qa.py \
  --model_name_or_path t5-small \
  --train_file data/qa.en.c.bm25.json \
  --context_column url \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir tmp/debug_seq2seq_squad/
