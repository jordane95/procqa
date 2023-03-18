export CUDA_VISIBLE_DEIVCES=1

python seq2seq/run_seq2seq_qa.py \
  --model_name_or_path t5-small \
  --train_file data/qa.c.clean.json \
  --context_column title \
  --question_column question \
  --answer_column answer \
  --do_train \
  --logging_steps 10 \
  --per_device_train_batch_size 12 \
  --learning_rate 1e-3 \
  --num_train_epochs 100 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir tmp/debug_seq2seq_squad/ \
  --overwrite_output_dir
