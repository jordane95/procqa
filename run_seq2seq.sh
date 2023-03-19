export CUDA_VISIBLE_DEIVCES=1

python seq2seq/run_seq2seq_qa.py \
  --model_name_or_path t5-small \
  --train_file data/qa.c.clean.json \
  --question_column title \
  --context_column question \
  --answer_column answer \
  --do_train \
  --logging_steps 100 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-3 \
  --num_train_epochs 30 \
  --max_seq_length 256 \
  --output_dir tmp/debug_seq2seq_squad/ \
  --overwrite_output_dir \
  --do_eval \
  --per_device_eval_batch_size 16 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --predict_with_generate \
  --do_predict
