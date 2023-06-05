export CUDA_VISIBLE_DEVICES=1

python seq2seq/run_seq2seq_qa.py \
  --model_name_or_path tmp/debug_seq2seq_procqa/ \
  --train_file data/qa.c.clean.json \
  --question_column title \
  --context_column question \
  --answer_column answer \
  --do_train False \
  --logging_steps 100 \
  --save_steps 1000 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-3 \
  --num_train_epochs 100 \
  --max_seq_length 256 \
  --max_answer_length 256 \
  --output_dir tmp/debug_seq2seq_procqa/ \
  --overwrite_output_dir \
  --do_eval \
  --per_device_eval_batch_size 64 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --predict_with_generate \
  --do_predict
