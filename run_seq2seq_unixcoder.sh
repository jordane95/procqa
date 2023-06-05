export CUDA_VISIBLE_DEVICES=2

python seq2seq/run_seq2seq_qa.py \
  --model_name_or_path ../models/unixcoder-base \
  --train_file ../pls/qa.en.c.json \
  --question_column title \
  --context_column question \
  --answer_column answer \
  --do_train True \
  --logging_steps 100 \
  --save_steps 1000 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 256 \
  --max_answer_length 256 \
  --output_dir tmp/seq2seq_unixcoder_c/ \
  --overwrite_output_dir \
  --do_eval \
  --per_device_eval_batch_size 64 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --predict_with_generate \
  --do_predict
