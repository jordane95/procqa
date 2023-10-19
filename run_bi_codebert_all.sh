
pls=(lisp go c++ c# ruby php java python)


gpu_ids=($(seq 0 2))

 
n_cards=${#gpu_ids[@]} 
echo "Using ${n_cards} gpu cards..." 

 
declare -A gpuid2pids

 
for gpu_id in ${gpu_ids[@]}; do 
    echo $gpu_id 
    gpuid2pids[$gpu_id]=0
done 

 
idx=0
for pl in ${pls[@]}; do 
    echo "" 
    echo "dataset id ${idx}" 
    gpu_idx=$(($idx % $n_cards)) 
    gpu_id=${gpu_ids[$gpu_idx]} 
    echo "GPU id ${gpu_id}" 
    echo "${pl} will run on card ${gpu_id}" 
    pid=${gpuid2pids[$gpu_id]} 
    echo "Previous job pid on ${gpu_id} is ${pid}, wating it to be done" 
    wait $pid 
    CUDA_VISIBLE_DEVICES=$gpu_id nohup bash run_biencoder_codebert_pl.sh $pl > $pl.log 2>&1 &
    # python -c "import time; time.sleep(10); print('${dataset}')" > debug/${dataset}.log & 
    pid=$! 
    gpuid2pids[$gpu_id]=$pid 
    echo "New pid on gpu ${gpu_id} is ${gpuid2pids[$gpu_id]}" 
    ((idx++)) 
done 
