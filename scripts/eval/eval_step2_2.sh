export CUDA_VISIBLE_DEVICES=4,5,6,7

CUDA_VISIBLE_DEVICES=5 python3 eval/eval_step2.py --frac_len 5000 --data_frac 0  --input_dir /data/visual_llama_eval/results/spin/seed_llama_iter2
CUDA_VISIBLE_DEVICES=5 python3 eval/eval_step2.py --frac_len 5000 --data_frac 1  --input_dir /data/visual_llama_eval/results/spin/seed_llama_iter2
CUDA_VISIBLE_DEVICES=5 python3 eval/eval_step2.py --frac_len 5000 --data_frac 2  --input_dir /data/visual_llama_eval/results/spin/seed_llama_iter2
CUDA_VISIBLE_DEVICES=5 python3 eval/eval_step2.py --frac_len 5000 --data_frac 3  --input_dir /data/visual_llama_eval/results/spin/seed_llama_iter2
