export CUDA_VISIBLE_DEVICES=4,5

python3 spin/generate_vllm_seed_llama_si.py --frac_len 5000 --data_frac 0 --world_size 2
python3 spin/generate_vllm_seed_llama_si.py --frac_len 5000 --data_frac 1 --world_size 2
python3 spin/generate_vllm_seed_llama_si.py --frac_len 5000 --data_frac 2 --world_size 2
python3 spin/generate_vllm_seed_llama_si.py --frac_len 5000 --data_frac 3 --world_size 2
python3 spin/generate_vllm_seed_llama_si.py --frac_len 5000 --data_frac 4 --world_size 2
python3 spin/generate_vllm_seed_llama_si.py --frac_len 5000 --data_frac 5 --world_size 2
python3 spin/generate_vllm_seed_llama_si.py --frac_len 5000 --data_frac 6 --world_size 2
python3 spin/generate_vllm_seed_llama_si.py --frac_len 5000 --data_frac 7 --world_size 2
python3 spin/generate_vllm_seed_llama_si.py --frac_len 5000 --data_frac 8 --world_size 2
python3 spin/generate_vllm_seed_llama_si.py --frac_len 5000 --data_frac 9 --world_size 2
python3 spin/generate_vllm_seed_llama_si.py --frac_len 5000 --data_frac 10 --world_size 2

# Generate for the test split as well
python3 spin/generate_vllm_seed_llama_si.py --frac_len 5000 --data_frac 0 --world_size 2 --split test