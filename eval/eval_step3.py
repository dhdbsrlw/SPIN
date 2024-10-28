import argparse
import torch, time, json, os, sys
from pathlib import Path
from tqdm import tqdm
import traceback
import warnings

warnings.filterwarnings("ignore")

sys.path.append("/home/yjoh/project/")
from mDPO.seed_llama.SEED.models.seed_llama_tokenizer import SeedLlamaTokenizer 
from MAGVLT2.MultiModalLLM.calculate_clip_score import calculate_clip_s_for_folder


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--org_dir', type=str, default='/data/visual_llama_eval/results/spin/seed_llama_8b_sft_official') # GT
    parser.add_argument('--gen_dir', type=str, default='/data/visual_llama_eval/results/spin/seed_llama_8b_sft_official')
    return parser.parse_args()


def main():

    args = parse_arguments()
    
    print("\n\n *** Start calculating CLIP score ***")

    num_img = len(os.listdir(args.gen_dir))
    clip_score = calculate_clip_s_for_folder(args.org_dir, args.gen_dir)

    print('# Number of original images: 18k-') # TODO
    print(f'# Number of generated images: {num_img}')
    print(f'# CLIP score: {clip_score}')    


if __name__ == "__main__":
    main()
