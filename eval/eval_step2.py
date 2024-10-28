# TODO: multi-GPU, multi-processing 으로 변경

# CLIP score 측정코드 END-To-END 로 만들기 (in shell script)
# step 1. img token 생성 (as a text)
# step 2. text 를 image 로 decode 
# step 3. image 를 저장한 폴더와 GT 폴더를 비교하여 클립스코어 계산

import argparse
import torch, time, json, os, sys
from pathlib import Path
from tqdm import tqdm
import traceback

import warnings

import warnings
warnings.filterwarnings("ignore")

sys.path.append("/home/yjoh/project/")
from mDPO.seed_llama.SEED.models.seed_llama_tokenizer import SeedLlamaTokenizer 

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--input_dir', type=str, default='/data/visual_llama_eval/results/spin/seed_llama_8b_sft_official')
    parser.add_argument('--cache_dir', type=str, default='/data/checkpoints/hf_cache_yj') # fix
    return parser.parse_args()


def main():

    """ Step 1. Basic setting """
    args = parse_arguments()
    data_frac = args.data_frac
    output_dir = Path(os.path.join(args.input_dir, "images")) # Decoded images must be saved in sub-folder.
    # output_dir = Path(args.output_dir) 
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the cache directory for Hugging Face models
    os.environ['HF_HOME'] = args.cache_dir  


    tokenizer = SeedLlamaTokenizer.from_pretrained(
                        pretrained_model_name_or_path='AILab-CVC/seed-tokenizer-2',
                        vit_precision='fp16',
                        diffusion_precision='fp16',
                        load_diffusion=True,  
                        device='cuda', # modify
                        encoder_url='https://huggingface.co/AILab-CVC/seed-tokenizer-2/resolve/main/seed_quantizer.pt',
                        diffusion_path='stabilityai/stable-diffusion-2-1-unclip',
                        ) 
    tokenizer.pad_token_id = tokenizer.eos_token_id



    """ Step 2. Loading Generated for Evaluation Data """
    eval_data = []
    # loop through the files in the folder
    for file_name in sorted(os.listdir(args.input_dir)):
        if file_name.startswith("eval_") and file_name.endswith(".jsonl"):
            file_path = os.path.join(args.input_dir, file_name)
            
            # open and read each JSONL file
            with open(file_path, "r") as file:
                for line in file:
                    data = json.loads(line.strip())  # load each line as a JSON object
                    eval_data.append(data)            # append to the list
    if args.frac_len > 0:
        sub_len = args.frac_len 
        if sub_len*(data_frac+1) > len(eval_data):
            eval_data = eval_data[sub_len*data_frac:]
        else:
            eval_data = eval_data[sub_len*data_frac:sub_len*(data_frac+1)]
    else:
        eval_data = eval_data[:]


    """ Step 3. Data conversion """
    for sample in tqdm(eval_data, desc=f"Decoding and saving generated images in {args.data_frac} ..."):

        # Extract all image_token_ids from the "generated" string
        sample["generated"] = (sample["generated"].split("<img>")[1]).split("</img>")[0] # add
        image_token_ids = [int(x) for x in sample["generated"].replace("<img_", "").replace(">", "")]

        # Ensure the list has exactly 32 elements
        # If more than 32 elements, truncate; if less, pad with zeros
        image_token_ids = image_token_ids[:32] + [0] * (32 - len(image_token_ids))
        image_tensor = torch.tensor(image_token_ids, dtype=torch.int).reshape(1,-1)
        image_fname = os.path.join(output_dir, sample["idx"]) 

        """ Step 4. Decode into PIL.Image and save """
        try:
            image_ids = (image_tensor) 
            images = tokenizer.decode_image(image_ids)
            images[0].save(image_fname)
        except Exception as e1:
            print(e1)
            continue


if __name__ == "__main__":
    main()
