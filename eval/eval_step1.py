# CLIP score 측정코드 END-To-END 로 만들기 (in shell script)
# step 1. img token 생성 (as a text)
# step 2. text 를 image 로 decode 
# step 3. image 를 저장한 폴더와 GT 폴더를 비교하여 클립스코어 계산


from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import argparse
import torch, time, json, os, sys, re
from pathlib import Path
from tqdm import tqdm

import warnings

import warnings
warnings.filterwarnings("ignore")

sys.path.append("/home/yjoh/project/")
from mDPO.seed_llama.SEED.models.seed_llama_tokenizer import SeedLlamaTokenizer 
# from MAGVLT2.MultiModalLLM.calculate_clip_score import calculate_clip_s_for_folder


BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
image_id_shift = 32000


def process_img_token(model_output: str) -> str:
    """
    model_output example:
    'Here is an image.<img_00247><img_00680><img_03121><img_04030><img_02157><img_05950><img_03121><img_02751><img_02751><img_02157><img_02157><img_07854><img_07773><img_03121><img_03374><img_07434><img_02157><img_07773><img_04824><img_04030><img_03121><img_04030><img_04030><img_02315><img_04030><img_01335><img_06209><img_02751><img_07773><img_02751><img_04030><img_07208>'
    """

    # Step 1: Find all tokens in the format <img_XXXX>
    img_tokens = re.findall(r'<img_(\d+)>', model_output)
    
    # Filter tokens to ensure the number is between 0 and 8191
    filtered_tokens = [f'<img_{num}>' for num in img_tokens if 0 <= int(num) <= 8191]
    
    # if len(filtered_tokens) != 32:
    #     print(f"Error! Image Token number must be 32. But it has {len(filtered_tokens)}.")
    #     return None

    # If no valid tokens remain after filtering, return None
    if not filtered_tokens:
        print(f"Error! Invalid range of tokens found.")
        return None

    # Step 2: Add BOI_TOKEN and EOI_TOKEN
    BOI_TOKEN = '<img>'
    EOI_TOKEN = '</img>'
    
    # Step 3: Build the final sequence
    final_sequence = BOI_TOKEN + ''.join(filtered_tokens) + EOI_TOKEN
    return final_sequence


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/home/yjoh/project/mDPO/seed_llama/SEED/pretrained/seed_llama_8b_sft') 
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1) 
    parser.add_argument('--output_dir', type=str, default='/data/visual_llama_eval/results/spin/seed_llama_8b_sft_official') 
    # where to save generation output

    # FIX Part
    parser.add_argument('--input_dir', type=str, default='data_seed_llama/reformatted') 
    parser.add_argument('--split', type=str, default='test') # 'eval' 아님 주의
    parser.add_argument('--cache_dir', type=str, default='/data/checkpoints/hf_cache_yj') 
    return parser.parse_args()
    


# This code is implemented from https://github.com/AILab-CVC/SEED/blob/main/scripts/seed_llama_inference_8B.py 
def decode_image_text(generate_ids, tokenizer, save_path=None) -> bool:

    boi_list = torch.where(generate_ids == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    eoi_list = torch.where(generate_ids == tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]

    try:
        if len(boi_list) == 0 and len(eoi_list) == 0: 
            text_ids = generate_ids
            texts = tokenizer.decode(text_ids, skip_special_tokens=True)
            print(texts)

        else:
            boi_index = boi_list[0]
            eoi_index = eoi_list[0]

            text_ids = generate_ids[:boi_index]
            if len(text_ids) != 0:
                texts = tokenizer.decode(text_ids, skip_special_tokens=True)
                print(texts)
                
            image_ids = (generate_ids[boi_index+1:eoi_index] - image_id_shift).reshape(1,-1)

            images = tokenizer.decode_image(image_ids)

            images[0].save(save_path)


    except Exception as e:
        print(e)



def create_prompt(content, instruction):
    system_message = ""
    return system_message + "USER: " + content + " " + instruction + "\nASSISTANT: "


def main():

    """ Step 1. Basic setting """
    args = parse_arguments()
    model_path = args.model
    data_frac = args.data_frac
    world_size = args.world_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the cache directory for Hugging Face models
    os.environ['HF_HOME'] = args.cache_dir  

    llm = LLM(
        model=model_path,
        tensor_parallel_size=world_size,
        tokenizer="AILab-CVC/seed-tokenizer-2", # tokenizer, # add (use custom tokenizer)
        trust_remote_code=True,
    )

    # tokenizer = SeedLlamaTokenizer.from_pretrained(
    #                     pretrained_model_name_or_path='AILab-CVC/seed-tokenizer-2',
    #                     vit_precision='fp16',
    #                     diffusion_precision='fp16',
    #                     load_diffusion=True,  
    #                     device='cuda', # modify
    #                     encoder_url='https://huggingface.co/AILab-CVC/seed-tokenizer-2/resolve/main/seed_quantizer.pt',
    #                     diffusion_path='stabilityai/stable-diffusion-2-1-unclip',
    #                     ) 
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # 인자명 참고: https://docs.vllm.ai/en/v0.6.0/dev/sampling_params.html
    sampling_params = SamplingParams(temperature=0.0, best_of=5, use_beam_search=True, max_tokens=120) 
    # sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256) 


    """ Step 2. Loading Data """
    data = load_dataset(args.input_dir, split=args.split)
    data = data.shuffle(seed=42)

    if args.frac_len > 0:
        sub_len = args.frac_len 
        if sub_len*(data_frac+1) > len(data):
            data = data[sub_len*data_frac:]['eval'] # modify (reformat_eval.py 참고)
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]['eval'] 
    else:
        data = data[:]['eval']


    """ Step 3. Creating Original-Prompts """
    instruction = 'Please generate an image.' # 'Please generate an image.'

    idxs_all = [data[idx][0]['content'] for idx in range(len(data))] 
    prompts_all = [create_prompt(data[idx][1]['content'], instruction) for idx in range(len(data))]
    corrects_all = [data[idx][2]['content'] for idx in range(len(data))] 


    """ Step 4. Generation """
    # run vllm 
    results_gathered = list(map(lambda x: x.outputs[0].text, 
                                llm.generate(prompts_all, sampling_params)))
    results = [r.replace("Here is an image.","").strip() for r in results_gathered] 

    # results = list(map(lambda x: x.outputs[0].text, 
    #                             llm.generate(prompts_all, sampling_params)))
    # results = [r.replace("</s>","").lstrip() for r in results_gathered] 


    """ Step 5. Collecting Data into single Sample """
    for idx in range(len(corrects_all)): 
        generated_token = process_img_token(results[idx]) 
        if generated_token is None:
            continue
        
        d = {"data": "DenseFusion", "idx": idxs_all[idx], "generated": generated_token}
        
        filename = f"{args.output_dir}/eval_{data_frac}.jsonl" # modify
        with open(filename, 'a') as f:
            json.dump(d, f)
            f.write('\n')
        

    # DO NOT EXECUTE HERE (HERE IS IN FRACTION)

    # """ Step 6. Calcuate Metrics """
    # print('\n\nStart calculating clip score...')
    # clip_score = calculate_clip_s_for_folder(args.gt_dir, args.output_dir) # DIR must have same file name


    # print(f"\n\nEvaluation Finished. Generated images are saved at {args.output_dir}")  
    # print("======================================")
    # print(f"# Total: {len(corrects_all)}")
    # print(f"# Success (Generated): {len(corrects_all) - fail}")
    # print(f"# Fail: {fail}")
    # print("======================================")
    # print(f"CLIP score: {clip_score:.02f}") 



if __name__ == "__main__":
    main()
