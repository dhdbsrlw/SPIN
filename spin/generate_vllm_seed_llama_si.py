# Self-Improved Data for T2I generation ver. (SPIN 2)

# Use generate_vllm.sh
# 'iter' 가 바뀜에 따라 TODO 라인 수정

# Dense-Caption 80K Dataset (T2I)
import sys
import re
sys.path.append("/home/yjoh/project/")
from mDPO.seed_llama.SEED.models.seed_llama_tokenizer import SeedLlamaTokenizer 
from mDPO.seed_llama.SEED.MultiModalLLM.src.model.llama_xformer import LlamaForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

# import pyrootutils
# pyrootutils.setup_root(__file__, indicator=".spin-root", pythonpath=True, cwd=True)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta

import warnings

from accelerate.utils import InitProcessGroupKwargs

import warnings
warnings.filterwarnings("ignore")
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/home/yjoh/project/mDPO/seed_llama/SEED/pretrained/seed_llama_8b_sft') # TODO
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='data_seed_llama/generated_imp_1/iter0') # TODO
    parser.add_argument('--world_size', type=int, default=1) 
    parser.add_argument('--input_dir', type=str, default='data_seed_llama/reformatted') # do not need to care
    parser.add_argument('--split', type=str, default='train') 
    parser.add_argument('--cache_dir', type=str, default='/data/checkpoints/hf_cache_yj') 
    return parser.parse_args()

def process_img_token(model_output: str) -> str:
    """
    model_output example:
    'Here is an image.<img_00247><img_00680><img_03121><img_04030><img_02157><img_05950><img_03121><img_02751><img_02751><img_02157><img_02157><img_07854><img_07773><img_03121><img_03374><img_07434><img_02157><img_07773><img_04824><img_04030><img_03121><img_04030><img_04030><img_02315><img_04030><img_01335><img_06209><img_02751><img_07773><img_02751><img_04030><img_07208>'
    """

    # Step 1: Find all tokens in the format <img_XXXX>
    img_tokens = re.findall(r'<img_(\d+)>', model_output)
    
    # Filter tokens to ensure the number is between 0 and 8191
    # filtered_tokens = [f'<img_{num}>' for num in img_tokens if 0 <= int(num) <= 8191]
    img_tokens = map(int, img_tokens)  # convert to integers upfront
    filtered_tokens = [f'<img_{num}>' for num in img_tokens if 0 <= num <= 8191]
    
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
        # enable_lora=True,
        # max_lora_rank=64,
        # dtype=torch.bfloat16,
        # lora_config=seed_llama_lora_config,
    )

    # 인자명 참고: https://docs.vllm.ai/en/v0.6.0/dev/sampling_params.html
    # sampling_params = SamplingParams(temperature=0.8, num_beams=5, max_tokens=120) 
    sampling_params = SamplingParams(temperature=0.0, best_of=5, use_beam_search=True, max_tokens=120) 
    # sampling_params = SamplingParams(temperature=1.0, top_p=0.5, use_beam_search=False, max_tokens=120) 
    # sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256) 


    """ Step 2. Loading Data """
    data = load_dataset(args.input_dir, split=args.split)
    data = data.shuffle(seed=42)
    if args.frac_len > 0:
        sub_len = args.frac_len 
        if sub_len*(data_frac+1) > len(data):
            data = data[sub_len*data_frac:]['real']
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]['real']
    else:
        data = data[:]['real']



    """ Step 3. Creating Original-Prompts """
    instruction = 'Please generate an image.'

    prompts_all = [create_prompt(data[idx][0]['content'], instruction) for idx in range(len(data))]
    # prompts_all = ["### Instruction: " + data[idx][0]['content'] + "\n\n### Response: " for idx in range(len(data))]
    prompts_old = [data[idx][0]['content'] + " " + instruction for idx in range(len(data))]
    corrects_all = [data[idx][1]['content'] for idx in range(len(data))] # TODO: 수정



    """ Step 4. 1st Generation """
    start=time.time()

    # run vllm (1차 생성)
    results_gathered_1 = list(map(lambda x: x.outputs[0].text, 
                                llm.generate(prompts_all, sampling_params)))
    # results = [r.replace("</s>","").lstrip() for r in results_gathered_1] 
    results_1 = [r.replace("Here is an image.","").strip() for r in results_gathered_1] 

    timediff=time.time()-start
    
    print(f"*** (1st) time elapsed: {timediff}")

    fail_1 = 0
    rejects_all = []
    for idx in range(len(corrects_all)): 
        generated_token_1 = process_img_token(results_1[idx]) 

        # filter out failed sample 
        if generated_token_1 is None:
            fail_1 += 1
            continue 
        else:
            rejects_all.append(generated_token_1)
    print(f"*** (1st) total: {len(corrects_all)} | success: {len(corrects_all) - fail_1} | fail: {fail_1}")



    """ Step 5. 2nd Generation """
    # run vllm (2차 생성)

    # imp_prompt = f"For a given question-answer pair, improve the answer (image) by correcting errors, bolstering aesthetic, aligning with the text description in the question, and providing comprehensive details.\n \
    #         Given Question: {prompts_old[idx]}\nOriginal Answer: {generated_token_1}\nRewritten Answer: "
    
    imp_prompts_all = [
    create_prompt(f"For a given question-answer pair, improve the answer (image) by correcting errors, bolstering aesthetic, aligning with the text description in the question, and providing comprehensive details.\nGiven Question: {prompts_old[idx]}\nOriginal Answer: {rejects_all[idx]}\nRewritten Answer: ", "") 
    for idx in range(len(rejects_all))
    ]
    
    # imp_prompts_all = [system_message + "USER: " + f"For a given question-answer pair, improve the answer (image) by correcting errors, bolstering aesthetic, aligning with the text description in the question, and providing comprehensive details.\n \
    #         Given Question: {prompts_old[idx]}\nOriginal Answer: {rejects_all[idx]}\nRewritten Answer: " + "\nASSISTANT: " for idx in range(len(rejects_all))]

    start=time.time()
    results_gathered_2 = list(map(lambda x: x.outputs[0].text, 
                                llm.generate(imp_prompts_all, sampling_params))) # same sampling strategy
    results_2 = [r.replace("Here is an image.","").strip() for r in results_gathered_2] 
    timediff=time.time()-start
    print(f"\n\n*** (2nd) time elapsed: {timediff}")    



    """ Step 6. Collecting Data into single Sample """
    fail_2 = 0
    for idx in range(len(rejects_all)):    
        generated_token_2 = process_img_token(results_2[idx]) 

        # filter out failed sample 
        if generated_token_2 is None:
            fail_2 += 1
            continue
        
        d = {"real": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": generated_token_2}], 
                "generated": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": rejects_all[idx]}]}
        
        if args.split == 'test':
            filename = f"{args.output_dir}/loser_{data_frac}_test.jsonl" # data convert 편의상 'loser' 유지
        else:
            filename = f"{args.output_dir}/loser_{data_frac}.jsonl" 
        with open(filename, 'a') as f:
            json.dump(d, f)
            f.write('\n')
    print(f"*** (2nd) total: {len(rejects_all)} | success: {len(rejects_all) - fail_2} | fail: {fail_2}")
    


    print("\n\n*** Sampling Strategy: ", sampling_params)
    ratio = (len(rejects_all) - fail_2) / len(corrects_all)
    print(f"\n*** Total Data Usage: {ratio:02f}%")


    # # Total 20 
    # gen_prompt = [
    # "Please show me a picture of",
    # "Please design an image of",
    # "Please produce a photo of",
    # "Please generate an image of",
    # "Please draw a painting of",
    # "I'd like to see a drawing of",
    # "I'd love to see an illustration of",
    # "I'd like to view an image of",
    # "I want to see a picture of",
    # "I would like to see a photo of",
    # "Show me a photo of",
    # "Generate a picture of",
    # "Show me a photograph of",
    # "Generate an image of",
    # "Can you make an image of",
    # "Can you draw a painting of",
    # "Can you produce a picture of",
    # "Can you generate a photo of",
    # "Can you depict a picture of",
    # "Can you show me an illustration of"
    # ]
    

    # failed_dir = os.path.join(args.output_dir, "fail")
    # if not os.path.exists(failed_dir):
    #     os.makedirs(failed_dir)
    # fail_log = f"{args.output_dir}/fail_log_{data_frac}.jsonl"
    
    # Microsoft ver.
    # imp_prompt = f"For a given question-answer pair, improve the answer by correcting errors, bolstering informativeness, aligning with the questions, and providing comprehensive detail.\n \
    #                 Given Question: {} \ 
    #                 Original Answer: {} \
    #                 Rewritten Answer: "


if __name__ == "__main__":
    main()
