
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
    parser.add_argument('--model', type=str, default='/data/checkpoints/t2i_dpo/spin/1019_spin_seed_llama_hf/lora-merged') # iter0-model
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='data_seed_llama/generated/iter1_vllm')
    parser.add_argument('--world_size', type=int, default=1) # TODO
    parser.add_argument('--input_dir', type=str, default='data_seed_llama/reformatted')
    parser.add_argument('--split', type=str, default='train') # TODO: generated one more for 'test; split
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


def main():
    args = parse_arguments()
    model_path = args.model
    data_frac = args.data_frac
    world_size = args.world_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    # Set the cache directory for Hugging Face models
    os.environ['HF_HOME'] = args.cache_dir  

    # load a base model and tokenizer
    # 추가) 사실 토크나이저는 크게 역할이 없다.

    # tokenizer = SeedLlamaTokenizer.from_pretrained(
    #                     pretrained_model_name_or_path='AILab-CVC/seed-tokenizer-2',
    #                     vit_precision='fp16',
    #                     diffusion_precision='fp16',
    #                     load_diffusion=False, # do not decode image, just save as `text token`
    #                     device='cpu',
    #                     encoder_url='https://huggingface.co/AILab-CVC/seed-tokenizer-2/resolve/main/seed_quantizer.pt',
    #                     diffusion_path='stabilityai/stable-diffusion-2-1-unclip',
    #                     ) 
    # tokenizer.pad_token_id = tokenizer.eos_token_id


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

# generation_config = {
#         'temperature': 1.0,
#         'num_beams': 1,
#         'max_new_tokens': 512,
#         'top_p': 0.5,
#         'do_sample': True
#     }

    # 인자명 참고: https://docs.vllm.ai/en/v0.6.0/dev/sampling_params.html
    # sampling_params = SamplingParams(temperature=0.8, num_beams=5, max_tokens=120) 
    sampling_params = SamplingParams(temperature=0.0, best_of=5, use_beam_search=True, max_tokens=120) 
    # sampling_params = SamplingParams(temperature=1.0, top_p=0.5, use_beam_search=False, max_tokens=120) 
    # sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256) 

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

    # modify prompt
    system_message = ""
    instruction = 'Please generate an image.'

    prompts_all = [system_message + "USER: " + data[idx][0]['content'] + " " + instruction + "\nASSISTANT: " for idx in range(len(data))]
    # prompts_all = ["### Instruction: " + data[idx][0]['content'] + "\n\n### Response: " for idx in range(len(data))]
    prompts_old = [data[idx][0]['content'] for idx in range(len(data))]
    corrects_all = [data[idx][1]['content'] for idx in range(len(data))]

    start=time.time()

    #run vllm
    results_gathered = list(map(lambda x: x.outputs[0].text, 
                                llm.generate(prompts_all, sampling_params)))

    # results = [r.replace("</s>","").lstrip() for r in results_gathered] 
    results = [r.replace("Here is an image.","").strip() for r in results_gathered] 

    timediff=time.time()-start
    print(f"time elapsed: {timediff}")

    # collecting data 
    failed = 0
    for idx in range(len(corrects_all)):
        generated_token = process_img_token(results[idx]) 
        if generated_token is None:
            generated_token = ""
            failed += 1
        # print("\n\n*** debug 2 ", generated_token)
        d = {"real": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": corrects_all[idx]}], "generated": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": generated_token}]}
        # d = {"real": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": corrects_all[idx]}], "generated": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": results[idx]}]}
        if args.split == 'test':
            filename = f"{args.output_dir}/loser_{data_frac}_test.jsonl" 
        else:
            filename = f"{args.output_dir}/loser_{data_frac}.jsonl" 
        with open(filename, 'a') as f:
            json.dump(d, f)
            f.write('\n')
    print(f"\n\n*** Total failed case in generation: {failed} / out of {len(corrects_all)} ***")


if __name__ == "__main__":
    main()
