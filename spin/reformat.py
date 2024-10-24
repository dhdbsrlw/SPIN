# 수정 부분 modify 표기

from datasets import load_dataset
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
import logging
import os
import random

from seed_llama_utils import load_data, load_data_from_tar
from tqdm import tqdm


def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data_seed_llama/reformatted')
    parser.add_argument('--data', type=str, default='/data/preference_data/seed_tokenized/dense_fusion_4v_100k_w_gen_seed_llama_sft_8b')
    parser.add_argument('--train_size', type=int) # modify
    return parser.parse_args()


def seed_image_tokenize(image_token_ids: list) -> str:
    BOI_TOKEN = '<img>'
    EOI_TOKEN = '</img>'
    IMG_TOKEN = '<img_{:05d}>'
    
    image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_token_ids]) + EOI_TOKEN
    return image_tokens



# modify (add)
def load_and_process_data_densefusion(dataset_dir, split): 
    # complete dataset dir
    dataset = []
    reformatted_dataset = []
    dataset_dir = os.path.join(dataset_dir, split)
    
    # load and process
    try:
        df = load_data(dataset_dir) # return as List
        for data_file in df:
            dataset.extend(load_data_from_tar(data_file))

        for message in tqdm(dataset):
            caption = message["text"]
            real = message["real_image_ids"]

            reformatted_dataset.append({
                "generated": [
                    {"role": "user",
                    "content": caption},  # same
                    {"role": "assistant",
                    "content": ""} # fix
                ],
                "real": [
                    {"role": "user",
                    "content": caption}, # same
                    {"role": "assistant",
                    "content": seed_image_tokenize(real)} 
                ]
                })
            
        # reformatted_data = [{
        #     'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
        #     'real': [message['messages'][0], message['messages'][1]]
        # } for message in dataset]

        return reformatted_dataset
    
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        # raise ValueError
        return []


def load_and_process_data_ultrachat(dataset_name, split, train_size: int=50000): # modify
    try:
        dataset = load_dataset(dataset_name, split=split)
        reformatted_data = [{
            'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset]
        # modify
        if "train" in split: 
            reformatted_data = reformatted_data[:train_size]
        return reformatted_data
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []

def load_and_process_data_tulu(dataset_name, input_split, test_split: float=0.1): 
    try:
        dataset = load_dataset(dataset_name, split=input_split)
        dataset = dataset.train_test_split(test_size=test_split)
        reformatted_train_data = [{
            'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset["train"]]
        reformatted_test_data = [{
            'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset["test"]]
        return reformatted_train_data, reformatted_test_data 
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []

def save_to_json(data, path):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logging.error(f"Error saving data to {path}: {e}")

def save_to_parquet(dataset, path):
    try:
        pq.write_table(dataset.data.table, path)
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")

def main():
    args = setup_arg_parser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data == 'HuggingFaceH4/ultrachat_200k':
        train_data = load_and_process_data_ultrachat(args.data, 'train_sft', train_size=args.train_size) # modify
        test_data = load_and_process_data_ultrachat(args.data, 'test_sft') 
    elif "tulu-v2-sft-mixture" in args.data:
        train_data, test_data = load_and_process_data_tulu(args.data, 'train', test_split=0.1)
    elif "dense_fusion" in args.data:
        train_data = load_and_process_data_densefusion(args.data, 'train') # args.data is 'base dataset dir'
        test_data = load_and_process_data_densefusion(args.data, 'val') 
    else:
        raise ValueError(f"current {args.data} dataset is not supported")

    train_json_path = output_dir / 'train.json'
    test_json_path = output_dir / 'test.json'

    save_to_json(train_data, train_json_path)
    save_to_json(test_data, test_json_path)

    dataset = load_dataset('json', data_files=str(train_json_path), split='train')
    dataset_test = load_dataset('json', data_files=str(test_json_path), split='train')

    save_to_parquet(dataset, output_dir / 'train_prefs-00000-of-00001.parquet')
    save_to_parquet(dataset_test, output_dir / 'test_prefs-00000-of-00001.parquet')

    os.remove(train_json_path)
    os.remove(test_json_path)

if __name__ == "__main__":
    main()