# last update: 10/17

import json
import os
from datasets import load_dataset
import pyarrow.parquet as pq
import random

from seed_llama_utils import load_data, load_data_from_tar
from tqdm import tqdm

random.seed(42)
BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='/ssd0/data/seed_tokenized/dense_fusion_4v_100k_w_gen_seed_llama_sft_8b')
parser.add_argument('--output_dir', type=str, default='data_seed_llama/synthetic/iter0')

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 1. load train dataset 
train_data = []
try: 
    data_files = load_data(data_dir=os.path.join(input_dir, "train"), recursive=True)
    for data_file in tqdm(data_files, desc="\nLoading TRAIN pre-tokenized data ..."):
        train_data.extend(load_data_from_tar(data_file))

except:
    try:
        train_data = load_data_from_tar(os.path.join(input_dir, "train"))
    except Exception as e:
        print(e)
        raise ValueError("Error while loading chosen data !")
print("\n*** len of train data: ", len(train_data))


# 2. load test(val) dataset 
test_data = []
try: 
    data_files = load_data(data_dir=os.path.join(input_dir, "val"), recursive=True)
    for data_file in tqdm(data_files, desc="\nLoading TEST(VAL) pre-tokenized data ..."):
        test_data.extend(load_data_from_tar(data_file))

except:
    try:
        test_data = load_data_from_tar(os.path.join(input_dir, "val"))
    except Exception as e:
        print(e)
        raise ValueError("Error while loading chosen data !")
print("\n*** len of test data: ", len(test_data))



# 3. format the data 

# 3.1 train
formatted_train_data = []
for sample in train_data:
    # 기존) {"text": "", "real_image_ids": [], "gen_image_ids": [], "image_path": ""}
    
    # get item
    text = sample.get("text", None)
    real_image_ids = sample.get("real_image_ids", None)
    gen_image_ids = sample.get("gen_image_ids", None)
    # image_path = sample.get("image_path", None)

    # filter
    if (text is None) or (real_image_ids is None) or (gen_image_ids is None):
        print("filtered")
        continue


    real = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in real_image_ids]) + EOI_TOKEN 
    generated = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in gen_image_ids]) + EOI_TOKEN

    # create new format
    formatted_sample = {
        # chosen response
        "real": [
            {
                "role": "user",
                "content": f"{text} Please generate an image." # same
            },
            {
                "role": "assistant",
                "content": f"{real}"
            }
        ],
        # rejected response
        "generated": [
            {
                "role": "user",
                "content": f"{text} Please generate an image." # same
            },
            {
                "role": "assistant",
                "content": f"{generated}"
            }
        ]
    }

    formatted_train_data.append(formatted_sample)


# 3.2 test
formatted_test_data = []
for sample in test_data:
    # 기존) {"text": "", "real_image_ids": [], "gen_image_ids": [], "image_path": ""}
    
    # get item
    text = sample.get("text", None)
    real_image_ids = sample.get("real_image_ids", None)
    gen_image_ids = sample.get("gen_image_ids", None)

    # filter
    if (text is None) or (real_image_ids is None) or (gen_image_ids is None):
        print("filtered")
        continue

    real = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in real_image_ids]) + EOI_TOKEN 
    generated = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in gen_image_ids]) + EOI_TOKEN

    # create new format
    formatted_sample = {
        # chosen response
        "real": [
            {
                "role": "user",
                "content": f"{text} Please generate an image." # same
            },
            {
                "role": "assistant",
                "content": f"{real}"
            }
        ],
        # rejected response
        "generated": [
            {
                "role": "user",
                "content": f"{text} Please generate an image." # same
            },
            {
                "role": "assistant",
                "content": f"{generated}"
            }
        ]
    }

    formatted_test_data.append(formatted_sample)




print("\n\n========= Formatting Finished =========")
print("*** len of formatted train data: ", len(formatted_train_data))
print("*** len of formatted test data: ", len(formatted_test_data))


    

# 4. Save the data 

with open(f'{input_dir}/synthetic_train.json', 'w') as f:
    json.dump(formatted_train_data, f, indent=4)
with open(f'{input_dir}/synthetic_test.json', 'w') as f:
    json.dump(formatted_test_data, f, indent=4)

dataset = load_dataset('json', data_files=f'{input_dir}/synthetic_train.json',split='train')
dataset_test = load_dataset('json', data_files=f'{input_dir}/synthetic_test.json',split='train')

print(len(dataset))
print(len(dataset_test))

pq.write_table(dataset.data.table, f'{output_dir}/train_prefs-00000-of-00001.parquet')
pq.write_table(dataset_test.data.table, f'{output_dir}/test_prefs-00000-of-00001.parquet')


print(f"\n\n========= Saving Finished ... check {output_dir}=========")


