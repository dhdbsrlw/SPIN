# The code is from "https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py".

# bunny
# conda env: seed2
# /project/SPIN$

# seed-llama
# conda env: spin2 (for vllm inference)
# /project/SPIN$

# TODO: use_vllm arg 추가

import sys
import time
from nltk.stem import *
import nltk
import json
# from pattern.en import singularize # 설치 오류
import inflect # 대체
# from misc import *

from vllm import LLM, SamplingParams
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import transformers
import traceback
from accelerate.utils import DistributedType
from peft import LoraConfig
sys.path.append("/home/yjoh/project/")
from transformers import GPTQConfig, deepspeed
from mDPO.bunny.bunny_utils.constants import *
from mDPO.bunny.bunny_utils.conversation import *
from mDPO.bunny.bunny_utils.util.mm_utils import *
from mDPO.bunny.modeling_bunny_phi import mDPOBunnyPhiForCausalLM
from mDPO.seed_llama.SEED.MultiModalLLM.src.model.llama_xformer import LlamaForCausalLM 
from mDPO.seed_llama.SEED.models.seed_llama_tokenizer import SeedLlamaTokenizer 
from MAGVLT2.MultiModalLLM.utils.config import *

lemma = nltk.wordnet.WordNetLemmatizer()

# Object HalBench (Rohrbach et al., 2018) is a widely adopted benchmark to assess object hallucination. 
# We follow the setting of Yu et al. (2024a) to augment the benchmark with eight diverse prompts and evaluating on 300 instances. 

def get_transform(type='clip', keep_ratio=False, image_size=224, normalize=True):
    if type == 'clip':
        transform = []
        if keep_ratio:
            transform.extend([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ])
        else:
            transform.append(transforms.Resize((image_size, image_size)))
        transform.extend([
            transforms.ToTensor(),
        ])
        if normalize:
            transform.append(
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            )
        return transforms.Compose(transform)
    else:
        raise NotImplementedError

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def generate_eval_captions(cfg, model, tokenizer): 

    # COCO karpathy test set - img_dir, ann_dir
    # 8 different prompt

    model_type = cfg.model_args.model_type
    print(f"Start generating captions with {model_type} model ...\n\n")
    generation_config = cfg.eval.generation_config

    with open(cfg.eval.input_dir, 'r') as f:
        data = [json.loads(line.strip()) for line in f]

    prompts_all = []
    generates_all = []
    image_ids_all = []
    
    if model_type == "seed_llama":
        image_transform = get_transform()
        assert image_transform is not None, "SEED LLaMA needs image_transform for image decoding."
            
        IMG_TOKEN = '<img_{:05d}>'
        BOI_TOKEN = '<img>'
        EOI_TOKEN = '</img>'
        SYSTEM_PROMPT = ''
        S_TOKEN = 'USER: '
        E_TOKEN = 'ASSISTANT: '
        SEP = '\n'

        sampling_params = SamplingParams(temperature=cfg.eval.generation_config.temperature,
                                        top_p=cfg.eval.generation_config.top_p,
                                        best_of=cfg.eval.generation_config.num_beams,
                                        max_tokens=cfg.eval.generation_config.max_new_tokens,
                                        # min_tokens=cfg.eval.generation_config.min_new_tokens,
                                        # do_sample: True
                                        )

    # generate captions
    start=time.time()
    for item in tqdm(data, desc="Generating: "):

        image = Image.open(BytesIO(base64.b64decode(item["image"]))) 
        image_id = item["image_id"]   
        image_ids_all.append(image_id)
        
        # Model 1
        if model_type == "bunny":

            prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{item['question']} ASSISTANT:"
            text_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
            input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(model.device)
            image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=model.device) # [bs, 3, 384, 384]
            prompts_all.append(prompt)

            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=input_ids,
                        images=image_tensor, # image_tensor.unsqueeze(0).to(dtype=model.dtype, device='cuda', non_blocking=True), # IMPORTANT
                        **generation_config,
                        pad_token_id=tokenizer.pad_token_id
                    )

                output_ids = output_ids[:, input_ids.shape[1]:]

                # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                # outputs = outputs.strip().split("##OUTPUT\n")[1]

                generates_all.append(outputs)

            except Exception as e1:
                print(e1)
                continue

            finally:
                torch.cuda.empty_cache() # 필수로 GPU memory flush

        # Model 2
        elif model_type == "seed_llama": # use vllm otherwise Bunny

            try:
                # transform and encode image
                image_tensor = image_transform(image)
                image_id = tokenizer.encode_image(image_torch=image_tensor.cuda())
                image_token = image_id.view(-1).cpu().tolist()

                # format and complete input sentence 
                image_token = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_token]) + EOI_TOKEN        

                prompt = SYSTEM_PROMPT + S_TOKEN + image_token + item["question"] + SEP + E_TOKEN 
                # print(prompt)
                # """
                # (example)
                # USER: <img><img_05115><img_02157><img_06971><img_06971><img_02157><img_04952><img_02732><img_06670><img_08094><img_02157><img_02157><img_08094><img_02732><img_02732><img_06971><img_02
                # 852><img_02157><img_03556><img_02157><img_02852><img_02161><img_01718><img_02161><img_02157><img_01448><img_02852><img_06919><img_08132><img_04202><img_06971><img_02334><img_05730></i
                # mg>Provide a thorough description of the given image.                                                                                                                                  
                # ASSISTANT:   
                # """
                prompts_all.append(prompt)

                # add for official sft
                input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False) # 주의, 절대 EOS token_id 를 붙여서는 안된다.
                input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(model.device)
                # print("DEBUG")
                # print(input_ids)
                # print()

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=input_ids,
                        **generation_config,
                        pad_token_id=tokenizer.pad_token_id
                    )

                output_ids = output_ids[:, input_ids.shape[1]:]

                # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                outputs = outputs.replace("Here is an image.","").strip()
                # outputs = outputs.strip().split("##OUTPUT\n")[1]

                # print(outputs)
                # print()

                generates_all.append(outputs)

            except Exception as e2:
                traceback.print_exc()
                print(f"Error: {e2}")
                continue

            finally:
                torch.cuda.empty_cache() # 필수로 GPU memory flush
 
            # end of the loop

    # if model_type == "seed_llama":

    #     print("\nRemoving the existing tokenizer on the CPU ...")
    #     del tokenizer 

    #     model = LLM(
    #     model=cfg.model_args.model_name_or_path, 
    #     tensor_parallel_size=world_size,
    #     tokenizer="AILab-CVC/seed-tokenizer-2", # tokenizer,
    #     )

    #     with torch.no_grad():
    #         outputs = list(map(lambda x: x.outputs[0].text, 
    #                     model.generate(prompts_all, sampling_params)))
    #         # print("\n\nDEBUG 2")
    #         # print(outputs)
    #         generates_all = [o.replace("Here is an image.","").strip() for o in outputs] 

    timediff=time.time()-start
    print(f"\n# Time elapsed: {timediff}")
    print("# Total num of Data: ", len(data))
    print("# Success: ",len(generates_all))
   

    print(f"Start saving generated results at {cfg.eval.output_dir} ...")
    if not os.path.exists(cfg.eval.output_dir):
        os.makedirs(cfg.eval.output_dir, exist_ok=True)

    filename = f"{cfg.eval.output_dir}/{model_type}_generated_caption.json"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('[')  # Start JSON array

    # collecting data
    for idx in range(len(generates_all)):
        # generated_token = results[idx]
        d = {
            "image_id": image_ids_all[idx],
            # "question": prompts_all[idx],
            "caption": generates_all[idx],
        }
        
        # with open(result_file, 'w') as f:
        #     json.dump(unique_data, f, indent=4)
        with open(filename, 'a') as f:
            if idx > 0:
                f.write(',\n')  # Add a comma and newline before each entry except the first
            json.dump(d, f, indent=4)

    # Finalize the JSON array
    with open(filename, 'a') as f:
        f.write('\n]')
        
    print("\n\n*** Generating Captions Done ! ***")


def combine_coco_captions(annotation_path): # =coco_path

    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'val')))
    train_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'train')))
    all_caps = {'info': train_caps['info'],
                'licenses': train_caps['licenses'],
                'images': val_caps['images'] + train_caps['images'],
                'annotations': val_caps['annotations'] + train_caps['annotations']}

    return all_caps 

def combine_coco_instances(annotation_path):

    # TODO: coco-instance 파일 존재여부 확인
    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'val')))
    train_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'train')))
    all_instances = {'info': train_instances['info'],
                     'licenses': train_instances['licenses'],
                     'type': train_instances['licenses'],
                     'categories': train_instances['categories'],
                     'images': train_instances['images'] + val_instances['images'],
                     'annotations': val_instances['annotations'] + train_instances['annotations']}

    return all_instances 

class CHAIR(object):

    def __init__(self, imids, coco_path, synonyms_path):

        self.imid_to_objects = {imid: [] for imid in imids}

        self.coco_path = coco_path

        #read in synonyms
        synonyms = open(synonyms_path).readlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = [] #mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        #Some hard coded rules for implementing CHAIR metrics on MSCOCO
        
        #common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']
        
        #Hard code some rules for special cases in MSCOCO
        #qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        #qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ['jet', 'train']
        
        #double_word_dict will map double words to the word they should be treated as in our analysis
        
        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' %animal_word] = animal_word
            self.double_word_dict['adult %s' %animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' %vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'

    def _load_generated_captions_into_evaluator(self, cap_file):

        '''
        Meant to save time so imid_to_objects does not always need to be recomputed.
        '''
        #Read in captions        
        self.caps, self.imids, self.metrics = load_generated_captions(cap_file) # modify

        assert self.imids == set(self.imid_to_objects.keys())

    def caption_to_words(self, caption):
    
        '''
        Input: caption
        Output: MSCOCO words in the caption
        '''
    
        #standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        
        p = inflect.engine()
        words = [p.singular_noun(w) if p.singular_noun(w) else w for w in words]
        # words = [singularize(w) for w in words]
    
        #replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
           idxs.append(i) 
           double_word = ' '.join(words[i:i+2])
           if double_word in self.double_word_dict: 
               double_words.append(self.double_word_dict[double_word])
               i += 2
           else:
               double_words.append(words[i])
               i += 1
        words = double_words
    
        #toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']
    
        #get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) \
                if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        #return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words

    def get_annotations_from_segments(self):
        '''
        Add objects taken from MSCOCO segmentation masks
        '''

        coco_segments = combine_coco_instances(self.coco_path)
        segment_annotations = coco_segments['annotations']

        #make dict linking object name to ids
        id_to_name = {} #dict with id to synsets 
        for cat in coco_segments['categories']:
            id_to_name[cat['id']] = cat['name']

        for i, annotation in enumerate(segment_annotations):
            sys.stdout.write("\rGetting annotations for %d/%d segmentation masks" 
                              %(i, len(segment_annotations)))
            imid = annotation['image_id']
            if imid in self.imid_to_objects:
                node_word = self.inverse_synonym_dict[id_to_name[annotation['category_id']]]
                self.imid_to_objects[imid].append(node_word)
        print("\n")
        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def get_annotations_from_captions(self):
        '''
        Add objects taken from MSCOCO ground truth captions 
        '''

        coco_caps = combine_coco_captions(self.coco_path)
        caption_annotations = coco_caps['annotations']

        for i, annotation in enumerate(caption_annotations):
            sys.stdout.write('\rGetting annotations for %d/%d ground truth captions' 
                              %(i, len(coco_caps['annotations'])))
            imid = annotation['image_id']
            if imid in self.imid_to_objects:
                _, node_words, _, _ = self.caption_to_words(annotation['caption'])
                self.imid_to_objects[imid].update(node_words)
        print("\n")

        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def get_annotations(self):

        '''
        Get annotations from both segmentation and captions.  Need both annotation types for CHAIR metric.
        '''

        self.get_annotations_from_segments() 
        self.get_annotations_from_captions() 

    def compute_chair(self, cap_file):
    
        '''
        Given ground truth objects and generated captions, determine which sentences have hallucinated words.
        '''
    
        self._load_generated_captions_into_evaluator(cap_file)

        imid_to_objects = self.imid_to_objects
        caps = self.caps
        imids = self.imids # modify
 
        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.

        output = {'sentences': []} 
    

        # for i, cap_eval in enumerate(caps):
    
        #     cap = cap_eval['caption']
        #     imid = cap_eval['image_id']

        for cap, imid in zip(caps, imids):
    
            #get all words in the caption, as well as corresponding node word
            words, node_words, idxs, raw_words = self.caption_to_words(cap) 
 
            gt_objects = imid_to_objects[imid]
            cap_dict = {'image_id': imid, # cap_eval['image_id'], 
                        'caption': cap,
                        'mscoco_hallucinated_words': [],
                        'mscoco_gt_words': list(gt_objects),
                        'mscoco_generated_words': list(node_words),
                        'hallucination_idxs': [], 
                        'words': raw_words 
                        }
   
            # cap_dict['metrics'] = {'Bleu_1': cap_eval['Bleu_1'],
            #                        'Bleu_2': cap_eval['Bleu_2'],
            #                        'Bleu_3': cap_eval['Bleu_3'],
            #                        'Bleu_4': cap_eval['Bleu_4'],
            #                        'METEOR': cap_eval['METEOR'],
            #                        'CIDEr': cap_eval['CIDEr'],
            #                        'SPICE': cap_eval['SPICE'],
            #                        'ROUGE_L': cap_eval['ROUGE_L'],
            #                        'CHAIRs': 0,
            #                        'CHAIRi': 0}

            cap_dict['metrics'] = {'CHAIRs': 0,
                                   'CHAIRi': 0}
 
            #count hallucinated words
            coco_word_count += len(node_words) 
            hallucinated = False
            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated_word_count += 1 
                    cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                    cap_dict['hallucination_idxs'].append(idx)
                    hallucinated = True      
    
            #count hallucinated caps
            num_caps += 1
            if hallucinated:
               num_hallucinated_caps += 1
    
            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = 0.
            if len(words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words'])/float(len(words))
   
            output['sentences'].append(cap_dict)
 
        chair_s = (num_hallucinated_caps/num_caps)
        chair_i = (hallucinated_word_count/coco_word_count)
    
        # output['overall_metrics'] = {'Bleu_1': self.metrics['Bleu_1'],
        #                              'Bleu_2': self.metrics['Bleu_2'],
        #                              'Bleu_3': self.metrics['Bleu_3'],
        #                              'Bleu_4': self.metrics['Bleu_4'],
        #                              'METEOR': self.metrics['METEOR'],
        #                              'CIDEr': self.metrics['CIDEr'],
        #                              'SPICE': self.metrics['SPICE'],
        #                              'ROUGE_L': self.metrics['ROUGE_L'],
        #                              'CHAIRs': chair_s,
        #                              'CHAIRi': chair_i}

        output['overall_metrics'] = {'CHAIRs': chair_s,
                                     'CHAIRi': chair_i}
    
    
        return output 

def load_generated_captions(cap_file):
    # Read in captions        
    caps = json.load(open(cap_file))

    # 원본 코드
    #    try:
    #        metrics = caps['overall']
    #        caps = caps['imgToEval'].values()
    #        imids = set([cap['image_id'] for cap in caps])
    #    except:
    #        raise Exception("Expect caption file to consist of a dectionary with sentences correspdonding to the key 'imgToEval'")

    # 수정 코드
    try:
       metrics = None
       imids = set([cap['image_id'] for cap in caps])
       caps = set([cap['caption'] for cap in caps])
    
    except Exception as e:
       print(f"Error! {e}")

    # `imids` are used in meaningful way.
    return caps, imids, metrics

def save_hallucinated_words(save_dir, cap_file, cap_dict): 
    tag = cap_file.split('/')[-1] 
    with open(f'{save_dir}/hallucinated_words_%s' %tag, 'w') as f:
        json.dump(cap_dict, f)

def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    # metric_string = "%0.01f\t%0.01f\t%0.01f\t%0.01f\t%0.01f" %(
    #                                               sentence_metrics['SPICE']*100,
    #                                               sentence_metrics['METEOR']*100,
    #                                               sentence_metrics['CIDEr']*100,
    #                                               sentence_metrics['CHAIRs']*100,
    #                                               sentence_metrics['CHAIRi']*100)

    metric_string = "%0.01f\t%0.01f" %(
                                        sentence_metrics['CHAIRs']*100,
                                        sentence_metrics['CHAIRi']*100)

    if not quiet:
        # print("SPICE\tMETEOR\tCIDEr\tCHAIRs\tCHAIRi")
        print("CHAIRs\tCHAIRi")
        print(metric_string)

    else:
        return metric_string
 

if __name__ == '__main__':

    # 0. 세팅
    cfg, _ = build_config(cfg_path="eval/obj_halbench/config.yaml")

    model_type = cfg.model_args.model_type

    # 1. 생성 시작
    if cfg.eval.do_gen:

        device_map = None
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1

        if model_type == "bunny":

            if getattr(cfg.training_args, "deepspeed", None) and getattr(
                cfg.lora_args, "q_lora", False
            ):
                cfg.training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

            # 해당없음
            if cfg.lora_args.q_lora:
                device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
                if len(cfg.training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
                    print("FSDP or ZeRO3 are not incompatible with QLoRA.")

            # Set RoPE scaling factor
            config = transformers.AutoConfig.from_pretrained(
                cfg.model_args.model_name_or_path,
                cache_dir=cfg.model_args.cache_dir,
                trust_remote_code=True,
                fp32=True,
            )
            config.use_cache = False
            config.embd_pdrop = 0

            # Load model and tokenizer
            model = mDPOBunnyPhiForCausalLM.from_pretrained(
                cfg.model_args.model_name_or_path,
                config=config,
                cache_dir=cfg.model_args.cache_dir,
                device_map='auto',
                trust_remote_code=True,
                quantization_config=GPTQConfig(bits=4, disable_exllama=True)
                if cfg.lora_args.use_lora and cfg.lora_args.q_lora
                else None,
            )

            if not cfg.lora_args.use_lora:
                if (
                    cfg.training_args.fix_vit
                    and hasattr(model, "transformer")
                    and hasattr(model.transformer, "visual")
                ):
                    model.transformer.visual.requires_grad_(False)
                    if hasattr(model.transformer.visual, "attn_pool"):
                        model.transformer.visual.attn_pool.requires_grad_(True)
            
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                cfg.model_args.model_name_or_path,
                cache_dir=cfg.model_args.cache_dir,
                model_max_length=cfg.training_args.model_max_length,
                padding_side="right",
                use_fast=False,
                trust_remote_code=True,
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id

            if cfg.lora_args.use_lora:
                if cfg.lora_args.lora_target_modules == "all-linear":
                    lora_target_modules = find_all_linear_names(model)
                elif "," in cfg.lora_args.lora_target_modules:
                    lora_target_modules = cfg.lora_args.lora_target_modules.split(",")
                else:
                    lora_target_modules = cfg.lora_args.lora_target_modules

                lora_config = LoraConfig(
                    r=cfg.lora_args.lora_r,
                    lora_alpha=cfg.lora_args.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=cfg.lora_args.lora_dropout,
                    bias=cfg.lora_args.lora_bias,
                    task_type="CAUSAL_LM",
                    # modules_to_save=None,  # This argument serves for adding new tokens.
                )

        # cfg.model_args.model_type == "seed_llama":
        else:
            # model = None

            config = transformers.AutoConfig.from_pretrained(
                cfg.model_args.model_name_or_path,
                cache_dir=cfg.model_args.cache_dir,
                torch_dtype="fp16", # 이슈
                low_cpu_mem_usage=True
            )
            
            model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=cfg.model_args.model_name_or_path,
                config=config,
                device_map='cuda',
            )

            tokenizer = SeedLlamaTokenizer.from_pretrained(
                            pretrained_model_name_or_path='AILab-CVC/seed-tokenizer-2',
                            vit_precision='fp16',
                            diffusion_precision='fp16',
                            load_diffusion=False, # do not decode image, just save as `text token`
                            device='cuda', # 필수로 CUDA 로 로드
                            encoder_url='https://huggingface.co/AILab-CVC/seed-tokenizer-2/resolve/main/seed_quantizer.pt',
                            diffusion_path='stabilityai/stable-diffusion-2-1-unclip',
                            ) 
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left" 
            # Warning: A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.

        generate_eval_captions(cfg, model, tokenizer)


    # 2. 측정 시작
    if cfg.eval.do_cal:

        cap_file = cfg.eval.cap_file
        if cap_file is None:
            cap_file = f"{cfg.eval.output_dir}/{model_type}_generated_caption.json"

        _, imids, _ = load_generated_captions(cap_file)

        evaluator = CHAIR(imids, cfg.eval.coco_path, cfg.eval.synm_path) 
        evaluator.get_annotations()
        cap_dict = evaluator.compute_chair(cap_file) 
        
        # 3. 결과 출력
        print_metrics(cap_dict)

        # add
        save_dir=cfg.eval.hallucinated_word_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        save_hallucinated_words(save_dir, cap_file, cap_dict)

        print("\n*** All Done ***")
