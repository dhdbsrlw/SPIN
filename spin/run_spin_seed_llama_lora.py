#!/usr/bin/env python
# 
# Adapted from https://github.com/huggingface/alignment-handbook 
import logging
import sys
import os

from mDPO.seed_llama.SEED.models.seed_llama_tokenizer import SeedLlamaTokenizer # modify
from mDPO.seed_llama.SEED.MultiModalLLM.src.model.llama_xformer import LlamaForCausalLM # modify

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".spin-root", pythonpath=True, cwd=True)

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
from peft import LoraConfig, prepare_model_for_kbit_training

from accelerate import Accelerator
from alignment import (
    DataArguments,
    SPINConfig,
    H4ArgumentParser,
    ModelArguments,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel
from alignment import SPINTrainer
from torch.utils.data import Subset
import re

from spin.alignment.data import DEFAULT_CHAT_TEMPLATE # modify
from dataclasses import dataclass, field


# # Borrowed from peft.util.get_peft_model_state_dict
# def get_peft_state_maybe_zero_3(named_params, bias):
#     if bias == "none":
#         to_return = {k: t for k, t in named_params if "lora_" in k}
#     elif bias == "all":
#         to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
#     elif bias == "lora_only":
#         to_return = {}
#         maybe_lora_bias = {}
#         lora_bias_names = set()
#         for k, t in named_params:
#             if "lora_" in k:
#                 to_return[k] = t
#                 bias_name = k.split("lora_")[0] + "bias"
#                 lora_bias_names.add(bias_name)
#             elif "bias" in k:
#                 maybe_lora_bias[k] = t
#         for k, t in maybe_lora_bias:
#             if bias_name in lora_bias_names:
#                 to_return[bias_name] = t
#     else:
#         raise NotImplementedError
#     to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
#     return to_return

# def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
#     to_return = {k: t for k, t in named_params if "lora_" not in k}
#     if require_grad_only:
#         to_return = {k: t for k, t in to_return.items() if t.requires_grad}
#     to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
#     return to_return


# def safe_save_model_for_hf_trainer(
#     trainer: transformers.Trainer, output_dir: str, bias="none"
# ):
#     """Collects the state dict and dump to disk."""
#     # check if zero3 mode enabled
#     if deepspeed.is_deepspeed_zero3_enabled():
#         state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
#     else:
#         if trainer.args.use_lora:
#             state_dict = get_peft_state_maybe_zero_3(
#                 trainer.model.named_parameters(), bias
#             )
#         else:
#             state_dict = trainer.model.state_dict()
#     if trainer.args.should_save and trainer.args.local_rank == 0:
#         trainer._save(output_dir, state_dict=state_dict)

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


def apply_chat_template(
    example, tokenizer, task, assistant_prefix="<|assistant|>\n"
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if all(k in example.keys() for k in ("real", "generated")):
        # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
        prompt_messages = [[msg for msg in example["real"] if msg["role"] == "user"][0]]
        # Insert system message
        if example["real"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
        else:
            prompt_messages.insert(0, example["real"][0])

        real_messages = example["real"][1:]
        generated_messages = example["generated"][1:]
        example["text_real"] = tokenizer.apply_chat_template(real_messages, tokenize=False)
        example["text_generated"] = tokenizer.apply_chat_template(generated_messages, tokenize=False)
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        example["text_real"] = _strip_prefix(example["text_real"], assistant_prefix)
        example["text_generated"] = _strip_prefix(example["text_generated"], assistant_prefix)
    else:
        raise ValueError(
            f"Require `[real, generated]` keys but found {list(example.keys())}"
            )
    return example

logger = logging.getLogger(__name__)


def main():

    # modify (add)
    os.environ["WANDB_PROJECT"] = "VLM_SPIN"

    parser = H4ArgumentParser((ModelArguments, DataArguments, SPINConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator()

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn

    # Qformer.bert.encoder.layer error 해결 (qformer_qunatizer.py 참고)
    tokenizer = SeedLlamaTokenizer.from_pretrained(
                        pretrained_model_name_or_path='AILab-CVC/seed-tokenizer-2',
                        vit_precision='fp16',
                        diffusion_precision='fp16',
                        load_diffusion=False, 
                        device='cpu',
                        encoder_url='https://huggingface.co/AILab-CVC/seed-tokenizer-2/resolve/main/seed_quantizer.pt',
                        diffusion_path='stabilityai/stable-diffusion-2-1-unclip',
                        ) 

    tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    # if tokenizer.model_max_length > 100_000:
        # tokenizer.model_max_length = 8192

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE


    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "spin"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_real -> real and text_generated -> generated
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_real": "real", "text_generated": "generated"}
        )

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        # pretrained_model_name_or_path=model_args.model_name_or_path, # revision=model_args.model_revision,  # modify (originally, not exist)
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None, # modify
        quantization_config=quantization_config, # modify
        cache_dir="/ssd0/checkpoints", # modify
    )

    # modify
    # model = LlamaForCausalLM.from_pretrained(
    #     **model_kwargs
    # )

    # config = transformers.AutoConfig.from_pretrained(
    #     model_args.model_name_or_path,
    #     trust_remote_code=True,
    #     fp32=True,
    # )
    # config.use_cache = False
    # config.embd_pdrop = 0

    # model = model_args.model_name_or_path
    model = model_args.model_name_or_path

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None


    if model_args.use_peft:
        if model_args.lora_target_modules == "all-linear":
            lora_target_modules = find_all_linear_names(model)
        elif "," in model_args.lora_target_modules:
            lora_target_modules = model_args.lora_target_modules.split(",") # TODO: 이에 따라 config.yaml 파일 수정
        else:
            lora_target_modules = model_args.lora_target_modules

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=model_args.lora_dropout,
            task_type="CAUSAL_LM",
            # modules_to_save=None,  # This argument serves for adding new tokens.
        )


    #########################
    # Instantiate spin trainer
    #########################
    spin_trainer = SPINTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=lora_config if model_args.use_peft else None, # get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    train_result = spin_trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    spin_trainer.log_metrics("train", metrics)
    spin_trainer.save_metrics("train", metrics)
    spin_trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    spin_trainer.save_model(training_args.output_dir)
    # Save everything else on main process 
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        # modify (주석 처리)
        # spin_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        spin_trainer.model.config.use_cache = True
        spin_trainer.model.config.save_pretrained(training_args.output_dir)

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()

# env: spin2
# loc: /project
# cmd: CUDA_VISIBLE_DEVICES=8,9 accelerate launch --config_file SPIN/configs/deepspeed_zero3_seed_llama.yaml --num_processes 2 --main_process_port 29500 SPIN/spin/run_spin_seed_llama.py configs/config_seed_llama.yaml