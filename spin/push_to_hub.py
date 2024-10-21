from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig

HUGGINGFACE_USERNAME="dhdbsrlw"
# huggingface-cli login --token "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"

# Load the base model
base_model_path = "/data/checkpoints/t2i_dpo/spin/1019_spin_seed_llama_hf/checkpoint-3750"  # Path to the folder with base model
model = LlamaForCausalLM.from_pretrained(base_model_path)

# Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained(base_model_path)

# Load the LoRA adapter config and model
# adapter_model_path = f"{base_model_path}/adapter_model.safetensors"
# adapter_config_path = f"{base_model_path}/adapter_config.json"
# peft_config = PeftConfig.from_json_file(adapter_config_path)

model = PeftModel.from_pretrained(model, base_model_path)

# Set model to eval mode
model.eval()

# Merge LoRA adapter weights into the base model
model = model.merge_and_unload()  # This can take several minutes on cpu
model._hf_peft_config_loaded = False


# Save the merged model
save_path = "/data/checkpoints/t2i_dpo/spin/1019_spin_seed_llama_hf/lora-merged"
model.save_pretrained(save_path)
# tokenizer.save_pretrained(save_path)


# Push the model and tokenizer to the hub
model.push_to_hub(f"{HUGGINGFACE_USERNAME}/merged_llama_lora_model")
# tokenizer.push_to_hub(f"{HUGGINGFACE_USERNAME}/merged_llama_lora_model")


# from huggingface_hub import HfApi, HfFolder

# # Define your model repository and paths
# repo_id = "dhdbsrlw/<model-name>"
# model_ckpt_path = "path_to_your_model/pytorch_model.bin"

# # Log in using your token
# api = HfApi()
# api.upload_file(
#     path_or_fileobj=model_ckpt_path,
#     path_in_repo="pytorch_model.bin",  # Where it should be stored in the repo
#     repo_id=repo_id
# )

