from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    hf_hub_download(repo_id="michaeltrs/text2face",
                    filename="checkpoints/lora30k/pytorch_lora_weights.safetensors",
                    local_dir="checkpoints")

