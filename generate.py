from diffusers import StableDiffusionPipeline
import torch


class Model:
    def __init__(self, checkpoint="checkpoints/lora30k", weight_name="pytorch_lora_weights.safetensors", device="cuda"):
        self.checkpoint = checkpoint
        state_dict, network_alphas = StableDiffusionPipeline.lora_state_dict(
            # Path to my trained lora output_dir
            checkpoint,
            weight_name=weight_name
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to(device)
        self.pipe.load_lora_into_unet(state_dict, network_alphas, self.pipe.unet, adapter_name='test_lora')
        self.pipe.load_lora_into_text_encoder(state_dict, network_alphas, self.pipe.text_encoder, adapter_name='test_lora')
        self.pipe.set_adapters(["test_lora"], adapter_weights=[1.0])


    def generate(self, prompt, negprompt='', steps=50, savedir=None, seed=1):
        lora_scale = 1.0
        image = self.pipe(prompt,
                     negative_prompt=negprompt,
                     num_inference_steps=steps,
                     cross_attention_kwargs={"scale": lora_scale},
                     generator=torch.manual_seed(seed)).images[0]
        if savedir is None:
            image.save(f"{self.checkpoint}/{'_'.join(prompt.replace('.', ' ').split(' '))}.png")
        else:
            image.save(f"{savedir}/{'_'.join(prompt.replace('.', ' ').split(' '))}.png")
        return image


if __name__ == "__main__":

    model = Model()

    prompt = 'A happy 55 year old male with blond hair and a goatee smiles with visible teeth.'
    negprompt = ''

    image = model.generate(prompt, negprompt=negprompt, steps=50, seed=42)
