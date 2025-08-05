import torch
from diffusers import DiffusionPipeline


class LoadQwenImageModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "Qwen/Qwen-Image"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen-Image"

    def load_model(self, model_path):
        model = model_path
        
        return (model,)


class LoadQwenImagePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "Qwen-Image"

    def load_prompt(self, text):
        prompt = text
        
        return (prompt,)


class QwenImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("PROMPT",),
                "positive_magic": ("STRING", {"default": "Repeat this image."}),
                "aspect_ratios": ("FLOAT", {"default": 3.0}),
                "num_inference_steps": ("FLOAT", {"default": 1.0}),
                "true_cfg_scale": ("STRING", {"default": "constant"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Qwen-Image"

    def generate(self, model, prompt, positive_magic, aspect_ratios, num_inference_steps, true_cfg_scale):
        
        # Load the pipeline
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device = "cuda"
        else:
            torch_dtype = torch.float32
            device = "cpu"
        
        pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        
        positive_magic = {
            "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
            "zh": "超清，4K，电影级构图" # for chinese prompt
        }
        
        # Generate image
        prompt = ''''''
        
        negative_prompt = " " # Recommended if you don't use a negative prompt.
        
        
        # Generate with different aspect ratios
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472)
        }
        
        width, height = aspect_ratios["16:9"]
        
        image = pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42)
        ).images[0]

        return (image,)


