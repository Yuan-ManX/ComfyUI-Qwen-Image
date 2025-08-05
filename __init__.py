from .nodes import LoadQwenImageModel, LoadQwenImagePrompt, LoadQwenImageNegativePrompt, QwenImage, SaveQwenImage

NODE_CLASS_MAPPINGS = {
    "LoadQwenImageModel": LoadQwenImageModel,
    "LoadQwenImagePrompt": LoadQwenImagePrompt,
    "LoadQwenImageNegativePrompt": LoadQwenImageNegativePrompt,
    "QwenImage": QwenImage,
    "SaveQwenImage": SaveQwenImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadQwenImageModel": "Load Qwen Image Model",
    "LoadQwenImagePrompt": "Load Qwen Image Prompt",
    "LoadQwenImageNegativePrompt": "Load Qwen Image Negative Prompt",
    "QwenImage": "Qwen Image",
    "SaveQwenImage": "Save Qwen Image",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
