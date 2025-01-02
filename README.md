# Modal FLUX LoRA toolkit

Have you ever wondered why [Fal AI](https://fal.ai)'s LoRA trainer is so fast? I recently was working on [Atelier AI](https://atelierai.me) which is an equivalent of PhotoAI for Iranian/Persian speaking users and I needed to make the train procedure fast. I first wanted to use Fal AI, but I realized that $2 per train is too much for me. In the other hand, I could use [Modal](https://modal.com) in order to do my training and stuff. 

So I did a lot of research personally to find out these tips about making train faster: 

- Using a powerful GPU can help, but it's not everything. 
- We only can use one or two layers of LoRA in order to fit our concepts in it (if you see [config_example.yaml](./config_example.yaml) file, you will notice I've used 4.)
- If we disable sampling, we can make it even faster. 
- On powerful GPU's such as _A100 80GB_ or _H100_ we won't need to worry about `lowvram` and we can disable it (and therefore make the process faster.)

<p align="center">
    <img src="https://mann-e-images.storage.c2.liara.space/a0746ff2-5c47-43b0-af5f-f88cdc7f4442.jpg" width=640px>
</p>

## What do you need for usig this project?

- An account on [Modal](https://modal.com) with sufficient funds.
- [Ostris' AI Toolkit](https://github.com/ostris/ai-toolkit)
- A [HuggingFace](https://huggingface.co) token with `read` permission. 
- Setting up your HF token on modal. 
- Enough time and courage for your AI weekend project!

## How to use training configuration

Please check AI Toolkit's documentations in order to find out how to use modal trainer script. I just provided the YAML file for configuring your model. Also do not forget to update the config file with true values.

## How to use inference

Just run the code like this:

```
modal run inference.py \
--prompt "a cat" \
--width 1024 \
--height 1024 \
--lora "HF_USERNAME/MODEL_NAME" \
--filename "my_amazing_image"
``` 

## Notes

- Since [Atelier AI](https://atelierai.me) is a part of my bigger startup [Mann-E](https://mann-e.com), the model is set to `mann-e/mann-e_flux`. You easily can change it to your desired FLUX based checkpoint. 
- The training configuration uploads trained LoRA (in this case, LoKR) on HuggingFace automatically. You can use the same weight on Fal AI, Replicate, Self-Hosted Forge or anywhere you have the ability to use LoRAs. 