# Modal FLUX LoRA toolkit

Have you ever wondered why [Fal AI](https://fal.ai)'s LoRA trainer is so fast? I recently was working on [Atelier AI](https://atelierai.me) which is an equivalent of PhotoAI for Iranian/Persian speaking users and I needed to make the train procedure fast. I first wanted to use Fal AI, but I realized that $2 per train is too much for me. In the other hand, I could use [Modal](https://modal.com) in order to do my training and stuff. 

So I did a lot of research personally to find out these tips about making train faster: 

- Using a powerful GPU can help, but it's not everything. 
- We only can use one or two layers of LoRA in order to fit our concepts in it (if you see [config_example.yaml](./config_example.yaml) file, you will notice I've used 4.)
- If we disable sampling, we can make it even faster. 
- On powerful GPU's such as _A100 80GB_ or _H100_ we won't need to worry about `lowvram` and we can disable it (and therefore make the process faster.)