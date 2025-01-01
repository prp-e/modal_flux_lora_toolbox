import time
from io import BytesIO
from pathlib import Path
import modal
import base64
from uuid import uuid4

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"

flux_image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "invisible_watermark",
        "transformers",
        "huggingface_hub[hf_transfer]",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "torc",
        f"git+https://github.com/huggingface/diffusers.git",
        "numpy",
        "protobuf",
        "peft"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

flux_image = flux_image.env(
    {"TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache"}
).env({"TORCHINDUCTOR_FX_GRAPH_CACHE": "1"})

app = modal.App("example-flux-lora", image=flux_image)

with flux_image.imports():
    import torch
    from diffusers import  DiffusionPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderTiny, AutoencoderKL

MINUTES = 60
VARIANT = "schnell"
NUM_INFERENCE_STEPS = 16

@app.cls(
    gpu="H100",
    container_idle_timeout= 3 * MINUTES,
    timeout=60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": modal.Volume.from_name(
            "triton-cache", create_if_missing=True
        ),
        "/root/.inductor-cache": modal.Volume.from_name(
            "inductor-cache", create_if_missing=True
        ),
    },
)
class Model:
    compile: int = modal.parameter(default=0)

    def setup_model(self):
        from huggingface_hub import snapshot_download
        from transformers.utils import move_cache

        #can be replaced with flux-dev repo

        snapshot_download(f"mann-e/mann-e_flux")

        move_cache()

        taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16)
        pipe = DiffusionPipeline.from_pretrained("mann-e/mann-e_flux", torch_dtype=torch.bfloat16, vae=taef1)


        return pipe

    @modal.build()
    def build(self):
        self.setup_model()

    @modal.enter()
    def enter(self):
        pipe = self.setup_model()
        pipe.to("cuda")
        self.pipe = optimize(pipe, compile=bool(self.compile))

    @modal.method()
    def inference(self, prompt: str, width: int, height: int, lora: str) -> bytes:
        print("🎨 generating image...")

        pipeline = self.pipe

        pipeline.load_lora_weights(lora)
        pipeline.fuse_lora(lora_scale = 1.0)

        out = pipeline(
            f"{prompt}, atelier_sks_768",
            output_type="pil",
            num_inference_steps=NUM_INFERENCE_STEPS,
            width=width,
            height=height,
            guidance_scale=3.5
        ).images[0]

        
        del pipeline

        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()

#@app.function()
#@modal.web_endpoint(method="POST")
@app.local_entrypoint()
def main(
    prompt: str,
    width: int, 
    height: int,
    lora: str,
    filename: str,
    #request: dict,
    twice: bool = False,
    compile: bool = False,
):
    # prompt = request['prompt']
    # width = request['width']
    # height = request['height']
    # lora = request['lora']
    
    t0 = time.time()
    image_bytes = Model(compile=compile).inference.remote(prompt, width, height, lora)
    print(f"🎨 first inference latency: {time.time() - t0:.2f} seconds")

    if twice:
        t0 = time.time()
        image_bytes = Model(compile=compile).inference.remote(prompt, width, height, lora)
        print(f"🎨 second inference latency: {time.time() - t0:.2f} seconds")

    # output_path = Path("/tmp") / "flux" / "output.jpg"
    # output_path.parent.mkdir(exist_ok=True, parents=True)
    # print(f"🎨 saving output to {output_path}")
    # output_path.write_bytes(image_bytes)

    #return {"image" : base64.b64encode(image_bytes)}

    output_path = Path(".") / f"{filename}.jpg"
    output_path.write_bytes(image_bytes)

def optimize(pipe, compile=True):
    return pipe