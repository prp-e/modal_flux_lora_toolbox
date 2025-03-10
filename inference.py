import time
from io import BytesIO
import modal

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

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
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        f"git+https://github.com/huggingface/diffusers.git",
        "numpy<2",
        "protobuf",
        "peft",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
)

flux_image = flux_image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    }
)

app = modal.App("example-flux-lora", image=flux_image)

with flux_image.imports():
    import torch
    from diffusers import DiffusionPipeline, AutoencoderTiny


MINUTES = 60
NUM_INFERENCE_STEPS = 16


@app.cls(
    gpu="H100",
    container_idle_timeout=3 * MINUTES,
    timeout=60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": modal.Volume.from_name("triton-cache", create_if_missing=True),
        "/root/.inductor-cache": modal.Volume.from_name("inductor-cache", create_if_missing=True),
    },
)
class Model:
    compile: int = modal.parameter(default=0)

    @modal.enter()
    def enter(self):
        from huggingface_hub import snapshot_download

        snapshot_download("mann-e/mann-e_flux")

        taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16)
        pipe = DiffusionPipeline.from_pretrained(
            "mann-e/mann-e_flux", torch_dtype=torch.bfloat16, vae=taef1
        )

        pipe.to("cuda")
        self.pipe = optimize(pipe, compile=bool(self.compile))

    @modal.method()
    def inference(self, prompt: str, width: int, height: int, lora: str) -> bytes:
        print("🎨 generating image...")

        self.pipe.load_lora_weights(lora)
        self.pipe.fuse_lora(lora_scale=1.0)

        out = self.pipe(
            prompt,
            output_type="pil",
            num_inference_steps=NUM_INFERENCE_STEPS,
            width=width,
            height=height,
            guidance_scale=3.5,
        ).images[0]

        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()


@app.local_entrypoint()
def main(
    prompt: str,
    width: int,
    height: int,
    lora: str,
    filename: str,
    twice: bool = False,
    compile: bool = False,
):
    t0 = time.time()
    image_bytes = Model(compile=compile).inference.remote(prompt, width, height, lora)
    print(f"🎨 first inference latency: {time.time() - t0:.2f} seconds")

    if twice:
        t0 = time.time()
        image_bytes = Model(compile=compile).inference.remote(prompt, width, height, lora)
        print(f"🎨 second inference latency: {time.time() - t0:.2f} seconds")

    with open(f"{filename}.jpg", "wb") as f:
        f.write(image_bytes)


def optimize(pipe, compile=True):
    return pipe
