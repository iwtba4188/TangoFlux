from setuptools import setup

setup(
    name="tangoflux",
    description="TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching",
    version="0.1.0",
    packages=["tangoflux"],
    install_requires=[
        "torch==2.7.0+cu128",
        "torchaudio==2.7.0+cu128",
        "torchlibrosa==0.1.0",
        "torchvision",
        "transformers==4.44.0",
        "diffusers==0.30.0",
        "accelerate==0.34.2",
        "datasets==2.21.0",
        "librosa",
        "tqdm",
        "wandb",
        "click",
        "gradio",
        "torchaudio",
    ],
    entry_points={
        "console_scripts": [
            "tangoflux=tangoflux.cli:main",
            "tangoflux-demo=tangoflux.demo:main",
        ],
    },
    find_links=[
        "https://download.pytorch.org/whl/cu128",
    ],
)
