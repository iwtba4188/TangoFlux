from diffusers import AutoencoderOobleck
import torch
from transformers import T5EncoderModel, T5TokenizerFast
from diffusers import FluxTransformer2DModel
from torch import nn
from typing import List
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
import copy
import torch.nn.functional as F
import numpy as np
from tangoflux.model import TangoFlux
from huggingface_hub import snapshot_download
from tqdm import tqdm
from typing import Optional, Union, List
from datasets import load_dataset, Audio
from math import pi
import json
import inspect
import yaml
from safetensors.torch import load_file
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


class TangoFluxInference:

    def __init__(
        self,
        name="declare-lab/TangoFlux",
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=None,
        local_files_only=False,
        device_map="auto",
    ):

        paths = snapshot_download(repo_id=name, cache_dir=cache_dir, local_files_only=local_files_only)           

        with init_empty_weights():
            self.vae = AutoencoderOobleck()
        self.vae = load_checkpoint_and_dispatch(
            self.vae, checkpoint="{}/vae.safetensors".format(paths), device_map=device_map
        )

        with open("{}/config.json".format(paths), "r") as f:
            config = json.load(f)

        with init_empty_weights():
            self.model = TangoFlux(config, cache_dir=cache_dir)
        self.model = load_checkpoint_and_dispatch(
            self.model, checkpoint="{}/tangoflux.safetensors".format(paths), device_map=device_map
        )

    def generate(self, prompt, steps=25, duration=10, guidance_scale=4.5):

        with torch.no_grad():
            latents = self.model.inference_flow(
                prompt,
                duration=duration,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
            )

            wave = self.vae.decode(latents.transpose(2, 1)).sample.cpu()[0]
        waveform_end = int(duration * self.vae.config.sampling_rate)
        wave = wave[:, :waveform_end]
        return wave
