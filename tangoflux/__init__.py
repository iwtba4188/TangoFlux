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
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from accelerate.utils import get_balanced_memory


class TangoFluxInference:

    def __init__(
        self,
        name="declare-lab/TangoFlux",
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=None,
        local_files_only=False,
        max_memory_setting=None,
    ):
        def setup_device_map_then_load(model, checkpoint):
            max_memory = get_balanced_memory(
                model,
                max_memory={0: "15GiB", 1: "15GiB"} if max_memory_setting is None else max_memory_setting,
                no_split_module_classes=[
                    "T5Block",
                    "T5LayerFF",
                    "T5LayerSelfAttention", 
                    "T5Stack",
                ],
                low_zero=False,
            )
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[
                    "T5Block",
                    "T5LayerFF",
                    "T5LayerSelfAttention", 
                    "T5Stack",
                ],
            )
            return load_checkpoint_and_dispatch(model, checkpoint=checkpoint, device_map=device_map)

        paths = snapshot_download(repo_id=name, cache_dir=cache_dir, local_files_only=local_files_only)           

        with init_empty_weights():
            self.vae = AutoencoderOobleck()
        self.vae = setup_device_map_then_load(self.vae, checkpoint="{}/vae.safetensors".format(paths))

        with open("{}/config.json".format(paths), "r") as f:
            config = json.load(f)
        with init_empty_weights():
            self.model = TangoFlux(config, cache_dir=cache_dir)
        self.model = setup_device_map_then_load(self.model, checkpoint="{}/tangoflux.safetensors".format(paths))

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
