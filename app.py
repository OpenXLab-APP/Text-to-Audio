import os
import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
import json

from models.tta.autoencoder.autoencoder import AutoencoderKL
from models.tta.ldm.inference_utils.vocoder import Generator
from models.tta.ldm.audioldm import AudioLDM
from transformers import T5EncoderModel, AutoTokenizer
from diffusers import PNDMScheduler

import matplotlib.pyplot as plt
from scipy.io.wavfile import write

from utils.util import load_config
import gradio as gr

from openxlab.model import download
download(model_repo='Amphion/Text-to-Audio', model_name='g_01250000', output='ckpts/tta/hifigan_checkpoints')
download(model_repo='Amphion/Text-to-Audio', model_name='step-0570000_loss-0.2521', output='ckpts/tta/audioldm_debug_latent_size_4_5_39/checkpoints')
download(model_repo='Amphion/Text-to-Audio', model_name='step-0445000_loss-0.3306', output='ckpts/tta/autoencoder_kl_debug/checkpoints')
download(model_repo='Amphion/Text-to-Audio', model_name='pytorch_model', output='ckpts/tta/text_encoder')


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_autoencoderkl(cfg, device):
    autoencoderkl = AutoencoderKL(cfg.model.autoencoderkl)
    autoencoder_path = cfg.model.autoencoder_path
    checkpoint = torch.load(autoencoder_path, map_location="cpu")
    autoencoderkl.load_state_dict(checkpoint["model"])
    autoencoderkl = autoencoderkl.to(device=device)
    autoencoderkl.requires_grad_(requires_grad=False)
    autoencoderkl.eval()
    return autoencoderkl


def build_textencoder(device):
    try:
        tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)
        text_encoder = T5EncoderModel.from_pretrained("t5-base")
    except:
        tokenizer = AutoTokenizer.from_pretrained("ckpts/tta/tokenizer")
        text_encoder = T5EncoderModel.from_pretrained("ckpts/tta/text_encoder")
    text_encoder = text_encoder.to(device=device)
    text_encoder.requires_grad_(requires_grad=False)
    text_encoder.eval()
    return tokenizer, text_encoder


def build_vocoder(device):
    config_file = os.path.join("ckpts/tta/hifigan_checkpoints/config.json")
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    vocoder = Generator(h).to(device)
    checkpoint_dict = torch.load(
        "ckpts/tta/hifigan_checkpoints/g_01250000", map_location=device
    )
    vocoder.load_state_dict(checkpoint_dict["generator"])
    return vocoder


def build_model(cfg):
    model = AudioLDM(cfg.model.audioldm)
    return model


def get_text_embedding(text, tokenizer, text_encoder, device):
    prompt = [text]

    text_input = tokenizer(
        prompt,
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding="do_not_pad",
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * 1, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return text_embeddings


def tta_inference(
    text,
    guidance_scale=4,
    diffusion_steps=100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["WORK_DIR"] = "./"
    cfg = load_config("egs/tta/audioldm/exp_config.json")

    autoencoderkl = build_autoencoderkl(cfg, device)
    tokenizer, text_encoder = build_textencoder(device)
    vocoder = build_vocoder(device)
    model = build_model(cfg)

    checkpoint_path = "ckpts/tta/audioldm_debug_latent_size_4_5_39/checkpoints/step-0570000_loss-0.2521.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    text_embeddings = get_text_embedding(text, tokenizer, text_encoder, device)

    num_steps = diffusion_steps

    noise_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        skip_prk_steps=True,
        set_alpha_to_one=False,
        steps_offset=1,
        prediction_type="epsilon",
    )

    noise_scheduler.set_timesteps(num_steps)

    latents = torch.randn(
        (
            1,
            cfg.model.autoencoderkl.z_channels,
            80 // (2 ** (len(cfg.model.autoencoderkl.ch_mult) - 1)),
            624 // (2 ** (len(cfg.model.autoencoderkl.ch_mult) - 1)),
        )
    ).to(device)

    model.eval()
    for t in tqdm(noise_scheduler.timesteps):
        t = t.to(device)

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = noise_scheduler.scale_model_input(
            latent_model_input, timestep=t
        )
        # print(latent_model_input.shape)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = model(
                latent_model_input, torch.cat([t.unsqueeze(0)] * 2), text_embeddings
            )

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        print(guidance_scale)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        # print(latents.shape)

    latents_out = latents

    with torch.no_grad():
        mel_out = autoencoderkl.decode(latents_out)

    melspec = mel_out[0, 0].cpu().detach().numpy()

    vocoder.eval()
    vocoder.remove_weight_norm()

    with torch.no_grad():
        melspec = np.expand_dims(melspec, 0)
        melspec = torch.FloatTensor(melspec).to(device)

        y = vocoder(melspec)
        audio = y.squeeze()
        audio = audio * 32768.0
        audio = audio.cpu().numpy().astype("int16")

    os.makedirs("result", exist_ok=True)
    write(os.path.join("result", text + ".wav"), 16000, audio)

    return os.path.join("result", text + ".wav")


demo_inputs = [
    gr.Textbox(
        value="birds singing and a man whistling",
        label="Text prompt you want to generate",
        type="text",
    ),
    gr.Slider(
        1,
        10,
        value=4,
        step=1,
        label="Classifier free guidance",
    ),
    gr.Slider(
        50,
        1000,
        value=100,
        step=1,
        label="Diffusion Inference Steps",
        info="As the step number increases, the synthesis quality will be better while the inference speed will be lower",
    ),
]

demo_outputs = gr.Audio(label="")

demo = gr.Interface(
    fn=tta_inference,
    inputs=demo_inputs,
    outputs=demo_outputs,
    title="Amphion Text to Audio",
)

if __name__ == "__main__":
    demo.launch()
