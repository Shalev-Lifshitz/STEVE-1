import os
import sys

import cv2
import torch
from tqdm import tqdm
import argparse

from steve1.config import PRIOR_INFO, DEVICE
from steve1.data.text_alignment.vae import load_vae_model
from steve1.run_agent.paper_prompts import load_text_prompt_embeds, load_visual_prompt_embeds
from steve1.run_agent.programmatic_eval import ProgrammaticEvaluator
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env, load_mineclip_wconfig
from steve1.utils.video_utils import save_frames_as_video

FPS = 30


def run_agent(prompt_embed, gameplay_length, save_video_filepath,
              in_model, in_weights, seed, cond_scale):
    assert cond_scale is not None
    agent, mineclip, env = load_mineclip_agent_env(in_model, in_weights, seed, cond_scale)

    # Make sure seed is set if specified
    obs = env.reset()
    if seed is not None:
        env.seed(seed)

    # Setup
    gameplay_frames = []
    prog_evaluator = ProgrammaticEvaluator(obs)

    # Run agent in MineRL env
    for _ in tqdm(range(gameplay_length)):
        with torch.cuda.amp.autocast():
            minerl_action = agent.get_action(obs, prompt_embed)

        obs, _, _, _ = env.step(minerl_action)
        frame = obs['pov']
        frame = cv2.resize(frame, (128, 128))
        gameplay_frames.append(frame)

        prog_evaluator.update(obs)

    # Make the eval episode dir and save it
    os.makedirs(os.path.dirname(save_video_filepath), exist_ok=True)
    save_frames_as_video(gameplay_frames, save_video_filepath, FPS, to_bgr=True)

    # Print the programmatic eval task results at the end of the gameplay
    prog_evaluator.print_results()


def generate_text_prompt_videos(prompt_embeds, in_model, in_weights, cond_scale, gameplay_length, save_dirpath):
    for name, prompt_embed in prompt_embeds.items():
        print(f'\nGenerating video for text prompt with name: {name}')
        save_video_filepath = os.path.join(save_dirpath, f'\'{name}\' - Text Prompt.mp4')
        if not os.path.exists(save_video_filepath):
            run_agent(prompt_embed, gameplay_length, save_video_filepath,
                      in_model, in_weights, None, cond_scale)
        else:
            print(f'Video already exists at {save_video_filepath}, skipping...')


def generate_visual_prompt_videos(prompt_embeds, in_model, in_weights, cond_scale, gameplay_length, save_dirpath):
    for name, prompt_embed in prompt_embeds.items():
        print(f'\nGenerating video for visual prompt with name: {name}')
        save_video_filepath = os.path.join(save_dirpath, f'{name} - Visual Prompt.mp4')
        if not os.path.exists(save_video_filepath):
            run_agent(prompt_embed, gameplay_length, save_video_filepath,
                      in_model, in_weights, None, cond_scale)
        else:
            print(f'Video already exists at {save_video_filepath}, skipping...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model', type=str, default='data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='data/weights/steve1/steve1.weights')
    parser.add_argument('--prior_weights', type=str, default='data/weights/steve1/steve1_prior.pt')
    parser.add_argument('--text_cond_scale', type=float, default=6.0)
    parser.add_argument('--visual_cond_scale', type=float, default=7.0)
    parser.add_argument('--gameplay_length', type=int, default=1000)
    parser.add_argument('--save_dirpath', type=str, default='data/generated_videos/')
    parser.add_argument('--custom_text_prompt', type=str, default=None)
    args = parser.parse_args()

    if args.custom_text_prompt is not None:
        # Generate a video for the text prompt
        mineclip = load_mineclip_wconfig()
        prior = load_vae_model(PRIOR_INFO)
        prompt_embed = get_prior_embed(args.custom_text_prompt, mineclip, prior, DEVICE)
        custom_prompt_embeds = {args.custom_text_prompt: prompt_embed}
        generate_text_prompt_videos(custom_prompt_embeds, args.in_model, args.in_weights, args.text_cond_scale,
                                    args.gameplay_length, args.save_dirpath)
    else:
        # Generate videos for the text and visual prompts used in the paper
        text_prompt_embeds = load_text_prompt_embeds()
        visual_prompt_embeds = load_visual_prompt_embeds()
        generate_text_prompt_videos(text_prompt_embeds, args.in_model, args.in_weights, args.text_cond_scale,
                                    args.gameplay_length, args.save_dirpath)
        generate_visual_prompt_videos(visual_prompt_embeds, args.in_model, args.in_weights, args.visual_cond_scale,
                                      args.gameplay_length, args.save_dirpath)
        sys.exit(0)
