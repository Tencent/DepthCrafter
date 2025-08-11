import gc
import os

import numpy as np
import spaces
import gradio as gr
import torch
from diffusers.training_utils import set_seed

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

import uuid
import random
from huggingface_hub import hf_hub_download

from depthcrafter.utils import read_video_frames, vis_sequence_depth, save_video

examples = [
    ["examples/example_01.mp4", 5, 1.0, 1024, -1, -1],
    ["examples/example_02.mp4", 5, 1.0, 1024, -1, -1],
    ["examples/example_03.mp4", 5, 1.0, 1024, -1, -1],
    ["examples/example_04.mp4", 5, 1.0, 1024, -1, -1],
    ["examples/example_05.mp4", 5, 1.0, 1024, -1, -1],
]


# Detect device - use CPU since MPS doesn't support Conv3D
device = "cuda" if torch.cuda.is_available() else "cpu"

unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
    "tencent/DepthCrafter",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32,
)
pipe = DepthCrafterPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    unet=unet,
    torch_dtype=torch.float32,
)
pipe.to(device)


@spaces.GPU(duration=120)
def infer_depth(
    video: str,
    num_denoising_steps: int,
    guidance_scale: float,
    max_res: int = 1024,
    process_length: int = -1,
    target_fps: int = -1,
    #
    save_folder: str = "./demo_output",
    window_size: int = 110,
    overlap: int = 25,
    seed: int = 42,
    track_time: bool = True,
    save_npz: bool = False,
):
    set_seed(seed)
    # Only enable xformers for CUDA devices
    if torch.cuda.is_available():
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Xformers not enabled: {e}")

    frames, target_fps = read_video_frames(video, process_length, target_fps, max_res)

    # inference the depth map using the DepthCrafter pipeline
    with torch.inference_mode():
        res = pipe(
            frames,
            height=frames.shape[1],
            width=frames.shape[2],
            output_type="np",
            guidance_scale=guidance_scale,
            num_inference_steps=num_denoising_steps,
            window_size=window_size,
            overlap=overlap,
            track_time=track_time,
        ).frames[0]
    # convert the three-channel output to a single channel depth map
    res = res.sum(-1) / res.shape[-1]
    # normalize the depth map to [0, 1] across the whole video
    res = (res - res.min()) / (res.max() - res.min())
    # visualize the depth map and save the results
    vis = vis_sequence_depth(res)
    # save the depth map and visualization with the target FPS
    save_path = os.path.join(save_folder, os.path.splitext(os.path.basename(video))[0])
    print(f"==> saving results to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_npz:
        np.savez_compressed(save_path + ".npz", depth=res)
    save_video(res, save_path + "_depth.mp4", fps=target_fps)
    save_video(vis, save_path + "_vis.mp4", fps=target_fps)
    save_video(frames, save_path + "_input.mp4", fps=target_fps)

    # clear the cache for the next video
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [
        save_path + "_input.mp4",
        save_path + "_vis.mp4",
        # save_path + "_depth.mp4",
    ]


def construct_demo():
    with gr.Blocks(analytics_enabled=False) as depthcrafter_iface:
        gr.Markdown(
            """
            <div align='center'> <h1> DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos </span> </h1> \
                        <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                        <a href='https://wbhu.github.io'>Wenbo Hu</a>, \
                        <a href='https://scholar.google.com/citations?user=qgdesEcAAAAJ&hl=en'>Xiangjun Gao</a>, \
                        <a href='https://xiaoyu258.github.io/'>Xiaoyu Li</a>, \
                        <a href='https://scholar.google.com/citations?user=tZ3dS3MAAAAJ&hl=en'>Sijie Zhao</a>, \
                        <a href='https://vinthony.github.io/academic'> Xiaodong Cun</a>, \
                        <a href='https://yzhang2016.github.io'>Yong Zhang</a>, \
                        <a href='https://home.cse.ust.hk/~quan'>Long Quan</a>, \
                        <a href='https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en'>Ying Shan</a>\
                    </h2> \
                    <a style='font-size:18px;color: #000000'>If you find DepthCrafter useful, please help ⭐ the </a>\
                    <a style='font-size:18px;color: #FF5DB0' href='https://github.com/Tencent/DepthCrafter'>[Github Repo]</a>\
                    <a style='font-size:18px;color: #000000'>, which is important to Open-Source projects. Thanks!</a>\
                        <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2409.02095'> [ArXiv] </a>\
                        <a style='font-size:18px;color: #000000' href='https://depthcrafter.github.io/'> [Project Page] </a> </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_video = gr.Video(label="Input Video")

            # with gr.Tab(label="Output"):
            with gr.Column(scale=2):
                with gr.Row(equal_height=True):
                    output_video_1 = gr.Video(
                        label="Preprocessed video",
                        interactive=False,
                        autoplay=True,
                        loop=True,
                        show_share_button=True,
                        scale=5,
                    )
                    output_video_2 = gr.Video(
                        label="Generated Depth Video",
                        interactive=False,
                        autoplay=True,
                        loop=True,
                        show_share_button=True,
                        scale=5,
                    )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Row(equal_height=False):
                    with gr.Accordion("Advanced Settings", open=False):
                        num_denoising_steps = gr.Slider(
                            label="num denoising steps",
                            minimum=1,
                            maximum=25,
                            value=5,
                            step=1,
                        )
                        guidance_scale = gr.Slider(
                            label="cfg scale",
                            minimum=1.0,
                            maximum=1.2,
                            value=1.0,
                            step=0.1,
                        )
                        max_res = gr.Slider(
                            label="max resolution",
                            minimum=512,
                            maximum=2048,
                            value=1024,
                            step=64,
                        )
                        process_length = gr.Slider(
                            label="process length",
                            minimum=-1,
                            maximum=280,
                            value=60,
                            step=1,
                        )
                        process_target_fps = gr.Slider(
                            label="target FPS",
                            minimum=-1,
                            maximum=30,
                            value=15,
                            step=1,
                        )
                    generate_btn = gr.Button("Generate")
            with gr.Column(scale=2):
                pass

        gr.Examples(
            examples=examples,
            inputs=[
                input_video,
                num_denoising_steps,
                guidance_scale,
                max_res,
                process_length,
                process_target_fps,
            ],
            outputs=[output_video_1, output_video_2],
            fn=infer_depth,
            cache_examples="lazy",
        )
        gr.Markdown(
            """
            <span style='font-size:18px;color: #E7CCCC'>Note: 
            For time quota consideration, we set the default parameters to be more efficient here,
            with a trade-off of shorter video length and slightly lower quality.
            You may adjust the parameters according to our 
            <a style='font-size:18px;color: #FF5DB0' href='https://github.com/Tencent/DepthCrafter'>[Github Repo]</a>
             for better results if you have enough time quota.
            </span>
            """
        )

        generate_btn.click(
            fn=infer_depth,
            inputs=[
                input_video,
                num_denoising_steps,
                guidance_scale,
                max_res,
                process_length,
                process_target_fps,
            ],
            outputs=[output_video_1, output_video_2],
        )

    return depthcrafter_iface


if __name__ == "__main__":
    demo = construct_demo()
    demo.queue()
    # demo.launch(server_name="0.0.0.0", server_port=12345, debug=True, share=False)
    demo.launch(share=True)
