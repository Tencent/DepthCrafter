import gc
import os
from copy import deepcopy

import gradio as gr
import numpy as np
import torch
from diffusers.training_utils import set_seed

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import read_video_frames, vis_sequence_depth, save_video
from run import DepthCrafterDemo

examples = [
    ["examples/example_01.mp4", 25, 1.2, 1024, 195],
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
                    <a style='font-size:18px;color: #000000'>If you find DepthCrafter useful, please help star the </a>\
                    <a style='font-size:18px;color: #FF5DB0' href='https://github.com/wbhu/DepthCrafter'>[Github Repo]</a>\
                    <a style='font-size:18px;color: #000000'>, which is important to Open-Source projects. Thanks!</a>\
                        <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2409.02095'> [ArXiv] </a>\
                        <a style='font-size:18px;color: #000000' href='https://depthcrafter.github.io/'> [Project Page] </a> </div>
            """
        )
        # demo
        depthcrafter_demo = DepthCrafterDemo(
            unet_path="tencent/DepthCrafter",
            pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
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
                            value=25,
                            step=1,
                        )
                        guidance_scale = gr.Slider(
                            label="cfg scale",
                            minimum=1.0,
                            maximum=1.2,
                            value=1.2,
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
                            minimum=1,
                            maximum=280,
                            value=195,
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
            ],
            outputs=[output_video_1, output_video_2],
            fn=depthcrafter_demo.run,
            cache_examples=False,
        )

        generate_btn.click(
            fn=depthcrafter_demo.run,
            inputs=[
                input_video,
                num_denoising_steps,
                guidance_scale,
                max_res,
                process_length,
            ],
            outputs=[output_video_1, output_video_2],
        )

    return depthcrafter_iface


demo = construct_demo()

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=80, debug=True)
