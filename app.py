import logging
import gradio as gr
import spaces

from depthcrafter.inference import DepthCrafterInference

logging.basicConfig(level=logging.INFO)

examples = [
    ["examples/example_01.mp4", 5, 1.0, 1024, -1, -1],
    ["examples/example_02.mp4", 5, 1.0, 1024, -1, -1],
    ["examples/example_03.mp4", 5, 1.0, 1024, -1, -1],
    ["examples/example_04.mp4", 5, 1.0, 1024, -1, -1],
    ["examples/example_05.mp4", 5, 1.0, 1024, -1, -1],
]

# Initialize the inference class globally
depthcrafter_inference = DepthCrafterInference(
    unet_path="tencent/DepthCrafter",
    pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
    cpu_offload="model",
)


@spaces.GPU(duration=120)
def infer_depth(
    video: str,
    num_denoising_steps: int,
    guidance_scale: float,
    max_res: int = 1024,
    process_length: int = -1,
    target_fps: int = -1,
    save_folder: str = "./demo_output",
    window_size: int = 110,
    overlap: int = 25,
    seed: int = 42,
    track_time: bool = True,
    save_npz: bool = False,
):
    """
    Gradio inference function.
    """
    res_paths = depthcrafter_inference.infer(
        video_path=video,
        num_denoising_steps=num_denoising_steps,
        guidance_scale=guidance_scale,
        save_folder=save_folder,
        window_size=window_size,
        process_length=process_length,
        overlap=overlap,
        max_res=max_res,
        target_fps=target_fps,
        seed=seed,
        track_time=track_time,
        save_npz=save_npz,
    )

    depthcrafter_inference.clear_cache()

    # Returning input and vis as per original code behavior
    return res_paths[:2]


def construct_demo():
    with gr.Blocks(analytics_enabled=False) as depthcrafter_iface:
        gr.Markdown(
            """
            <div align='center'>
            <h1> DepthCrafter: Generating Consistent Long Depth Sequences
            for Open-world Videos </h1>
            <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>
                <a href='https://wbhu.github.io'>Wenbo Hu</a>,
                <a href='https://scholar.google.com/citations?user=qgdesEcAAAAJ&hl=en'>
                Xiangjun Gao</a>,
                <a href='https://xiaoyu258.github.io/'>Xiaoyu Li</a>,
                <a href='https://scholar.google.com/citations?user=tZ3dS3MAAAAJ&hl=en'>
                Sijie Zhao</a>,
                <a href='https://vinthony.github.io/academic'> Xiaodong Cun</a>,
                <a href='https://yzhang2016.github.io'>Yong Zhang</a>,
                <a href='https://home.cse.ust.hk/~quan'>Long Quan</a>,
                <a href='https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en'>
                Ying Shan</a>
            </h2>
            <a style='font-size:18px;color: #000000'>
            If you find DepthCrafter useful, please help ⭐ the </a>
            <a style='font-size:18px;color: #FF5DB0'
            href='https://github.com/Tencent/DepthCrafter'>[Github Repo]</a>
            <a style='font-size:18px;color: #000000'>
            , which is important to Open-Source projects. Thanks!</a>
            <a style='font-size:18px;color: #000000'
            href='https://arxiv.org/abs/2409.02095'> [ArXiv] </a>
            <a style='font-size:18px;color: #000000'
            href='https://depthcrafter.github.io/'> [Project Page] </a>
            </div>
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
            For time quota consideration, we set the default parameters
            to be more efficient here, with a trade-off of shorter video
            length and slightly lower quality. You may adjust the parameters
            according to our
            <a style='font-size:18px;color: #FF5DB0'
            href='https://github.com/Tencent/DepthCrafter'>[Github Repo]</a>
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
    # demo.launch(server_name="0.0.0.0", server_port=12345,
    #             debug=True, share=False)
    demo.launch(share=True)
