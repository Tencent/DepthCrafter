"""Record3D visualizer
"""

import time
from decord import VideoReader, cpu

import numpy as np
import tyro
import viser
import viser.extras
import viser.transforms as tf
from tqdm.auto import tqdm


def main(
    data_path: str,
    vid_name: str,
    downsample_factor: int = 8,
    max_frames: int = 100,
    share: bool = False,
    point_size=0.01,
) -> None:

    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print("Loading frames!")
    dis_path = data_path + "/" + vid_name + ".npz"
    vid_path = data_path + "/" + vid_name + "_input.mp4"

    disp_map = np.load(dis_path)["depth"][:, :, :]
    T = disp_map.shape[0]
    H = disp_map.shape[1]
    W = disp_map.shape[2]

    disp_max = disp_map.max()
    disp_min = disp_map.min()
    disp_map = (disp_map - disp_min) / (disp_max - disp_min)

    vr = VideoReader(vid_path, ctx=cpu(0))
    vid = vr[:].asnumpy()[:, 0:H, 0:W]
    fps = vr.get_avg_fps()
    num_frames = min(max_frames, T)

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=fps
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(np.array([0.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    for i in tqdm(range(num_frames)):

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        position_image = np.where(np.zeros([H, W]) == 0)
        v = np.array(position_image[0])
        u = np.array(position_image[1])
        d = disp_map[i, v, u]

        zc = 1.0 / (d + 0.1)
        # zc = 1.0 / (d + 1e-8)

        xc = zc * (u - (W / 2.0)) / (W / 2.0)
        yc = zc * (v - (H / 2.0)) / (H / 2.0)

        zc -= 4  # disp_max * 0.2

        points = np.stack((xc, yc, zc), axis=1)
        colors = vid[i, v, u]

        points = points[::downsample_factor]
        colors = colors[::downsample_factor]

        # Place the point cloud in the frame.
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=points,
            colors=colors,
            point_size=point_size,  # 0.007,
            point_shape="rounded",
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(
        main(
            # dir path of saved rgb.mp4 and disp.npz, modify it to your own dir
            data_path="./demo_output",
            # sample name, modify it to your own sample name
            vid_name="example_01",
            # downsample factor of dense pcd
            downsample_factor=8,
            # point cloud size
            point_size=0.007,
        )
    )
