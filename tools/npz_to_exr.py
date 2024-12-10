import os

import numpy as np
import OpenEXR
import Imath
from fire import Fire


def npz_to_multiframe_exr(input_npz: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    # Load the .npz file
    data = np.load(input_npz)
    depth_frames = data["depth"]  # Replace 'depth' with the correct key
    num_frames, height, width = depth_frames.shape

    # Iterate over each frame and save as a separate EXR file
    for i, frame in enumerate(depth_frames):
        output_exr = f"{output_folder}/frame_{i:04d}.exr"

        # Prepare EXR header for each frame
        header = OpenEXR.Header(width, height)
        header["channels"] = {
            "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }

        # Create EXR file and write the frame
        exr_file = OpenEXR.OutputFile(output_exr, header)
        exr_file.writePixels({"Z": frame.tobytes()})
        exr_file.close()
        print(f"Saved frame {i} to {output_exr}")


if __name__ == "__main__":
    # Specify the input .npz file and output folder
    # npz_to_multiframe_exr(r"path_to_input", r"path_to_output")
    Fire(npz_to_multiframe_exr)
