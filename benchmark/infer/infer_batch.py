import os
import multiprocessing as mp
import csv
import argparse


def process_video(video_path, gpu_id, save_folder, args):
    os.system(f'sh ./benchmark/demo.sh {video_path} {gpu_id} {int(args.process_length)} {args.saved_root} {save_folder} {args.overlap} {args.dataset}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--meta_path', type=str)
    parser.add_argument('--saved_dataset_folder', type=str)
    parser.add_argument('--saved_root', type=str, default="./output")
    parser.add_argument('--input_rgb_root', type=str)

    parser.add_argument('--process_length', type=int, default=110)
    parser.add_argument('--gpus', type=str, default="0,1,2,3")
    
    parser.add_argument('--overlap', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="open")
    
    args = parser.parse_args()
    gpus = args.gpus.strip().split(',')

    with open(args.meta_path, mode="r", encoding="utf-8") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        test_samples = list(csv_reader)
    batch_size = len(gpus)
    video_batches = [test_samples[i:i+batch_size] for i in range(0, len(test_samples), batch_size)]
    print("gpus+++: ", gpus)

    processes = []
    for video_batch in video_batches:
        for i, sample in enumerate(video_batch):
            video_path = os.path.join(args.input_rgb_root, sample["filepath_left"])
            save_folder = os.path.join(args.saved_dataset_folder, os.path.dirname(sample["filepath_left"]))
            gpu_id = gpus[i % len(gpus)]
            p = mp.Process(target=process_video, args=(video_path, gpu_id, save_folder, args))
            p.start()
            processes.append(p)
        
        for p in processes: 
            p.join()