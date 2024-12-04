#!/bin/bash

root=$(pwd)
source "$(conda info --base)/etc/profile.d/conda.sh"


if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <video_path> <fps>"
    exit 1
fi

# Read input parameters
video_path="$1"
fps="$2"

conda activate amego
# Extract the video filename without the extension for output directory naming
video_name=$(basename "$video_path" .mp4)
output_dir="${root}/${video_name}/rgb_frames"

# Create output directory
mkdir -p "$output_dir"

# Run ffmpeg to extract frames
ffmpeg -i "$video_path" -vf "fps=${fps},scale=456:256" "${output_dir}/frame_%010d.jpg"

# Inform the user of the result
if [ $? -eq 0 ]; then
    echo "Frames extracted successfully to '${output_dir}/'!"
else
    echo "An error occurred during frame extraction."
fi

python -m tools.generate_flowformer_flow --root . --v_id ${video_name} --dset video --models_root submodules/flowformer/models --model sintel --video_fps ${fps} 

conda activate handobj
python -m tools.extract_bboxes --image_dir ${root}/${video_name}/rgb_frames --cuda --mGPUs --checksession 1 --checkepoch 8 --checkpoint 132028 --bs 32 --detections_pb ${video_name}.pb2
mkdir -p ${root}/${video_name}/hand-objects/
python -m submodules.epic-kitchens-100-hand-object-bboxes.src.scripts.convert_raw_to_releasable_detections ${video_name}.pb2 ${root}/${video_name}/hand-objects/${video_name}.pkl --frame-height 256 --frame-width 456
mv ${video_name}.pb2 ${root}/${video_name}/hand-objects/