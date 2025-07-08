#!/usr/bin/env bash

for images_dir in images-d-*; do
    if [ -d "$images_dir" ]; then
        echo "Processing $images_dir"
        basefoldername=$(basename "$images_dir")
        ffmpeg -pattern_type glob -r 24 -i "${images_dir}/frame_*.png" -c:v libx264 -pix_fmt yuv420p  "${basefoldername}.mp4"
        echo "Done"
    else
        echo "Skipping $images_dir, not a directory."
    fi
done