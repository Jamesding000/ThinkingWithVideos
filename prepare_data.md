# Install environment

run setup_new.sh

# Download Train / Val / Test json files

cd data
run huggingface-cli download zhang9302002/MultiTaskVideoReasoning --repo-type dataset --local-dir ./MultiTaskVideoReasoning

# Download Raw Videos


# Extract Video Frames
run bash scripts/extract_video_frames.sh


