# /bin/bash!

gdown --id 1_U_KwDK2OdEq02jW-VDsbtHLhgKqTnXT -O data/vidchapters/tagged_video_ids.pkl

# Train / Val / Test split Video IDs
gdown --id 1kW52N-KULXFdGfjnKK8gz90qdqEGxa6h -O data/vidchapters/train_video_ids.json
gdown --id 1tH5WCHfVxK0H8LcQm07ECRLuR47Iu6n4 -O data/vidchapters/val_video_ids.json
gdown --id 1hOt3_A6Htme6IMj8AXTRkRDfYDju78bR -O data/vidchapters/test_video_ids.json

# Annotations
# gdown --id 1_U_KwDK2OdEq02jW-VDsbtHLhgKqTnXT -O data/vidchapters/annotations.json

python download_youtube_videos.py \
    --data_base_dir /data/user_data/jamesdin/data/vidchapters \
    --annotation_json_path /home/jamesdin/James/ThinkingWithVideos/data/MultiTaskVideoReasoning/MTVR_Tool_CoT/vidchapters.json

python download_youtube_videos.py \
    --data_base_dir /data/user_data/jamesdin/data/vidchapters \
    --annotation_json_path /home/jamesdin/James/ThinkingWithVideos/data/MultiTaskVideoReasoning/MTVR_Tool_RL/vidchapters.json


