# /bin/bash!

data_dir=/data/user_data/jamesdin/data/vidchapters
tmp_video_dir=/scratch/job_tmp

gdown --id 1_U_KwDK2OdEq02jW-VDsbtHLhgKqTnXT -O ${data_dir}/tagged_video_ids.pkl

# Train / Val / Test split Video IDs
gdown --id 1kW52N-KULXFdGfjnKK8gz90qdqEGxa6h -O ${data_dir}/train_video_ids.json
gdown --id 1tH5WCHfVxK0H8LcQm07ECRLuR47Iu6n4 -O ${data_dir}/val_video_ids.json
gdown --id 1hOt3_A6Htme6IMj8AXTRkRDfYDju78bR -O ${data_dir}/test_video_ids.json

# Train / Val / Test split JSON files
gdown --id 1m6hj6AWlxfZJQ1Q45SsJoJ06i65pzyfH -O ${data_dir}/train.jsonl
gdown --id 1v_7vp4AlA36PZR7uXxFmHpNA6cubWJwd -O ${data_dir}/val.jsonl
gdown --id 1KPJc08l5JxxaNau4_eVtmJtvor-qa0NF -O ${data_dir}/test.jsonl

# Annotations
gdown --id 1_U_KwDK2OdEq02jW-VDsbtHLhgKqTnXT -O ${data_dir}/chapters.pkl

# Annotations
# gdown --id 1_U_KwDK2OdEq02jW-VDsbtHLhgKqTnXT -O data/vidchapters/annotations.json

data_dir=/data/user_data/jamesdin/data/vidchapters
tmp_video_dir=/scratch/job_tmp
python download_and_extract_frames.py \
    --annotation-json-path data/MultiTaskVideoReasoning/MTVR_Tool_CoT/vidchapters.json \
    --output-dir ${data_dir} \
    --tmp-video-dir tmp_video_dir \
    --fps 2 \
    --max-frames 14400 \
    --size 224

python download_and_extract_frames.py \
    --annotation-json-path data/MultiTaskVideoReasoning/MTVR_Tool_RL/vidchapters.json \
    --output-dir ${data_dir} \
    --tmp-video-dir tmp_video_dir \
    --fps 2 \
    --max-frames 14400 \
    --size 224
