#!/usr/bin/env python3

# Run in Mind-with-Eyes/data directory

# Clone NExT-GQA
git clone git@github.com:doc-doc/NExT-GQA.git

# Create data folder
mkdir -p data/nextqa
cd data/nextqa

# Download CLIP features and raw videos for NExT-QA
# CLIP features
gdown --id 101W4r6ibXJE2IOr6MINbNIMC3MFiN-us -O data/nextqa/CLIPL.zip
unzip -o data/nextqa/CLIPL.zip -d data/nextqa/
rm data/nextqa/CLIPL.zip

# Raw videos
gdown --id 1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH -O data/nextqa/NExTVideo.zip
unzip -o data/nextqa/NExTVideo.zip -d data/nextqa/
rm data/nextqa/NExTVideo.zip