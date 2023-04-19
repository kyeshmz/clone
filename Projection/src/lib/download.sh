#!/bin/bash


# for stylegan2
# this wget doesnt work
wget "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl" -O "./stylegan2/networks/stylegan2-ffhq-config-f.pkl"

# for ps2p
wget  "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0" -O './ps2p/pretrained_models/psp_ffhq_encode.pt'

# for dlib
wget "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -O "./dlib/shape_predictor_68_face_landmarks.dat.bz2"
bzip2 -d  "./dlib/shape_predictor_68_face_landmarks.dat.bz2"