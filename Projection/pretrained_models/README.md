wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0' -O- | sed -rn 's/._confirm=([0-9A-Za-z_]+).\_/\1\n/p')&id=1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0" -O ./psp_ffhq_encode.pt && rm -rf /tmp/cookies.txt

RuntimeError: Unable to open ./lib/dlib/shape_predictor_68_face_landmarks.dat

wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
