import os, dnnlib, torch, legacy
import numpy as np
import PIL.Image
# モデルのロード
device = torch.device('cuda')
with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)
# 画像生成の関数
def generate_images_from_ws_array(dlatents):
    "dlatentsはnp.arrayで，[steps, 18, 512]"
    dlatents= torch.tensor(dlatents, device=device)
    imgs = G.synthesis(dlatents, noise_mode='const') # 'none'にすると，noiseがなくなりつやっつやになる．
    imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return imgs
# 生成された画像の保存
for idx, img in enumerate(imgs):
    img = PIL.Image.fromarray(img.cpu().numpy(), 'RGB')
    img.save(f'{idx:02d}.png'
