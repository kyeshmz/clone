
# setup Avatarify python env

- python 3.7.10

~~~
cd Main/avatarify
python -m venv .venv
cd .venv/Scripts
activate

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -U scikit-image
~~~ 

###  Known issue.

- pip install issue
~~~
error: Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio": [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/)
~~~
→ [https://mebee.info/2020/07/18/post-13597/](https://mebee.info/2020/07/18/post-13597/)


 - submodule 
 just copy fomm folder
 
 - model data
 -'vox-adv-cpk.pth.tar' manually put in the 'avatarify folder' 


###  first launch of avatarify
- some model data load in the first launch. 

