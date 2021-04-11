
# setup Avatarify python env

- python 3.7.10

~~~
cd Main/avatarify
python -m venv .venv
cd .venv/Scripts
activate

pip install -r requirements.txt
pip install -U scikit-image
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
~~~ 

###  Known issue.

issue
~~~
error: Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio": [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/)
~~~
→ [https://mebee.info/2020/07/18/post-13597/](https://mebee.info/2020/07/18/post-13597/)
