## When Cuda fails, we need to reinstlal

We first remove everything from the current machine
`sudo apt-get --purge remove nvidia-*`

`sudo apt-get --purge remove cuda-*`

Originally for MAT2021, This was created using tensorflow 1.15, with the cuda compatibility list being this

and the tensorflow docker compatibility list being this
