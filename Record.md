## Env
```shell
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install av==12.3.0 # AttributeError: 'ImportError' object has no attribute 'open'
pip install numpy==1.24.4 # RuntimeError: Could not infer dtype of numpy.uint8
```
## Finetuning
you need revise:
- ```/home/v-wangxiaofa/lzl/simpler_gcr_rdt_1B_set_1/data/libero_vla_dataset.py```: LEROBOT_DIR
- ```/home/v-wangxiaofa/lzl/simpler_gcr_rdt_1B_set_1/train/dataset.py```: line 18 import
