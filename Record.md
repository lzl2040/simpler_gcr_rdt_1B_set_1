## Env
```shell
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install av==12.3.0 # AttributeError: 'ImportError' object has no attribute 'open'
pip install numpy==1.24.4 # RuntimeError: Could not infer dtype of numpy.uint8
```
## Finetuning

you need revise:
- finetune_datasets.json
- finetune_sample_weights.json
- dataset_control_freq.json
- dataset_stat.json
- run data/compute_dataset_stat_hdf5.py
- ```/home/v-wangxiaofa/lzl/simpler_gcr_rdt_1B_set_1/data/libero_vla_dataset.py```: LEROBOT_DIR
- ```/home/v-wangxiaofa/lzl/simpler_gcr_rdt_1B_set_1/train/dataset.py```: line 18 import


## Results
### LIBERO Goal
train on libero_goal, only primary images:
- 60K, step=10: 28.6%
- 120K, step=15: 27.2%
- 120K, step=10, 29.2%
- 120K, step=5, 25%
- 150K, step=10, 29.2%
- 190K, step=15, 29.8%

train on libero_all, both wrist and primary images:
- 110K, step=15, wo wrist: 22.8%
- 110K, step=10, wo wrist: 20.8%