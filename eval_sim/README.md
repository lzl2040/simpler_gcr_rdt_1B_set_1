# RDT Eval
## LIBERO
```shell
conda create -n rdt_libero python=3.10
conda activate rdt_libero
pip install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu121

# Install packaging
pip install packaging==24.0

# Install other prequisites
pip install -r requirements.txt

cd LIBERO
pip install -r requirements.txt
pip install -e .
cd RDT
pip install transformers==4.48.1
pip install -r requirements.txt
pip install tyro
```