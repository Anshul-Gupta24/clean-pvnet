# install torch 1.1 built from cuda 9.0
pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu90/stable

pip install Cython==0.28.2
apt-get install libglfw3-dev libglfw3
pip install -r requirements.txt
pip install transforms3d

ROOT=/pfs/rdi/cei/algo_train/gupansh/clean-pvnet
cd $ROOT/lib/csrc
export CUDA_HOME="/usr/local/cuda-9.0"
cd dcn_v2
python setup.py build_ext --inplace
cd ../ransac_voting
python setup.py build_ext --inplace
cd ../nn
python setup.py build_ext --inplace
cd ../fps
python setup.py build_ext --inplace

# If you want to use the uncertainty-driven PnP
cd ../uncertainty_pnp
apt-get install libgoogle-glog-dev
apt-get install libsuitesparse-dev
apt-get install libatlas-base-dev
python setup.py build_ext --inplace
