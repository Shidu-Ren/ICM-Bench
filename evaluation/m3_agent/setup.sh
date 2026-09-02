# Modified by the ICM-Bench authors in 2026 for benchmark integration.
pip install -r requirements.txt

pip install setuptools_scm torchdiffeq resampy x_transformers
pip install accelerate==0.34.2 # https://github.com/huggingface/trl/issues/2377
pip install ninja
apt-get -y install ninja-build

pip install flash-attn==2.6.3 --no-build-isolation
# CUDA 12.4 stable wheels
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install peft==0.15.2
pip install moviepy==2.1.2
pip install jupyter
pip install httpx==0.23.0
pip install pydub==0.25.1

pip install trl==0.16.0 # other versions may have problems
apt-get -y install ffmpeg # load audio in video(mp4)

pip install onnxruntime-gpu==1.22.1
pip install insightface==0.7.3
pip install hdbscan==0.8.40
