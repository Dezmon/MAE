FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

# The following `ARG` are mainly used to specify the versions explicitly & directly in this docker file, and not meant
# to be used as arguments for docker build (so far).

#ARG PYTORCH='2.2.0'
# (not always a valid torch version)
#ARG INTEL_TORCH_EXT='2.2.0'
# Example: `cu102`, `cu113`, etc.
#ARG CUDA='cu118'

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg git-lfs
RUN git lfs install
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -U torch==2.2.0 torchvision tensorboard wandb einops timm==0.4.12 tqdm

#ARG REF=main
#RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# During switch torch 2.2, we need to move (explicit) torch installation below but keep tf installation here.
# (otherwise we get `The runner has received a shutdown signal.` whose root cause is unknown but likely disk being full)
#RUN python3 -m pip install --no-cache-dir -U tensorflow==2.13 protobuf==3.20.3 tensorflow_text tensorflow_probability

#RUN python3 -m pip install --no-cache-dir -e ./transformers[dev,onnxruntime]

# RUN python3 -m pip uninstall -y torch torchvision torchaudio && python3 -m pip install --no-cache-dir -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TODO: Handle these in a python utility script
#RUN [ ${#PYTORCH} -gt 0 -a "$PYTORCH" != "pre" ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; echo "export VERSION='$VERSION'" >> ~/.profile
#RUN echo torch=$VERSION
# `torchvision` and `torchaudio` should be installed along with `torch`, especially for nightly build.
# Currently, let's just use their latest releases (when `torch` is installed with a release version)
# TODO: We might need to specify proper versions that work with a specific torch version (especially for past CI).
#RUN [ "$PYTORCH" != "pre" ] && python3 -m pip install --no-cache-dir -U $VERSION torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA || python3 -m pip install --no-cache-dir -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/$CUDA

#RUN python3 -m pip uninstall -y flax jax

#RUN python3 -m pip install --no-cache-dir intel_extension_for_pytorch==$INTEL_TORCH_EXT -f https://developer.intel.com/ipex-whl-stable-cpu

#RUN python3 -m pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git pytesseract
#RUN python3 -m pip install -U "itsdangerous<2.1.0"

#RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/accelerate@main#egg=accelerate

#RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/peft@main#egg=peft

# For bettertransformer
#RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/optimum@main#egg=optimum

# For video model testing
#RUN python3 -m pip install --no-cache-dir decord av==9.2.0

# For `dinat` model
#RUN python3 -m pip install --no-cache-dir 'natten<0.15.0' -f https://shi-labs.com/natten/wheels/$CUDA/

# For `nougat` tokenizer
#RUN python3 -m pip install --no-cache-dir python-Levenshtein

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
#RUN cd transformers && python3 setup.py develop
