
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04





ENV DEBIAN_FRONTEND=noninteractive
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    wget\
    manpages-dev\
    g++\
    gcc\
    nodejs\
    libssl-dev\
    unzip\
    #cuda-11.3\
    #nvidia-cuda-toolkit-11-3\
    && rm -rf /var/lib/apt/lists/*
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda


RUN apt update
RUN apt update && apt upgrade -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.9
RUN apt install -y python3-pip
# RUN apt-get -y purge python3.8
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

RUN update-alternatives --set python /usr/bin/python3.9

RUN apt-get install python3-dev 



RUN python3.9 -m pip install --upgrade pip


RUN python3.9 -m pip install --no-cache-dir jupyter matplotlib
# RUN    pip install --no-cache-dir "intensity-normalization[ants]"
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
# Pin jedi; see https://github.com/ipython/ipython/issues/12740
RUN python3.9 -m  pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0 jedi==0.17.2
RUN python3.9 -m  pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html



#RUN    pip install pyradiomics
RUN python3.9 -m  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

RUN python3.9 -m  pip install \
    #itk \
    pytorch-ignite==0.4.8 \
    PyWavelets==1.3.0 \
    scipy==1.8.1 \
    nibabel==3.2.2 \
    pillow!=8.3.0 \  
    tensorboard==2.9.0 \
    scikit-image==0.19.2 \
    tqdm>=4.47.0 \
    lmdb==1.3.0 \
    flake8>=3.8.1 \
    flake8-bugbear \
    flake8-comprehensions \
    flake8-executable \
    flake8-pyi \
    pylint!=2.13 \
    mccabe \
    pep8-naming \
    pycodestyle \
    pyflakes==2.4.0 \
    black \
    isort \
    pytype>=2020.6.1 \
    types-pkg_resources \
    mypy>=0.790 \
    ninja==1.10.2.3 \
    #torchvision==0.12.0 \
    opencv-python==4.5.5.64 \
    psutil==5.9.1 \
    Sphinx==3.5.3 \
    recommonmark==0.6.0 \
    sphinx-autodoc-typehints==1.11.1 \
    sphinx-rtd-theme==0.5.2 \
    # cucim==22.2.1; platform_system == "Linux" \
    # imagecodecs; platform_system == "Linux" \
    # tifffile; platform_system == "Linux" \
    pandas==1.4.2 \
    requests==2.27.1 \
    einops==0.4.1 \
    transformers==4.19.2 \
    mlflow==1.26.1 \
    matplotlib!=3.5.0 \
    #tensorboardX==2.5 \
    types-PyYAML \
    pyyaml==6.0 \
    fire==0.4.0 \
    jsonschema==4.6.0 \
    kymatio==0.2.1 \
    juliacall==0.8.0 \
    jupyterlab_github==3.0.1 \
    torchio==0.18.77 \
    pytorch-lightning==1.6.4 \
    git+https://github.com/sat28/githubcommit.git \
    gdown==4.4.0 \
    seaborn==0.11.2 \
    # PyTorchUtils==0.0.3 \
    # optuna==2.10.0 \
    jupyterlab-git==0.37.1 \
    comet-ml==3.31.5 \
    # voxelmorph==0.2 \
    # tensorflow==2.9.1 \
    # tensorflow_addons==0.17.0 \
    ipywidgets==7.7.0 \
    h5py==3.7.0 \
    itk-elastix==0.14.1 \
    zenodopy==0.2.0 \
    evalutils==0.3.0 \
    # nnunet==1.7.0 \
    # SimpleITK \
    # git+https://github.com/balbasty/nitorch#egg=nitorch[all] \
    # threadpoolctl==3.1.0 \
    # batchgenerators==0.24 \
    # zenodo_get==1.3.4 \
    # git+https://github.com/DIAGNijmegen/picai_baseline \
    # git+https://github.com/DIAGNijmegen/picai_prep \
    # git+https://github.com/DIAGNijmegen/picai_eval \
    # #https://github.com/DIAGNijmegen/picai_prep/archive/refs/tags/v1.3.2.zip \
    # KevinSR==0.1.19 \
    # dask==2022.6.0 \
    # intensity-normalization[ants] \
    # numba==0.55.2 \
    # pyro-ppl==1.8.1 \
    # warp-lang==0.2.3 \
    report_guided_annotation \
    SimpleITK-SimpleElastix \
    geomloss[full] 

# ENV JAXLIB_VERSION=0.3.16
RUN apt-get install python-dev -y
RUN apt-get install python3.9-dev -y

RUN pip install pycocotools
RUN python3.9 -m  pip install 'monai[nibabel, skimage, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, lmdb, psutil, cucim, pandas, einops, transformers, mlflow, matplotlib, tensorboardX, tifffile, imagecodecs]'
RUN python3.9 -m  pip install torch-optimizer
# RUN pip3 install numpy scipy six wheel jaxlib==${JAXLIB_VERSION}+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
# RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html \

RUN python3.9 -m  pip install optax
RUN python3.9 -m  pip install ott-jax

RUN git clone https://github.com/google-research/scenic.git ${HOME}/build
RUN cd ${HOME}/build/scenic
RUN python3.9 -m  pip install ${HOME}/build

RUN python3.9 -m  pip install evojax[extra]
RUN python3.9 -m  pip install jraph

RUN python3.9 -m pip install --upgrade jax jaxlib
RUN python3.9 -m pip install flax




RUN git config --global user.name "Jakub Mitura"
RUN git config --global user.email "jakub.mitura14@gmail.com"
RUN git config -l


#ENV NVIDIA_DISABLE_REQUIRE=1

# RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html \
#   && pip install -q \
#   git+https://github.com/borisdayma/dalle-mini.git \
#   git+https://github.com/patil-suraj/vqgan-jax.git






# # WORKDIR /workspace

# ## { Mitura start}
# #FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
# # FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
# FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

# ################################################################################
# # Prevent apt-get from prompting for keyboard choice
# #  https://superuser.com/questions/1356914/how-to-install-xserver-xorg-in-unattended-mode
# ENV DEBIAN_FRONTEND=noninteractive

# # Remove any third-party apt sources to avoid issues with expiring keys.
# RUN rm -f /etc/apt/sources.list.d/*.list

# # Install some basic utilities
# RUN apt-get update && apt-get install -y \
#     curl \
#     ca-certificates \
#     sudo \
#     git \
#     bzip2 \
#     libx11-6 \
#     build-essential \
#     wget\
#     manpages-dev\
#     g++\
#     gcc\
#     nodejs\
#     libssl-dev\
#     unzip\
#     #cuda-11.3\
#     #nvidia-cuda-toolkit-11-3\
#     && rm -rf /var/lib/apt/lists/*


# ## installing github CLI - https://github.com/cli/cli/blob/trunk/docs/install_linux.md
# RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
# RUN sudo apt update
# RUN sudo apt install gh

# RUN apt update && sudo apt upgrade -y
# RUN apt install software-properties-common -y
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt install -y python3.10
# RUN apt install -y python3-pip
# RUN apt-get -y purge python3.8
# # Add 3.7 to the available alternatives
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# # Set python3.7 as the default python
# RUN update-alternatives --set python /usr/bin/python3.10


# RUN mkdir /app
# WORKDIR /app


# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
# RUN dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
# RUN apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
# RUN apt-get update
# RUN apt-get -y install cuda

# # RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# # RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# # RUN wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
# # RUN dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
# # RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
# # RUN apt-get update
# # RUN apt-get -y install cuda-11.3
# # RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
# ## { Mitura end}


# ################################################################################
# # get packages
# # - update (without update, old cached server addresses may be used and apt-get install may fail)
# # - install things
# #   - basic  tools
# #   - slicer dependencies
# #   - awesome window manager
# #   - install ca-certs to prevent - fatal: unable to access 'https://github.com/novnc/websockify/': server certificate verification failed. CAfile: none CRLfile: none
# RUN apt-get update -q -y && \
#     apt-get install -q -y \
#     vim net-tools curl \
#     libgl1-mesa-glx \
#     xserver-xorg-video-dummy \
#     libxrender1 \
#     libpulse0 \
#     libpulse-mainloop-glib0  \
#     libnss3  \
#     libxcomposite1 \
#     libxcursor1 \
#     libfontconfig1 \
#     libxrandr2 \
#     libasound2 \
#     libglu1 \
#     x11vnc \
#     awesome \
#     jq \
#     nautilus\
#     jupyter-core\
#     zip\
#     p7zip-full\
#     apt-utils\
#     octave\
#     kmod\
#     zlib1g\
#     python-dev\
#     bzip2\
#     cmake\
#     cuda-command-line-tools-11.6 \
#     libcublas-11.6 \
#     cuda-nvrtc-11.6\
#     libcufft-11.6 \
#     libcurand-11.6 \
#     libcusolver-11.6 \
#     libcusparse-11.6 \
#     libfreetype6-dev \
#     curl\
#     libzmq3-dev \
#     pkg-config\
#     software-properties-common\
#     libhdf5-serial-dev\
#     #elastix=4.9.0-1\
#     git && \
#     apt-get install -q -y --reinstall ca-certificates

# RUN add-apt-repository ppa:deadsnakes/ppa

# # For CUDA profiling, TensorFlow requires CUPTI.
# ENV LD_LIBRARY_PATH /usr/local/cuda-11.0/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# # Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# # dynamic linker run-time bindings
# RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
#     && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
#     && ldconfig

# # See http://bugs.python.org/issue19846
# ENV LANG C.UTF-8


# ################################################################################
# # set up user
# ENV NB_USER sliceruser
# ENV NB_UID 1000
# ENV HOME /home/${NB_USER}

# RUN adduser --disabled-password \
#     --gecos "Default user" \
#     --uid ${NB_UID} \
#     ${NB_USER}

# WORKDIR ${HOME}




# # Install some basic utilities
# RUN mkdir \
#     ${HOME}/data \
#     ${HOME}/labels \
#     ${HOME}/data/preprocess \
#     ${HOME}/data/orig \
#     ${HOME}/data/preprocess/monai_persistent_Dataset \
#     ${HOME}/output \
#     ${HOME}/data/piCaiCode \
#     ${HOME}/build \
#     ${HOME}/data/lightning_logs \
#     ${HOME}/data/preprocess/standarizationModels \
#     ${HOME}/data/preprocess/Bias_field_corrected \
#     ${HOME}/data/metadata



# # COPY bashrc .
# # COPY bashrc /etc/bash.bashrc
# # RUN chown ${NB_USER} /etc/bash.bashrc
# #RUN chmod a+rwx /etc/bash.bashrc




# ENV VNCPORT=49053
# ENV JUPYTERPORT=8888
# ENV DISPLAY=:10

# #COPY xorg.conf .

# ################################################################################
# # Set up remote desktop access - step 1/2

# # Build rebind.so (required by websockify)
# RUN set -x && \
#     apt-get update -q -y && \
#     apt-get install -y build-essential --no-install-recommends && \
#     mkdir src && \
#     cd src && \
#     git clone https://github.com/novnc/websockify websockify && \
#     cd websockify && \
#     make && \
#     cp rebind.so /usr/lib/ && \
#     cd .. && \
#     rm -rf websockify && \
#     cd .. && \
#     rmdir src && \
#     apt install -y python3.10 && \
#     apt install -y python3-pip && \
#     #easy_install -y pip && \
#     apt-get purge -y --auto-remove build-essential


# # Set up launcher for websockify
# # (websockify must run in  Slicer's Python environment)
# # COPY websockify ./Slicer/bin/
# # RUN chmod +x ${HOME}/Slicer/bin/websockify

# ################################################################################
# # Need to run Slicer as non-root because
# # - mybinder requirement
# # - chrome sandbox inside QtWebEngine does not support root.
# # RUN chown ${NB_USER} ${HOME} ${HOME}/Slicer
# RUN chown ${NB_USER} ${HOME} ${HOME}/labels
# RUN chown ${NB_USER} ${HOME} ${HOME}/data/preprocess
# RUN chown ${NB_USER} ${HOME} ${HOME}/data/orig
# RUN chown ${NB_USER} ${HOME} ${HOME}/output
# RUN chown ${NB_USER} ${HOME} ${HOME}/data/piCaiCode
# RUN chown ${NB_USER} ${HOME} ${HOME}/build
# RUN chown ${NB_USER} /var/lib/dpkg
# RUN chown ${NB_USER} ${HOME} ${HOME}/data/lightning_logs
# RUN chown ${NB_USER} ${HOME} ${HOME}/data/preprocess/monai_persistent_Dataset
# RUN chown ${NB_USER} ${HOME} ${HOME}/data/preprocess/standarizationModels
# RUN chown ${NB_USER} ${HOME} ${HOME}/data/preprocess/Bias_field_corrected
# RUN chown ${NB_USER} ${HOME} ${HOME}/data/metadata/







# # First upgrade pip
# RUN python3 -m pip install --upgrade pip


# RUN pip install --no-cache-dir jupyter matplotlib
# # RUN    pip install --no-cache-dir "intensity-normalization[ants]"
# # Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
# # Pin jedi; see https://github.com/ipython/ipython/issues/12740
# RUN pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0 jedi==0.17.2



# #RUN    pip install pyradiomics
# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# RUN pip install \
#     #itk \
#     pytorch-ignite==0.4.8 \
#     PyWavelets==1.3.0 \
#     scipy==1.8.1 \
#     nibabel==3.2.2 \
#     pillow!=8.3.0 \  
#     tensorboard==2.9.0 \
#     scikit-image==0.19.2 \
#     tqdm>=4.47.0 \
#     lmdb==1.3.0 \
#     flake8>=3.8.1 \
#     flake8-bugbear \
#     flake8-comprehensions \
#     flake8-executable \
#     flake8-pyi \
#     pylint!=2.13 \
#     mccabe \
#     pep8-naming \
#     pycodestyle \
#     pyflakes==2.4.0 \
#     black \
#     isort \
#     pytype>=2020.6.1 \
#     types-pkg_resources \
#     mypy>=0.790 \
#     ninja==1.10.2.3 \
#     #torchvision==0.12.0 \
#     opencv-python==4.5.5.64 \
#     psutil==5.9.1 \
#     Sphinx==3.5.3 \
#     recommonmark==0.6.0 \
#     sphinx-autodoc-typehints==1.11.1 \
#     sphinx-rtd-theme==0.5.2 \
#     # cucim==22.2.1; platform_system == "Linux" \
#     # imagecodecs; platform_system == "Linux" \
#     # tifffile; platform_system == "Linux" \
#     pandas==1.4.2 \
#     requests==2.27.1 \
#     einops==0.4.1 \
#     transformers==4.19.2 \
#     mlflow==1.26.1 \
#     matplotlib!=3.5.0 \
#     #tensorboardX==2.5 \
#     types-PyYAML \
#     pyyaml==6.0 \
#     fire==0.4.0 \
#     jsonschema==4.6.0 \
#     kymatio==0.2.1 \
#     juliacall==0.8.0 \
#     jupyterlab_github==3.0.1 \
#     torchio==0.18.77 \
#     pytorch-lightning==1.6.4 \
#     git+https://github.com/sat28/githubcommit.git \
#     gdown==4.4.0 \
#     seaborn==0.11.2 \
#     PyTorchUtils==0.0.3 \
#     optuna==2.10.0 \
#     jupyterlab-git==0.37.1 \
#     comet-ml==3.31.5 \
#     voxelmorph==0.2 \
#     tensorflow==2.9.1 \
#     tensorflow_addons==0.17.0 \
#     ipywidgets==7.7.0 \
#     h5py==3.7.0 \
#     itk-elastix==0.14.1 \
#     zenodopy==0.2.0 \
#     evalutils==0.3.0 \
#     nnunet==1.7.0 \
#     SimpleITK \
#     git+https://github.com/balbasty/nitorch#egg=nitorch[all] \
#     threadpoolctl==3.1.0 \
#     batchgenerators==0.24 \
#     zenodo_get==1.3.4 \
#     git+https://github.com/DIAGNijmegen/picai_baseline \
#     git+https://github.com/DIAGNijmegen/picai_prep \
#     git+https://github.com/DIAGNijmegen/picai_eval \
#     #https://github.com/DIAGNijmegen/picai_prep/archive/refs/tags/v1.3.2.zip \
#     KevinSR==0.1.19 \
#     dask==2022.6.0 \
#     intensity-normalization[ants] \
#     numba==0.55.2 \
#     pyro-ppl==1.8.1 \
#     warp-lang==0.2.3 \
#     report_guided_annotation \
#     SimpleITK-SimpleElastix \
#     geomloss[full] 

# # ENV JAXLIB_VERSION=0.3.16

# RUN pip install 'monai[nibabel, skimage, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, lmdb, psutil, cucim, pandas, einops, transformers, mlflow, matplotlib, tensorboardX, tifffile, imagecodecs]'
# RUN pip install torch-optimizer
# # RUN pip3 install numpy scipy six wheel jaxlib==${JAXLIB_VERSION}+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
# RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html \

# RUN pip install optax
# RUN pip install ott-jax

# # USER root

# # COPY scenic.sh .
# # RUN scenic.sh

# #from https://github.com/NVIDIA/nvidia-docker/issues/1409
# #ENV NVIDIA_DISABLE_REQUIRE=1

# RUN git clone https://github.com/google-research/scenic.git ${HOME}/build
# RUN cd ${HOME}/build/scenic
# RUN pip install ${HOME}/build



# USER ${NB_USER}

# # RUN /home/sliceruser/install.sh ${HOME}/Slicer/Slicer 
# # #RUN ["chmod", "+x", "/home/sliceruser/Slicer/Slicer"]

# EXPOSE $VNCPORT $JUPYTERPORT

# # ENTRYPOINT ["/home/sliceruser/run.sh"]

# # CMD ["sh", "-c", "./Slicer/bin/PythonSlicer -m jupyter notebook --port=$JUPYTERPORT --ip=0.0.0.0 --no-browser --NotebookApp.default_url=/lab/"]

# #login to github cli 
# # COPY mytoken.txt .
# # RUN gh auth login --with-token < mytoken.txt
# RUN git config --global user.name "Jakub Mitura"
# RUN git config --global user.email "jakub.mitura14@gmail.com"
# RUN git config -l




# #CMD ["/bin/bash"]
# ################################################################################
# # Build-time metadata as defined at http://label-schema.org
# ARG BUILD_DATE
# ARG IMAGE
# ARG VCS_REF
# ARG VCS_URL
# LABEL org.label-schema.build-date=$BUILD_DATE \
#     org.label-schema.name=$IMAGE \
#     org.label-schema.vcs-ref=$VCS_REF \
#     org.label-schema.vcs-url=$VCS_URL \
#     org.label-schema.schema-version="1.0"



# ENTRYPOINT [ "/bin/bash", "-l", "-c" ]






