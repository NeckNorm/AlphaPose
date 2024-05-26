# Use NVIDIA CUDA Image
FROM nvidia/cuda:11.5.2-devel-ubuntu20.04

# Install necessary packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget \
    vim \
    git \
    build-essential \
    libyaml-dev \
    libgl1-mesa-glx \
    libglib2.0-0 # 추가된 glib 라이브러리

# Install locales and generate C.UTF-8 locale
RUN apt-get install -y locales
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    /bin/bash /tmp/anaconda.sh -b -p /root/anaconda3 && \
    rm /tmp/anaconda.sh

# Set PATH for Anaconda
ENV PATH /root/anaconda3/bin:$PATH

# Create a Conda virtual environment for AlphaPose
RUN conda create -n alphapose python=3.7 -y

# Activate the virtual environment and install PyTorch
RUN echo "source activate alphapose" > ~/.bashrc && \
    /bin/bash -c "source activate alphapose && \
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia -y"

# Install additional Python packages in the virtual environment
RUN /bin/bash -c "source activate alphapose && \
    pip install flask flask_socketio natsort opencv-python easydict Cython matplotlib" # 추가된 matplotlib

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda/bin/:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

# No need to clone AlphaPose as it will be mounted as a volume

# Install PyTorch3D (Optional)
RUN echo "source activate alphapose" > ~/.bashrc && \
    /bin/bash -c "source activate alphapose && \
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y && \
    conda install -c bottler nvidiacub -y && \
    pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'"

# Ensure the conda environment is activated when the container starts
CMD ["/bin/bash", "-c", "source activate alphapose && /bin/bash"]
