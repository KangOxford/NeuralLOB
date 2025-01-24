FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set the working directory
WORKDIR /home/duser
RUN apt update && apt install python3-pip git vim sudo curl wget apt-transport-https ca-certificates gnupg libgl1 -y
# Install basic dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    vim \
    sudo \
    curl \
    wget \
    apt-transport-https \
    ca-certificates \
    gnupg \
    libgl1 &&\
    apt-get clean && rm -rf /var/lib/apt/lists/* 

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Install additional Python packages
RUN pip install \
    notebook \
    matplotlib \
    pandas \
    tqdm \
    jupyter \
    ipython \
    wandb \
    rich 
    

# Install JAX and dependencies with specific versions
RUN pip3 install --upgrade "jax[cuda]==0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
   pip3 install --upgrade jaxlib==0.4.16 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
   pip3 install --upgrade flax==0.6.11 optax==0.1.7 jaxopt==0.8.1 brax==0.9.2 chex==0.1.8
   RUN pip3 install distrax==0.1.5 gym==0.26.2 gymnax==0.0.6 mujoco==2.3.7 tensorflow-probability==0.22.0 scipy==1.11.3


# Ensure typing-extensions is up-to-date
RUN pip install --upgrade typing-extensions

RUN apt-get update && apt-get install -y p7zip-full
RUN apt-get update && apt-get install -y unrar
RUN apt-get update && apt-get install -y htop

# Install Python packages from requirements.txt
COPY docs/requirements.txt /home/duser/requirements.txt
RUN pip install -r /home/duser/requirements.txt

# Create and configure a non-root user
ARG UID
RUN useradd -u $UID --create-home duser && \
    echo "duser:duser" | chpasswd && \
    adduser duser sudo && \
    mkdir -p /home/duser/.local/bin && \
    chown -R duser:duser /home/duser

# Switch to non-root user
USER duser

# Add alias for ipython
RUN echo "alias i='/usr/local/bin/ipython'" >> ~/.bashrc

RUN git config --global user.email "reuben@robots.ox.ac.uk"
RUN git config --global user.name "reuben"

##Rerun this at the end as it seems to need to be. IDK why?
RUN pip3 install --upgrade "jax[cuda]==0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Default command (can be overridden in the docker run command)
CMD ["/bin/bash"]