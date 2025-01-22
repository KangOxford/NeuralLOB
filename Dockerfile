FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# Install basic dependencies
RUN apt update && apt install python3-pip git vim sudo curl wget apt-transport-https ca-certificates gnupg libgl1 -y
# Upgrade setuptools
RUN pip install setuptools
# Install JAX and dependencies with specific versions
RUN pip3 uninstall -y jax jaxlib jaxopt flax brax && \
pip3 install --upgrade "jax[cuda]==0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
   pip3 install --upgrade jaxlib==0.4.16 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
   pip3 install --upgrade flax==0.6.11 optax==0.1.7 jaxopt==0.8.1 brax==0.9.2 chex==0.1.8
# Install additional Python packages
RUN pip3 install notebook matplotlib tqdm jupyter ipython wandb rich
RUN pip3 install distrax==0.1.5 gym==0.26.2 gymnax==0.0.6 mujoco==2.3.7 tensorflow-probability==0.22.0 scipy==1.11.3
# Update PATH for local binaries
RUN echo 'export PATH=$PATH:/home/duser/.local/bin' >> ~/.bashrc
# Login to wandb => Change this to your own wandb login
RUN wandb login 136bb7201038330730374a4df59ec4bcc9d39b3a
# Install utilities
RUN apt-get update && apt-get install -y p7zip-full
RUN apt-get update && apt-get install -y unrar
RUN apt-get update && apt-get install -y htop
# Create and configure non-root user
ARG UID
RUN useradd -u $UID --create-home duser && \
   echo "duser:duser" | chpasswd && \
   adduser duser sudo && \
   mkdir -p /home/duser/.local/bin && \
   chown -R duser:duser /home/duser
# Switch to non-root user and configure git
USER duser
WORKDIR /home/duser/
RUN git config --global user.email "valentinm@hotmail.de"
RUN git config --global user.name "valentin"
# Add alias for ipython
RUN echo "alias i='/usr/local/bin/ipython'" >> ~/.bashrc