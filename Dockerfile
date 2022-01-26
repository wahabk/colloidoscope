# FROM pytorch/pytorch
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
# docker pull pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
ENV PATH="/root/miniconda3/bin:$PATH"
ARG PATH="/root/miniconda3/bin:$PATH"

# RUN apt-get install libgl1 -y
RUN apt-get update -y \
	&& apt-get install wget -y \
    && apt-get install libopenmpi-dev -y
# RUN echo 'export PATH=/root/miniconda3/bin:$PATH' >> /root/.bashrc 

# RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

WORKDIR /colloidoscope
COPY environment.yml .
# from https://stackoverflow.com/questions/55123637/activate-conda-environment-in-docker
RUN conda init bash \
    && . ~/.bashrc \
    && conda env create -f environment.yml \
    && conda activate colloids \
    && pip install ipython 
    
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# COPY tests/test_gpu.py .
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "colloids", "python3", "tests/test_gpu.py"]
# ENTRYPOINT ["conda", "activate", "colloids"]

WORKDIR /colloidoscope
# Update pip
# RUN pip3 install --upgrade pip
# # Install pip requirements
# COPY requirements.txt .
# RUN python -m pip install -r requirements.txt



# TODO use buildkit https://stackoverflow.com/questions/58018300/using-a-pip-cache-directory-in-docker-builds