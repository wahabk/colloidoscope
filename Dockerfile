FROM pytorch/pytorch

# RUN apt-get install libgl1 -y
RUN apt-get update -y \
	&& apt-get install wget -y \
    && apt-get install libopenmpi-dev -y
RUN echo 'export PATH=/root/miniconda3/bin:$PATH' >> /root/.bashrc 

# RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

RUN . /root/.bashrc && \
    /root/miniconda3/bin/conda init bash && \
    /root/miniconda3/bin/conda create -n colloids python=3.8 anaconda && \
    /root/miniconda3/bin/conda install -c conda-forge hoomd -y && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt
    # /root/miniconda3/bin/conda activate colloids

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
# Update pip
# RUN pip3 install --upgrade pip
# # Install pip requirements
# COPY requirements.txt .
# RUN python -m pip install -r requirements.txt

# CMD ["apt-get", "install", "libgl1", "-y"]

WORKDIR /deepcolloid




# TODO use buildkit https://stackoverflow.com/questions/58018300/using-a-pip-cache-directory-in-docker-builds