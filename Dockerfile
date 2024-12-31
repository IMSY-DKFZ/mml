FROM nvidia/cuda:11.7.1-base-ubuntu22.04

# Avoid Docker build freeze due to region selection
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN apt update && apt -y install tzdata

# Basic tools
RUN apt update && apt install -y \
     build-essential \
     wget \
     git

# Setup Python via conda (throuh miniforge)
ENV PATH=/opt/conda/bin:$PATH
RUN wget -O Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
 && bash Miniforge3.sh -b -p "/opt/conda" \
 && rm -f Miniforge3.sh \
 && . "opt/conda/etc/profile.d/conda.sh" \
 && conda update -y -n base conda \
 && conda activate \
 && conda install -y python=3.10

# Copy necessary files
WORKDIR /mml
COPY plugins/ plugins/
COPY src/ src/
COPY tests/ tests/
COPY MANIFEST.in pyproject.toml README.md setup.cfg ./

# Install mml (insert your extras like e.g. "[dev, docs]" here)
ENV EXTRAS=""
RUN pip install ."$EXTRAS" && cd plugins

# Install plugins
CMD /bin/bash -c 'while IFS='' read -r LINE || [ -n "${LINE}" ]; do \
        cd ${LINE} \n\
        pip install . \n\
        cd .. \n\
      done' < index.txt

# Adapt mml.env and set env path
ENV NUM_WORKERS=8
RUN mkdir /data \
	&& mkdir /results \
	&& mml-env-setup \
	&& sed -i -e 's/\/path\/to\/data/\/data/;s/\/path\/to\/results/\/results/;s/available_local_CPU_cores/'$NUM_WORKERS'/' mml.env
ENV MML_ENV_PATH=/mml/mml.env

CMD ["/bin/bash"]
