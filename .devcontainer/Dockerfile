FROM mcr.microsoft.com/devcontainers/base:ubuntu

# The container comes minimized by default, which is intended for production containers that users do not log into.
RUN yes | unminimize

# Install Python.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    sudo \
    build-essential \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN (. /etc/lsb-release && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | \
    sudo env os=ubuntu dist="${DISTRIB_CODENAME}" bash) && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js.
RUN curl -SLO https://deb.nodesource.com/nsolid_setup_deb.sh && \
    chmod 500 nsolid_setup_deb.sh && \
    ./nsolid_setup_deb.sh 21 && \
    apt-get install nodejs -y

# Install Docker.
RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh

RUN usermod -aG docker vscode

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt && \
    rm requirements.txt requirements-dev.txt
