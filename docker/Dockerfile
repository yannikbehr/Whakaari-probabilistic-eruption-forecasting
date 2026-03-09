
FROM "mcr.microsoft.com/devcontainers/python:3.11-trixie" 

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libnss3 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    graphviz && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --index-url https://support.bayesfusion.com/pysmile-B/ pysmile && \
    wget https://artifactory.gns.cri.nz:443/artifactory/container-files/pysmile_license.py && \
    cp pysmile_license.py /usr/local/lib/python3.11/site-packages

USER vscode 
