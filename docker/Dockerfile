ARG D_UID="1001"
ARG D_GID="1001"

FROM "mcr.microsoft.com/devcontainers/python:3.13-trixie" AS base

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

RUN --mount=type=bind,target=/build \
    if [ -f /build/pysmile_license.py ]; then \
        /usr/local/bin/pip3 install --index-url https://support.bayesfusion.com/pysmile-B/ pysmile && \
        cp /build/pysmile_license.py /usr/local/lib/python3.13/site-packages/; \
    fi

FROM base AS dev

USER vscode

FROM base AS app

ARG D_UID
ARG D_GID

COPY /docker/fix-permissions /usr/local/bin/fix-permissions

# Configure environment
ENV SHELL=/bin/bash \
    D_UID=$D_UID \
    D_GID=$D_GID \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    HOME=/home/bayes

RUN groupadd -g $D_GID bayes && \
    useradd -m -s /bin/bash -N -u $D_UID -G bayes bayes && \
    chmod a+rx /usr/local/bin/fix-permissions && \
    mkdir $HOME/data && \
    fix-permissions $HOME && \
    fix-permissions $HOME/data && \
    fix-permissions /usr/local

USER bayes
WORKDIR $HOME
COPY . $HOME 
RUN --mount=type=cache,target=${HOME}/.cache/pip /usr/local/bin/pip3 install .
RUN echo 'y' | plotly_get_chrome
