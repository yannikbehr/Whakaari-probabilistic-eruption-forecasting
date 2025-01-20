ARG ROOT_CONTAINER=condaforge/miniforge3:23.11.0-0
ARG BASE_CONTAINER=$ROOT_CONTAINER

FROM $BASE_CONTAINER AS base 

ARG D_USER="linus"
ARG D_UID="1000"
ARG D_GID="1000"

USER root

# Create D_USER with name linus user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN groupadd -g $D_GID $D_USER && \ 
    useradd -m -s /bin/bash -N -u $D_UID -g $D_GID $D_USER && \
    mkdir -p /opt/data && \
    mkdir -p /home/$D_USER/workspace && \
    mkdir /env && \
    chown $D_USER:$D_GID /env && \
    chown -R $D_USER:$D_GID /home/$D_USER/workspace && \
    chown -R $D_USER:$D_GID /opt/data

WORKDIR /tmp

USER $D_USER

COPY environment.yml .

RUN --mount=type=cache,target=/opt/conda/pkgs mamba env create -p /env -f environment.yml

RUN /env/bin/pip install --index-url https://support.bayesfusion.com/pysmile-B/ pysmile && \
    wget https://artifactory.gns.cri.nz:443/artifactory/container-files/pysmile_license.py && \
    cp pysmile_license.py /env/lib/python3.10/site-packages

COPY . .

#RUN /env/bin/pip install -e .

VOLUME ["/opt/data"]
VOLUME ["/home/$D_USER/workspace"]
EXPOSE 8003
