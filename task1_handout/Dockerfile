FROM mambaorg/micromamba:latest
COPY env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu && \
    pip install pyarmor==6.7.4 \
    pip install gpytorch
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/code/pytransform
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
WORKDIR /code
CMD python -u checker_client.py
