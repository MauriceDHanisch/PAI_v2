FROM mambaorg/micromamba:latest
COPY env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
RUN pip install pyarmor==6.7.4
USER root
RUN apt-get update -y && apt-get install -y gcc || (apt-get update -y && apt-get install -y gcc)
RUN apt-get update && apt-get install -y procps
RUN pip install hypno
WORKDIR /code
ADD * /code/
ADD code.py /code/code.py
ADD hypno_script.py /code/hypno_script.py
ADD entrypoint.sh /code/entrypoint.sh
RUN chmod +x /code/entrypoint.sh
ADD pytransform /code/pytransform
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/code/pytransform
WORKDIR /code
#CMD ["/bin/bash"]
CMD ["/bin/bash", "/code/entrypoint.sh"]
# CMD python -u checker_client.py
