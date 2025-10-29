FROM biomlbench-env

# where to put submission, will be extracted
ARG SUBMISSION_DIR=/home/submission
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR=/home/logs
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR=/home/code
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR=/home/agent
ENV AGENT_DIR=${AGENT_DIR}

ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG http_proxy=
ARG https_proxy=
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

RUN if [ -n "$http_proxy" ]; then \
        echo "Acquire::http::Proxy \"$http_proxy\";" > /etc/apt/apt.conf.d/99proxy; \
    fi && \
    if [ -n "$https_proxy" ]; then \
        echo "Acquire::https::Proxy \"$https_proxy\";" >> /etc/apt/apt.conf.d/99proxy; \
    fi

ARG CONDA_ENV_NAME=agent

# copy just the requirements file, so that we can cache conda separately from the agent files
COPY requirements.txt ${AGENT_DIR}/requirements.txt

# Requirements for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# create conda environment and install the requirements to it
RUN conda run -n ${CONDA_ENV_NAME} pip install --upgrade pip && \
    conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt --upgrade && \
    conda run -n ${CONDA_ENV_NAME} pip install --upgrade "openai>=1.50.0" && \
    conda clean -afy


# copy local MLAgentBench directory for debugging
COPY ./ref/MLAgentBench ${AGENT_DIR}/MLAgentBench
COPY ./benchmark_dir ${AGENT_DIR}/MLAgentBench/MLAgentBench/benchmarks/biomlbench

# put remaining necessary agent files in the expected location
COPY additional_notes.txt /home/additional_notes.txt
COPY start.sh ${AGENT_DIR}

# set working directory to MLAgentBench
WORKDIR ${AGENT_DIR}/MLAgentBench
