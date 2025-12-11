FROM biomlbench-env:latest

# Declare standard directories so build args can override defaults
ARG SUBMISSION_DIR=/home/submission
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
ARG LOGS_DIR=/home/logs
ENV LOGS_DIR=${LOGS_DIR}
ARG CODE_DIR=/home/code
ENV CODE_DIR=${CODE_DIR}
ARG AGENT_DIR=/home/agent
ENV AGENT_DIR=${AGENT_DIR}

# Ensure expected directories exist inside the image
RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

WORKDIR ${AGENT_DIR}

# Install dependencies in base agent environment
RUN conda run -n agent pip install openai python-dotenv

# Copy agent files
COPY start.sh ${AGENT_DIR}/start.sh
COPY zeroshot_agent.py ${AGENT_DIR}/zeroshot_agent.py

RUN chmod +x ${AGENT_DIR}/start.sh
