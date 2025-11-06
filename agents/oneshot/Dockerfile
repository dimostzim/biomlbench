FROM biomlbench-env:latest

WORKDIR /home/agent

# Install dependencies in base agent environment
RUN conda run -n agent pip install openai python-dotenv

# Copy agent files
COPY start.sh /home/agent/start.sh
COPY oneshot_agent.py /home/agent/oneshot_agent.py

RUN chmod +x /home/agent/start.sh

ENTRYPOINT ["/home/agent/start.sh"]
