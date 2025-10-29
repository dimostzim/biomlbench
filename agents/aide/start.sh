#!/bin/bash
set -x # Print commands and their arguments as they are executed

cd ${AGENT_DIR}

eval "$(conda shell.bash hook)" # make conda available to the shell
conda activate agent

# determine hardware available
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
else
  HARDWARE="a CPU"
fi
export HARDWARE
# check that we can use the GPU in PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')"

# convert $TIME_LIMIT_SECS to more readable format for prompt
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)

# overwrite instructions.txt with instructions_obfuscated.txt if $OBFUSCATE is set
if [ "$OBFUSCATE" = "true" ]; then
  if [ ! -w /home/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv /home/instructions_obfuscated.txt /home/instructions.txt
fi

# start a new file to store the full instructions, starting with general instructions
cp /home/instructions.txt ${AGENT_DIR}/full_instructions.txt

# add agent-specific instructions with a linebreak in between
echo "" >> ${AGENT_DIR}/full_instructions.txt
envsubst < ${AGENT_DIR}/additional_notes.txt >> ${AGENT_DIR}/full_instructions.txt

# append the task instructions with a linebreak in between
printf "\nCOMPETITION INSTRUCTIONS\n------\n\n" >> ${AGENT_DIR}/full_instructions.txt

# overwrite description.md with description_obfuscated.md if $OBFUSCATE is set
if [ "$OBFUSCATE" = "true" ]; then
  if [ ! -w /home/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv /home/data/description_obfuscated.md /home/data/description.md
fi
cat /home/data/description.md >> ${AGENT_DIR}/full_instructions.txt

# Create workspace and logs directories
mkdir -p ${AGENT_DIR}/logs
mkdir -p ${AGENT_DIR}/workspaces

# Monkey-patch AIDE's OpenRouter backend to support function calling.
PATCH_DIR="${AGENT_DIR}/patches"
mkdir -p "${PATCH_DIR}"
cat <<'PY' > "${PATCH_DIR}/openrouter_patch.py"
import json
import time

from funcy import notnone, select_values

from aide.backend import backend_openrouter
from aide.backend.utils import backoff_create


def _query_with_function_support(
    system_message: str | None,
    user_message: str | None,
    func_spec=None,
    convert_system_to_user: bool = False,
    **model_kwargs,
):
    backend_openrouter._setup_openrouter_client()
    filtered_kwargs = select_values(notnone, model_kwargs)  # type: ignore

    messages = []
    if system_message:
        if convert_system_to_user:
            messages.append({"role": "user", "content": system_message})
        else:
            messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    client = backend_openrouter._client
    if client is None:
        raise RuntimeError("OpenRouter client not initialised")

    t0 = time.time()
    completion = backoff_create(
        client.chat.completions.create,
        backend_openrouter.OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        extra_body={
            "provider": {
                "order": ["Fireworks"],
                "ignore": ["Together", "DeepInfra", "Hyperbolic"],
            },
        },
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]
    if func_spec is None:
        output = choice.message.content
    else:
        tool_calls = getattr(choice.message, "tool_calls", None)
        if not tool_calls:
            raise RuntimeError("Function call requested but tool_calls missing in response")
        func_call = tool_calls[0].function
        if func_call.name != func_spec.name:
            raise RuntimeError(
                f"Function name mismatch: expected {func_spec.name}, got {func_call.name}"
            )
        output = json.loads(func_call.arguments)

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info


backend_openrouter.query = _query_with_function_support
PY
export PYTHONPATH="${PATCH_DIR}:${PYTHONPATH:-}"

# Create a goal description from the task description
GOAL="Build a machine learning model to solve this biomedical task. Focus on understanding the dataset structure, implementing appropriate data preprocessing, selecting suitable algorithms for the task type, and optimizing performance. Full instructions can be found in the instructions.txt file."

# Create evaluation description based on task type
EVAL="Use appropriate evaluation metrics for the task type. See the instructions.txt file for more details."

# run with timeout, using the AIDE format with correct parameters
timeout $TIME_LIMIT_SECS aide \
  data_dir="/home/data/" \
  goal="$GOAL" \
  eval="$EVAL" \
  desc_file="${AGENT_DIR}/full_instructions.txt" \
  $@ # forward the bash arguments to aide

if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi

# Copy results to expected output directories
echo "Copying AIDE results..."

# Copy logs
if [ -d "${AGENT_DIR}/logs" ]; then
  cp -r ${AGENT_DIR}/logs/* ${LOGS_DIR}/ 2>/dev/null || true
fi

# Also check the default logs directory that AIDE creates
if [ -d "./logs" ]; then
  cp -r ./logs/* ${LOGS_DIR}/ 2>/dev/null || true
fi

# Copy the best solution code if it exists
if [ -d "./logs" ]; then
  find ./logs -name "best_solution.py" -type f -exec cp {} ${CODE_DIR}/ \; -quit
elif [ -d "${AGENT_DIR}/logs" ]; then
  find ${AGENT_DIR}/logs -name "best_solution.py" -type f -exec cp {} ${CODE_DIR}/ \; -quit
fi


echo "AIDE execution complete."
