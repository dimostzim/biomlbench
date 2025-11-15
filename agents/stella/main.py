"""
Simple BioMLBench interface for STELLA
"""
import argparse
import os
import sys
from pathlib import Path


def _register_stella_path() -> None:
    """Ensure the bundled STELLA sources are importable."""
    stella_path = Path(__file__).resolve().parent / "ref" / "STELLA"
    if stella_path.exists():
        sys.path.insert(0, str(stella_path))


_register_stella_path()

# Import stella_core module early to initialize all global components
import stella_core
from stella_core import *


def main():
    parser = argparse.ArgumentParser(description="STELLA agent for BioMLBench")
    parser.add_argument("--model", type=str, default="claude-sonnet-4")
    # FIX: BioMLBench passes boolean values, so use type=bool instead of action="store_true"
    parser.add_argument("--use_templates", type=lambda x: x.lower() in ('true', '1', 'yes'), default=False)
    parser.add_argument("--use_mem0", type=lambda x: x.lower() in ('true', '1', 'yes'), default=False)
    parser.add_argument("--max_tools", type=int, default=30)
    parser.add_argument("--timeout_seconds", type=int, default=21600)
    
    args = parser.parse_args()

    timeout_seconds = int(os.getenv("TIME_LIMIT_SECS", args.timeout_seconds))
    max_steps = int(os.getenv("STEP_LIMIT", args.max_tools))
    
    print(f"Initializing STELLA with model: {args.model}")
    print(f"Configured runtime limits → timeout: {timeout_seconds}s, max steps: {max_steps}")
    
    # Verify all core components exist after stella_core import
    print(f"🤖 Dev agent: {type(stella_core.dev_agent).__name__ if stella_core.dev_agent else 'Missing'}")
    print(f"🎯 Critic agent: {type(stella_core.critic_agent).__name__ if stella_core.critic_agent else 'Missing'}")
    print(f"🛠️ Tool creation agent: {type(stella_core.tool_creation_agent).__name__ if stella_core.tool_creation_agent else 'Missing'}")
    print(f"🧠 Models available: grok={stella_core.grok_model is not None}, gemini={stella_core.gemini_model is not None}")
    
    # Initialize STELLA (creates manager_agent and memory system)
    success = stella_core.initialize_stella(
        use_template=args.use_templates,
        use_mem0=args.use_mem0
    )
    
    if not success:
        raise RuntimeError("Failed to initialize STELLA")
    
    # Access manager_agent after initialization
    manager_agent = stella_core.manager_agent
    
    if manager_agent is None:
        raise RuntimeError("Manager agent was not created properly")
    
    print(f"✅ Manager agent: {type(manager_agent).__name__} with {len(manager_agent.tools)} tools")
    
    # Read full instructions (includes benchmark instructions + agent notes + task description)
    instructions_path = os.path.join(os.path.dirname(__file__), "instructions.txt")
    with open(instructions_path, 'r') as f:
        full_instructions = f.read()

    # Combine high-level guidance with detailed instructions
    prompt = f"""
Build a machine learning model to solve this biomedical task.
Focus on understanding the dataset structure, implementing appropriate data preprocessing,
selecting suitable algorithms for the task type, and optimizing performance.
Use appropriate evaluation metrics for the task type.

Full instructions below:

{full_instructions}
"""
    
    print("Running STELLA on biomedical task...")
    # Note: timeout_seconds is handled by the shell timeout command in start.sh
    result = manager_agent.run(prompt, max_steps=max_steps)
    print("STELLA execution completed.")
    
    return result


if __name__ == "__main__":
    main() 
