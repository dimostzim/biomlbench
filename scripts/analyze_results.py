#!/usr/bin/env python3
"""
BioML-bench Results Analysis Script

Downloads and analyzes biomlbench results from the organized S3 structure.

Features:
- Parallel downloads (configurable workers)
- Filters for organized structure only (agent/task/file)
- Agent filtering (--agents stella mlagentbench)
- Task filtering (--tasks polarishub kaggle)
- Comprehensive analysis outputs:
  * Markdown summary report
  * CSV files for further analysis
  * Agent performance comparison
  * Task difficulty rankings
  * Detailed results matrix

Usage Examples:
  # Full analysis
  python scripts/analyze_results.py --output-dir results_analysis
  
  # Filter specific agents
  python scripts/analyze_results.py --agents stella biomni --parallel-downloads 15
  
  # Filter specific task types
  python scripts/analyze_results.py --tasks polarishub manual --output-dir polaris_manual_analysis
  
  # Dry run to see what would be downloaded
  python scripts/analyze_results.py --dry-run
"""

import argparse
import gzip
import json
import shutil
import subprocess
import tarfile
# Removed defaultdict - explicit error handling only
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import os

def log(message: str):
    """Simple logging with timestamp."""
    import time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_command(cmd: List[str], description: str = "") -> Tuple[int, str]:
    """Run a command and return (exit_code, output)."""
    try:
        if description:
            log(f"Running: {description}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr
    except Exception as e:
        return 1, str(e)

def discover_s3_artifacts(bucket: str, prefix: str, agents: List[str] = None, tasks: List[str] = None) -> Dict[str, List[str]]:
    """Discover S3 artifacts with smart filtering."""
    log("Discovering S3 artifacts...")
    
    artifacts = {
        'runs': [],
        'grades': [],
        'failed_runs': [],
        'failed_grades': []
    }
    
    for artifact_type in artifacts.keys():
        s3_path = f"s3://{bucket}/{prefix}/{artifact_type}/"
        exit_code, output = run_command([
            "aws", "s3", "ls", s3_path, "--recursive"
        ], f"List {artifact_type}")
        
        if exit_code == 0:
            for line in output.strip().split('\n'):
                if line.strip():
                    # Parse S3 ls output: "2025-08-20 01:10:23    123 path/to/file"
                    parts = line.split()
                    if len(parts) >= 4:
                        s3_key = ' '.join(parts[3:])
                        full_path = f"s3://{bucket}/{s3_key}"
                        
                        # Check if this is organized structure or flat structure
                        key_parts = s3_key.split('/')
                        should_include = False
                        
                        if len(key_parts) >= 5:  # Potential organized structure: prefix/artifact_type/agent/task/file
                            agent_part = key_parts[-3]
                            task_part = key_parts[-2] 
                            filename = key_parts[-1]
                            
                            # Check for organized structure (agent/task/file)
                            if (not agent_part.startswith('2025-') and 
                                not task_part.startswith('2025-') and
                                filename.startswith('2025-')):
                                
                                # Apply agent filter
                                if agents and agent_part not in agents:
                                    continue
                                    
                                # Apply task filter
                                if tasks:
                                    task_matches = any(pattern in task_part for pattern in tasks)
                                    if not task_matches:
                                        continue
                                
                                should_include = True
                        
                        # For runs: include organized structure
                        if artifact_type == 'runs':
                            if should_include:
                                artifacts[artifact_type].append(full_path)
                        
                        # For grades: only include individual reports (skip aggregated reports)
                        elif artifact_type == 'grades':
                            if should_include and 'individual_reports' in filename:
                                artifacts[artifact_type].append(full_path)
                        
                        # For failed_runs/failed_grades: include both organized and flat structure
                        elif artifact_type in ['failed_runs', 'failed_grades']:
                            if should_include:
                                artifacts[artifact_type].append(full_path)
                            else:
                                # Check for flat structure: artifacts/failed_runs/2025-08-19T00-19-11-GMT_run-group_dummy.tar.gz
                                filename = key_parts[-1]
                                if filename.startswith('2025-') and '_run-group_' in filename:
                                    # Apply agent filter for flat structure
                                    if agents and '_run-group_' in filename:
                                        file_agent = filename.split('_run-group_')[-1].replace('.tar.gz', '').replace('.json.gz', '')
                                        if file_agent not in agents:
                                            continue
                                    
                                    artifacts[artifact_type].append(full_path)
    
    log(f"Found {len(artifacts['runs'])} organized run artifacts, {len(artifacts['grades'])} organized grade artifacts")
    log(f"Found {len(artifacts['failed_runs'])} failed runs, {len(artifacts['failed_grades'])} failed grades")
    
    return artifacts

def download_and_extract_single_artifact(s3_path: str, output_dir: Path, artifact_type: str, force_download: bool = False) -> bool:
    """Download and extract a single artifact."""
    try:
        # Parse agent/task/filename from S3 path
        # e.g., s3://bucket/v1/artifacts/runs/stella/polarishub-tdcommons-herg/file.tar.gz
        path_parts = s3_path.replace('s3://', '').split('/')
        
        type_dir = output_dir / artifact_type
        
        if len(path_parts) >= 6:  # organized structure
            agent = path_parts[4]
            task_safe = path_parts[5]
            filename = path_parts[6]
            
            # Create agent/task directory structure locally
            local_dir = type_dir / agent / task_safe
            local_dir.mkdir(parents=True, exist_ok=True)
            local_file = local_dir / filename
            
            # Check if extracted content already exists (skip re-download unless forced)
            if not force_download:
                if filename.endswith('.tar.gz'):
                    extracted_name = filename.replace('.tar.gz', '')
                    if (local_dir / extracted_name).exists():
                        return True  # Already extracted
                elif filename.endswith('.gz'):
                    uncompressed_name = filename.replace('.gz', '')
                    if (local_dir / uncompressed_name).exists():
                        return True  # Already decompressed
        else:  # flat structure fallback
            filename = path_parts[-1]
            local_file = type_dir / filename
            
            # Check if extracted content already exists
            if not force_download:
                if filename.endswith('.tar.gz'):
                    extracted_name = filename.replace('.tar.gz', '')
                    if (type_dir / extracted_name).exists():
                        return True
                elif filename.endswith('.gz'):
                    uncompressed_name = filename.replace('.gz', '')
                    if (type_dir / uncompressed_name).exists():
                        return True
        
        # Skip download if file already exists locally (unless forced)
        if not force_download and local_file.exists():
            return True
        
        # Download file
        exit_code, output = run_command([
            "aws", "s3", "cp", s3_path, str(local_file)
        ])
        
        if exit_code != 0:
            raise RuntimeError(f"Failed to download {s3_path}: {output}")
        
        # Extract if compressed
        if filename.endswith('.tar.gz'):
            try:
                with tarfile.open(local_file, 'r:gz') as tar:
                    tar.extractall(local_file.parent, filter='data')
                local_file.unlink()  # Remove compressed file after extraction
            except Exception as e:
                log(f"⚠️  Failed to extract {filename}: {e}")
                return False
        
        elif filename.endswith('.gz') and not filename.endswith('.tar.gz'):
            try:
                uncompressed_file = local_file.with_suffix('')
                with gzip.open(local_file, 'rb') as f_in:
                    with open(uncompressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                local_file.unlink()  # Remove compressed file after extraction
            except Exception as e:
                log(f"⚠️  Failed to decompress {filename}: {e}")
                return False
        
        return True
        
    except Exception as e:
        log(f"⚠️  Error processing {s3_path}: {e}")
        return False

def download_and_extract_artifacts(artifacts: Dict[str, List[str]], output_dir: Path, max_workers: int = 10, force_download: bool = False):
    """Download and extract all artifacts in parallel."""
    output_dir.mkdir(exist_ok=True)
    
    # Collect all download tasks
    download_tasks = []
    for artifact_type, s3_paths in artifacts.items():
        if not s3_paths:
            continue
            
        type_dir = output_dir / artifact_type
        type_dir.mkdir(exist_ok=True)
        
        for s3_path in s3_paths:
            download_tasks.append((s3_path, output_dir, artifact_type))
    
    if not download_tasks:
        log("No artifacts to download")
        return
    
    log(f"Downloading {len(download_tasks)} artifacts with {max_workers} parallel workers...")
    
    successful_downloads = 0
    failed_downloads = 0
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_path = {
            executor.submit(download_and_extract_single_artifact, s3_path, output_dir, artifact_type, force_download): s3_path
            for s3_path, output_dir, artifact_type in download_tasks
        }
        
        # Process completed downloads with progress reporting
        for future in as_completed(future_to_path):
            s3_path = future_to_path[future]
            completed += 1
            
            try:
                success = future.result()
                if success:
                    successful_downloads += 1
                else:
                    failed_downloads += 1
            except Exception as e:
                log(f"⚠️  Exception downloading {s3_path}: {e}")
                failed_downloads += 1
            
            # Progress report every 25 downloads
            if completed % 25 == 0:
                log(f"Progress: {completed}/{len(download_tasks)} downloads completed ({completed/len(download_tasks)*100:.1f}%)")
    
    log(f"✅ Downloaded {successful_downloads} artifacts successfully")
    if failed_downloads > 0:
        log(f"⚠️  {failed_downloads} downloads failed")

def analyze_grading_results(grades_dir: Path) -> List[Dict[str, Any]]:
    """Extract ALL individual results - no aggregation."""
    log("Extracting individual grading results...")
    
    # Store ALL individual results (replicates) - NO AGGREGATION
    all_results = []
    
    # Find all individual report JSON files (extracted from individual_reports directories)
    report_files = []
    for root, dirs, files in os.walk(grades_dir):
        for file in files:
            # Look for individual task report JSON files (extracted from individual_reports.tar.gz)
            if file.endswith('.json') and not file.endswith('_grading_report.json'):
                report_files.append(Path(root) / file)
    
    log(f"Found {len(report_files)} individual result files to process")
    
    for report_file in report_files:
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            # REQUIRED fields - no defaults
            if 'task_id' not in report:
                raise ValueError(f"Missing required field 'task_id' in {report_file}")
            if 'score' not in report:
                raise ValueError(f"Missing required field 'score' in {report_file}")
            if 'leaderboard_percentile' not in report:
                raise ValueError(f"Missing required field 'leaderboard_percentile' in {report_file}")
            if 'gold_medal' not in report:
                raise ValueError(f"Missing required field 'gold_medal' in {report_file}")
            if 'silver_medal' not in report:
                raise ValueError(f"Missing required field 'silver_medal' in {report_file}")
            if 'bronze_medal' not in report:
                raise ValueError(f"Missing required field 'bronze_medal' in {report_file}")
            if 'above_median' not in report:
                raise ValueError(f"Missing required field 'above_median' in {report_file}")
            
            task_id = report['task_id']
            score = report['score']
            
            if score is None:
                raise ValueError(f"Score is null in {report_file}")
            
            # Extract agent from file path structure: .../grades/agent/task-safe/...
            path_parts = report_file.parts
            agent = None
            
            for i, part in enumerate(path_parts):
                if part == 'grades' and i + 1 < len(path_parts):
                    agent = path_parts[i + 1]
                    break
            
            if agent is None:
                raise ValueError(f"Could not extract agent from file path in {report_file}")
            
            # Convert types explicitly
            percentile = float(report['leaderboard_percentile'])
            gold_medal = bool(report['gold_medal'])
            silver_medal = bool(report['silver_medal'])
            bronze_medal = bool(report['bronze_medal'])
            above_median = bool(report['above_median'])
            
            # Store this individual result - each JSON file is one replicate
            result = {
                'agent': agent,
                'task_id': task_id,
                'score': float(score),
                'leaderboard_percentile': percentile,
                'above_median': above_median,
                'gold_medal': gold_medal,
                'silver_medal': silver_medal,
                'bronze_medal': bronze_medal,
                'any_medal': gold_medal or silver_medal or bronze_medal,
                'report_file': str(report_file)
            }
            all_results.append(result)
            
        except Exception as e:
            log(f"❌ FAILED to parse {report_file}: {e}")
            raise  # Fail loudly
    
    log(f"Extracted {len(all_results)} individual results")
    
    if len(all_results) == 0:
        raise ValueError("No individual results found!")
    
    return all_results

def analyze_run_metadata(runs_dir: Path) -> Dict[str, Any]:
    """Analyze run metadata for execution times and resource usage."""
    log("Analyzing run metadata...")
    
    metadata_files = []
    for root, dirs, files in os.walk(runs_dir):
        for file in files:
            if file == 'metadata.json':
                metadata_files.append(Path(root) / file)
    
    analysis = {
        'execution_times': {},
        'task_durations': {},
        'agent_performance': {}
    }
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Require essential fields
            if 'agent_id' not in metadata:
                raise ValueError(f"Missing required field 'agent_id' in {metadata_file}")
            if 'runs' not in metadata:
                raise ValueError(f"Missing required field 'runs' in {metadata_file}")
            
            agent_id = metadata['agent_id']
            runs = metadata['runs']
            
            # Initialize agent performance tracking
            if agent_id not in analysis['agent_performance']:
                analysis['agent_performance'][agent_id] = {
                    'total_time': 0.0,
                    'successful_runs': 0,
                    'failed_runs': 0
                }
            
            for run_id, run_data in runs.items():
                if 'task_id' not in run_data:
                    raise ValueError(f"Missing required field 'task_id' in run {run_id} of {metadata_file}")
                if 'success' not in run_data:
                    raise ValueError(f"Missing required field 'success' in run {run_id} of {metadata_file}")
                
                task_id = run_data['task_id']
                success = run_data['success']
                
                if success:
                    analysis['agent_performance'][agent_id]['successful_runs'] += 1
                else:
                    analysis['agent_performance'][agent_id]['failed_runs'] += 1
                
        except Exception as e:
            log(f"❌ FAILED to parse {metadata_file}: {e}")
            raise  # Fail loudly instead of continuing
    
    return analysis

# Removed generate_summary_report - only raw CSV output needed

def generate_csv_reports(all_results: List[Dict], output_dir: Path):
    """Generate CSV from raw results - each result is one replicate."""
    log("Generating CSV reports...")
    
    if len(all_results) == 0:
        raise ValueError("No results to generate CSV from!")
    
    # Replicate-level results CSV - this is the PRIMARY output
    replicate_df = pd.DataFrame(all_results)
    
    # Remove internal fields before saving
    if 'report_file' in replicate_df.columns:
        replicate_df = replicate_df.drop('report_file', axis=1)
    
    replicate_df.to_csv(output_dir / 'all_replicates.csv', index=False)
    log(f"Saved {len(all_results)} individual results to all_replicates.csv")

def main():
    parser = argparse.ArgumentParser(
        description="Download and analyze BioML-bench results from S3"
    )
    parser.add_argument(
        "--bucket",
        default="biomlbench",
        help="S3 bucket name (default: biomlbench)"
    )
    parser.add_argument(
        "--prefix", 
        default="v1/artifacts",
        help="S3 prefix (default: v1/artifacts)"
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_results",
        help="Local directory to store downloaded artifacts and analysis (default: analysis_results)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download phase and analyze existing local data"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files already exist locally"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Show what would be downloaded without actually downloading"
    )
    parser.add_argument(
        "--parallel-downloads",
        type=int,
        default=10,
        help="Number of parallel download workers (default: 10)"
    )
    parser.add_argument(
        "--agents",
        nargs='+',
        help="Filter to specific agents (e.g., --agents stella mlagentbench)"
    )
    parser.add_argument(
        "--tasks",
        nargs='+', 
        help="Filter to specific task patterns (e.g., --tasks polarishub kaggle)"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if not args.skip_download:
        # Discover artifacts
        artifacts = discover_s3_artifacts(args.bucket, args.prefix, args.agents, args.tasks)
        
        if args.dry_run:
            log("DRY RUN - Would download:")
            for artifact_type, paths in artifacts.items():
                log(f"  {artifact_type}: {len(paths)} files")
                for path in paths[:3]:  # Show first 3 examples
                    log(f"    {path}")
                if len(paths) > 3:
                    log(f"    ... and {len(paths) - 3} more")
            return 0
        
        # Download and extract
        download_and_extract_artifacts(artifacts, output_dir, args.parallel_downloads, args.force_download)
    
    # Analyze results
    grades_dir = output_dir / 'grades'
    runs_dir = output_dir / 'runs'
    
    if grades_dir.exists():
        all_results = analyze_grading_results(grades_dir)
        
        # Generate CSV with ALL individual results
        generate_csv_reports(all_results, output_dir)
        
        log("Analysis complete!")
        log(f"📈 All results CSV: {output_dir / 'all_replicates.csv'}")
        
        # Quick terminal summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        if len(all_results) == 0:
            raise ValueError("No results found!")
        
        total_replicates = len(all_results)
        unique_agents = set(r['agent'] for r in all_results)
        unique_tasks = set(r['task_id'] for r in all_results)
        medal_winners = sum(1 for r in all_results if r['any_medal'])
        
        print(f"Total replicates: {total_replicates}")
        print(f"Unique agents: {len(unique_agents)} ({', '.join(sorted(unique_agents))})")
        print(f"Unique tasks: {len(unique_tasks)}")
        print(f"Medal-winning replicates: {medal_winners}")
        
        # Show replicate counts per agent-task
        from collections import Counter
        agent_task_counts = Counter((r['agent'], r['task_id']) for r in all_results)
        print(f"\nReplicate counts per agent-task:")
        for (agent, task), count in sorted(agent_task_counts.items()):
            print(f"  {agent} x {task}: {count} replicates")
        
        print("="*60)
        
    else:
        raise ValueError("No grading results directory found!")
    
    return 0

if __name__ == "__main__":
    exit(main()) 