import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

def safe_filename(file: str) -> str:
    return re.sub(r'[^\w_. -]', '_', file)

def get_commit() -> Optional[str]:
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()[:7]
        return commit
    except Exception:
        return None

def run_mmseqs2(
    query_seqs_unique: List[str],
    result_dir: str,
    use_env: bool,
    use_templates: bool,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    host_url: str = "",
    user_agent: str = "",
) -> Tuple[Optional[List[str]], Optional[dict]]:
    # This is a placeholder function. In a real implementation, you would need to
    # implement the actual MMseqs2 search functionality here.
    print("Running MMseqs2 search...")
    return None, None

def setup_logging(log_file: Path, mode: str = "w") -> None:
    import logging
    logging.basicConfig(filename=str(log_file), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode=mode)

ENV = {"TF_FORCE_UNIFIED_MEMORY":"1", "XLA_PYTHON_CLIENT_MEM_FRACTION":"4.0"}
for k,v in ENV.items():
    if k not in os.environ: os.environ[k] = v