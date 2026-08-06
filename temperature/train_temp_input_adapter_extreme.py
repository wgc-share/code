from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = CODE_ROOT / "ablation"
if str(ABLATION_DIR) not in sys.path:
    sys.path.insert(0, str(ABLATION_DIR))

from ablation_temp_extreme_runner import run_cli


if __name__ == "__main__":
    run_cli("temp_input_adapter_no_ah")
