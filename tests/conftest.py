import sys
import pytest
from pathlib import Path

# Add src to python path for pytest executions
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

@pytest.fixture
def tmp_data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    sample_file = d / "sample.txt"
    sample_file.write_text("Annanya Sinha studied at IIT BHU and BIT Sindri.", encoding="utf-8")
    return d
