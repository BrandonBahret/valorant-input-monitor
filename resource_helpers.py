from pathlib import Path
import sys


def resource_path(relative_path: str) -> Path:
    exe_dir = Path(sys.argv[0]).resolve().parent
    return exe_dir / relative_path

def bundled_resource_path(relative_path: str) -> Path:
    try:
        import nuitka
        if getattr(nuitka, "Compiled", False):
            return Path(__file__).parent / relative_path
    except ImportError:
        pass
    return Path(__file__).parent / relative_path
