from pathlib import Path
import sys

from tree_sitter_language_pack import PackConfig, configure


BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
configure(PackConfig(cache_dir=str(BACKEND / ".tree-sitter-cache")))
