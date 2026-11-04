"""Run from a source checkout without installing the local package."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from immersed_elasticity.benchmark import main

if __name__ == "__main__":
    main()
