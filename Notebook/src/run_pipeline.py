"""Runs the whole SwiftLog Week 5 pipeline: scripts 01-06 in order, in one shared namespace
(they are notebook cells), then writes Notebook/Week5_DataScience.ipynb with real outputs,
all charts, cleaned data, the feature-engineered dataset and the profiling report.

Run from the project root:   python Notebook/src/run_pipeline.py
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import notebook_runner  # noqa: E402

ROOT = SRC.parents[1]
scripts = sorted(SRC.glob('0[1-6]_*.py'))
print('Running', len(scripts), 'scripts:', *[s.name for s in scripts], sep='\n  ')
notebook_runner.main([str(s) for s in scripts], str(ROOT / 'Notebook' / 'Week5_DataScience.ipynb'),
                     str(ROOT / 'Notebook'), str(ROOT / 'Notebook' / 'results.json'))
print('Done: Notebook/Week5_DataScience.ipynb written.')
