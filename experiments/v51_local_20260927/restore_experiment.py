"""Restore the archived experiment source to an ignored local work directory.

Copies code and the experiment protocol only. Never downloads data, starts compute,
copies model weights, overwrites an existing directory, or evaluates anything.
"""
from pathlib import Path
import shutil


def main():
    published = Path(__file__).resolve().parent
    repository = published.parents[1]
    destination = repository / 'artifacts' / 'v51_improve_20260927_0820'
    if destination.exists():
        raise SystemExit(f'Preserving existing experiment; destination already exists: {destination}')
    destination.mkdir(parents=True)
    for source in (published / 'source').glob('*.py'):
        shutil.copyfile(source, destination / source.name)
    shutil.copytree(published / 'source' / 'stage2', destination / 'stage2',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(published / 'metadata' / 'session.json', destination / 'session.json')
    print(f'Restored source and protocol only: {destination}')
    print('Supply the private dataset/export paths described in README.md before running experiments.')


if __name__ == '__main__':
    main()
