"""Publish this project to GitHub with Python (wraps the normal git commands).

Usage (from the project root, with the venv active):
    python push_to_github.py https://github.com/<username>/<repo>.git

Authentication: either rely on Git Credential Manager (a browser login pops up the first
time), or set a fine-grained personal access token in an environment variable (never
paste it into code or commit it):
    Windows (PowerShell):  $env:GITHUB_TOKEN = "github_pat_..."
    Mac/Linux:             export GITHUB_TOKEN="github_pat_..."
"""
import base64, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MESSAGE = 'Add SwiftLog Week 5 data science project (EDA, statistics, feature engineering, segmentation, reports)'


def git(*args, check=True, extra=None):
    cmd = ['git'] + (extra or []) + list(args)
    print('$ git', *args)
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=False)


def main():
    if len(sys.argv) != 2 or not sys.argv[1].startswith('https://github.com/'):
        sys.exit('Usage: python push_to_github.py https://github.com/<username>/<repo>.git')
    url = sys.argv[1]
    if subprocess.run(['git', '--version'], capture_output=True).returncode != 0:
        sys.exit('Git is not installed. Install it from https://git-scm.com and try again.')

    # 1. Safety checks before anything is committed
    big = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and p.stat().st_size > 50 * 1024 * 1024]
    if big:
        sys.exit('Files over 50 MB (GitHub warns above 50 MB, blocks above 100 MB): ' + ', '.join(map(str, big)))
    risky = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and p.name.lower() in {'.env', 'credentials.json', 'id_rsa'}]
    if risky:
        sys.exit('Possible secrets found, remove them first: ' + ', '.join(map(str, risky)))

    # 2. Initialise, commit, set branch and remote
    if not (ROOT / '.git').exists():
        git('init')
    git('add', '.')
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode != 0:
        git('commit', '-m', MESSAGE)
    else:
        print('Nothing new to commit.')
    git('branch', '-M', 'main')
    subprocess.run(['git', 'remote', 'remove', 'origin'], cwd=ROOT, capture_output=True)  # ok if none exists
    git('remote', 'add', 'origin', url)

    # 3. Push (token, if provided, is sent as a one-off header and is never stored in the repo or remote URL)
    token = os.environ.get('GITHUB_TOKEN')
    extra = None
    if token:
        basic = base64.b64encode(f'x-access-token:{token}'.encode()).decode()
        extra = ['-c', f'http.extraheader=AUTHORIZATION: basic {basic}']
    git('push', '-u', 'origin', 'main', extra=extra)
    print('\nDone. Open', url.removesuffix('.git'), 'to check the repository.')


if __name__ == '__main__':
    main()
