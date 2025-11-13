"""
---------------------------------------------------
🌐 GitHub Markdown Loader (API version)
---------------------------------------------------
Fetches .md help files directly from a GitHub repo
folder (no clone needed).

- Uses GitHub API to list all files under inst/help/
- Downloads each .md file from the raw content URL
- Skips localized (_nl) files
- Saves into data/raw_github/

Usage:
    from src.ingestion.github_md_loader import github_loader
    github_loader("jasp-stats/jaspRegression")
run:
    poetry run python -m src.ingestion.github_md_loader
 

---------------------------------------------------
"""

import os
import requests
from pathlib import Path
from loguru import logger


def github_loader(repo_path: str, folder_path: str = "inst/help", target_folder: str = "data/raw_github"):
    """
    Load Markdown files from a GitHub repo folder (via API).

    Args:
        repo_path (str): e.g. "jasp-stats/jaspRegression"
        folder_path (str): path inside the repo (default: inst/help)
        target_folder (str): where to save markdown files
    """
    api_url = f"https://api.github.com/repos/{repo_path}/contents/{folder_path}"
    base_raw_url = f"https://raw.githubusercontent.com/{repo_path}/master/{folder_path}/"

    output_dir = Path(target_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching file list from: {api_url}")
    response = requests.get(api_url)
    if response.status_code != 200:
        logger.error(f"Failed to list files: {response.status_code} {response.text}")
        return

    files = [f["name"] for f in response.json() if f["name"].endswith(".md")]
    logger.info(f"Found {len(files)} markdown files in {repo_path}/{folder_path}")

    for fname in files:
        if "_nl" in fname:
            logger.debug(f"Skipping localized file: {fname}")
            continue

        file_url = base_raw_url + fname
        logger.info(f"Downloading {file_url}")
        r = requests.get(file_url)
        if r.status_code == 200:
            file_path = output_dir / fname
            file_path.write_text(r.text, encoding="utf-8")
            logger.info(f"Saved → {file_path}")
        else:
            logger.warning(f"Failed to download {fname}: {r.status_code}")

    logger.success(f"✅ Completed loading files into {output_dir}")


# Example usage
if __name__ == "__main__":
    github_loader("jasp-stats/jaspRegression")
    github_loader("jasp-stats/jaspMixedModels")

