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
import json



def github_loader_from_reponame(repo_path: str, folder_path: str = "inst/help", target_folder: str = "data/raw_github"):
    """
    Load Markdown files from a GitHub repo folder (via API).
    Save them with repo prefix in filename to preserve metadata.
    """
    api_url = f"https://api.github.com/repos/{repo_path}/contents/{folder_path}"
    base_raw_url = f"https://raw.githubusercontent.com/{repo_path}/master/{folder_path}/"

    output_dir = Path(target_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert "jasp-stats/jaspRegression" → "jasp-stats_jaspRegression"
    safe_repo_prefix = repo_path.replace("/", "_")

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

            # ---------------------------------------------------
            # NEW: Save file with repo prefix
            # e.g., jasp-stats_jaspRegression__RegressionLinear.md
            # ---------------------------------------------------
            prefixed_name = f"{safe_repo_prefix}__{fname}"

            file_path = output_dir / prefixed_name
            file_path.write_text(r.text, encoding="utf-8")

            logger.info(f"Saved → {file_path}")

        else:
            logger.warning(f"Failed to download {fname}: {r.status_code}")

    logger.success(f"✅ Completed loading files into {output_dir}")



def github_loader(json_path="data/raw_github/github_list.json"):
    """
    Load GitHub repos listed in github_list.json using the existing github_loader()
    without modifying the original ingestion pipeline.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"GitHub list JSON not found: {json_path}")

    logger.info(f"Loading GitHub repo list from {json_path}")

    data = json.loads(json_path.read_text())

    for entry in data.get("github", []):
        repo = entry["repo"]
        folder = entry.get("folder", "inst/help")

        logger.info(f"📥 Loading GitHub repo: {repo} | folder: {folder}")

        # Call your existing pipeline exactly
        github_loader_from_reponame(repo_path=repo, folder_path=folder)

    logger.success("🎉 Completed loading all GitHub repos from list.")


if __name__ == "__main__":
    github_loader()

