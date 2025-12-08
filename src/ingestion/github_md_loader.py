"""
---------------------------------------------------
1_github: GitHub Markdown Loader 
---------------------------------------------------

This module downloads Markdown documentation files from JASP GitHub
repositories such as:

    jasp-stats/jaspRegression
    jasp-stats/jaspMixedModels

It supports both:
  • Single-repository ingestion  
  • Batch ingestion using `data/raw_github/github_list.json`

Workflow: 
Fetches .md help files directly from a GitHub repo folder (no clone needed).
--------
1. Query the GitHub API for Markdown files inside the target folder
   (default: `inst/help`).
2. Download each `.md` file via the raw-content URL.
3. Save it under `data/raw_github/` with a repository prefix:
     repo "jasp-stats/jaspRegression"
     file "RegressionLinear.md"
     → saved as "jasp-stats_jaspRegression__RegressionLinear.md"
4. These files are later consumed by the embedding and RAG pipelines.

Basic usage
-----------
Programmatic:

    from src.ingestion.github_md_loader import github_loader_from_reponame
    github_loader_from_reponame("jasp-stats/jaspRegression")

Batch mode (recommended):

    poetry run python -m src.ingestion.github_md_loader
---------------------------------------------------
"""

import os
import requests
from pathlib import Path
from loguru import logger
import json

def github_loader_from_reponame(
    repo_path: str,
    folder_path: str = "inst/help",
    target_folder: str = "data/raw_github",
):
    """
    Download Markdown documentation files from a specific GitHub repository.

    Example:
        github_loader_from_reponame(
            repo_path="jasp-stats/jaspRegression",
            folder_path="inst/help"
        )

    Behavior:
        • Calls the GitHub REST API to list files inside `folder_path`.
        • Downloads all `.md` files (except localized `_nl` versions).
        • Saves them into `data/raw_github/`, prefixing filenames with the
          repository name for traceability:

              jasp-stats_jaspRegression__RegressionLinear.md

    Args:
        repo_path:
            GitHub repository in the form "owner/repo", e.g. "jasp-stats/jaspAnova".

        folder_path:
            Folder inside the repo to fetch Markdown files from.
            Default is "inst/help", where JASP help files are stored.

        target_folder:
            Local directory where files will be written.

    Returns:
        None. Markdown files are saved locally.
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
    Batch-load Markdown help files from multiple GitHub repositories.

    Reads a JSON file specifying repositories and folders:

        {
          "github": [
            { "repo": "jasp-stats/jaspRegression", "folder": "inst/help" },
            { "repo": "jasp-stats/jaspAnova", "folder": "inst/help" }
          ]
        }

    For each entry, this function calls `github_loader_from_reponame` to download all Markdown files.

    Args:
        json_path:
            Path to a JSON list of repositories to ingest.

    Returns:
        None. Files are downloaded into `data/raw_github/`.
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

