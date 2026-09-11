
import re
import os
import requests
import base64
import json

from pathlib import Path
from typing import Tuple, Dict, Any, List
from urllib.parse import quote

from config import settings
from core.vectorstore import FaissVectorStore
from langchain_core.documents import Document
from utils.logger import logger


from datetime import datetime, timezone

MAX_GITHUB_FILE_SIZE = 300 * 1024
GITHUB_ALLOWED_EXTENSIONS = {
    ".py",
    ".java",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".html",
    ".css",
    ".md",
    ".txt",
    ".json",
    ".xml",
    ".yml",
    ".yaml",
    ".properties",
    ".sql",
    ".sh"
}

GITHUB_IGNORED_DIRS = {
    ".git",
    ".github",
    "node_modules",
    "target",
    "build",
    "dist",
    "__pycache__",
    ".idea",
    ".vscode",
    "venv",
    ".venv"
}
GITHUB_IGNORED_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.json",
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",

    # additional sensitive files
    "service-account.json",
    "firebase-adminsdk.json"
}

def extract_github_repo(url: str) -> Tuple[str, str]:

    if not url:
        raise ValueError("GitHub repository URL cannot be empty.")

    url = url.strip()

    pattern = r"^https?://github\.com/([^/]+)/([^/#]+)"

    match = re.match(pattern, url)

    if not match:
        raise ValueError("Invalid GitHub repository URL.")

    owner = match.group(1)
    repo = match.group(2)

    if repo.endswith(".git"):
        repo = repo[:-4]

    return owner, repo


def get_github_headers() -> Dict[str, str]:

    headers = {
        "Accept": "application/vnd.github+json"
    }

    github_token = os.getenv("GITHUB_TOKEN")

    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    return headers


def get_repository_info(repo_url: str) -> Dict[str, Any]:

    owner, repo = extract_github_repo(repo_url)

    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    response = requests.get(
        api_url,
        headers=get_github_headers(),
        timeout=10
    )

    if response.status_code == 404:
        raise ValueError("GitHub repository not found.")

    if response.status_code == 403:
        raise RuntimeError(
            "GitHub API access denied or rate limit exceeded."
        )

    response.raise_for_status()

    data = response.json()

    return {
        "owner": owner,
        "repo": repo,
        "full_name": data["full_name"],
        "default_branch": data["default_branch"],
        "private": data["private"],
        "html_url": data["html_url"]
    }


def get_latest_commit_sha(
    owner: str,
    repo: str,
    branch: str
) -> str:

    encoded_branch = quote(branch, safe="")


    api_url = (
    f"https://api.github.com/repos/"
    f"{owner}/{repo}/branches/{encoded_branch}"
    )

    response = requests.get(
        api_url,
        headers=get_github_headers(),
        timeout=10
    )

    if response.status_code == 404:
        raise ValueError("GitHub branch not found.")

    if response.status_code == 403:
        raise RuntimeError(
            "GitHub API access denied or rate limit exceeded."
        )

    response.raise_for_status()

    data = response.json()

    return data["commit"]["sha"]


def get_github_repo_metadata(repo_url: str) -> Dict[str, Any]:

    repo_info = get_repository_info(repo_url)

    commit_sha = get_latest_commit_sha(
        repo_info["owner"],
        repo_info["repo"],
        repo_info["default_branch"]
    )

    repo_info["commit_sha"] = commit_sha

    return repo_info



def get_repository_tree(
    owner: str,
    repo: str,
    branch: str
) -> List[Dict[str, Any]]:

    encoded_branch = quote(branch, safe="")

    api_url = (
        f"https://api.github.com/repos/"
        f"{owner}/{repo}/git/trees/{encoded_branch}"
    )

    response = requests.get(
        api_url,
        headers=get_github_headers(),
        params={"recursive": "1"},
        timeout=15
    )

    if response.status_code == 404:
        raise ValueError(
            "Could not retrieve GitHub repository tree."
        )

    if response.status_code == 403:
        raise RuntimeError(
            "GitHub API access denied or rate limit exceeded."
        )

    response.raise_for_status()

    data = response.json()

    if data.get("truncated"):
        raise RuntimeError(
            "Repository is too large to retrieve recursively."
        )

    return data.get("tree", [])

def is_ignored_path(file_path: str) -> bool:

    parts = Path(file_path).parts

    for part in parts:
        if part in GITHUB_IGNORED_DIRS:
            return True

    return False

def filter_repository_files(
    tree: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    useful_files = []

    for item in tree:

        # We only want files, not directories
        if item.get("type") != "blob":
            continue

        file_path = item.get("path", "")
        file_size = item.get("size", 0) or 0

        path_obj = Path(file_path)
        filename = path_obj.name
        extension = path_obj.suffix.lower()

        # Ignore unwanted folders
        if is_ignored_path(file_path):
            continue

        # Ignore sensitive/unnecessary files
        if filename in GITHUB_IGNORED_FILES:
            continue

        # Ignore private key/certificate files
        if extension in {
            ".pem",
            ".key",
            ".p12",
            ".pfx",
            ".crt"
        }:
            continue

        # Ignore unsupported file types
        if extension not in GITHUB_ALLOWED_EXTENSIONS:
            continue

        # Ignore very large files
        if file_size > MAX_GITHUB_FILE_SIZE:
            continue

        useful_files.append(item)

    return useful_files

def get_useful_repository_files(
    repo_url: str
) -> Dict[str, Any]:

    metadata = get_github_repo_metadata(repo_url)

    tree = get_repository_tree(
        metadata["owner"],
        metadata["repo"],
        metadata["default_branch"]
    )

    files = filter_repository_files(tree)

    return {
        "repository": metadata["full_name"],
        "branch": metadata["default_branch"],
        "commit_sha": metadata["commit_sha"],
        "total_tree_items": len(tree),
        "useful_files": files,
        "useful_file_count": len(files)
    }


def fetch_github_file_content(
    owner: str,
    repo: str,
    file_sha: str
) -> str:

    api_url = (
        f"https://api.github.com/repos/"
        f"{owner}/{repo}/git/blobs/{file_sha}"
    )

    response = requests.get(
        api_url,
        headers=get_github_headers(),
        timeout=15
    )

    if response.status_code == 403:
        raise RuntimeError(
            "GitHub API access denied or rate limit exceeded."
        )

    response.raise_for_status()

    data = response.json()

    encoded_content = data.get("content", "")

    if not encoded_content:
        return ""

    decoded_content = base64.b64decode(
        encoded_content
    )

    return decoded_content.decode(
        "utf-8",
        errors="replace"
    )

def get_language_from_extension(
    file_path: str
) -> str:

    extension = Path(file_path).suffix.lower()

    language_map = {
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
        ".json": "json",
        ".xml": "xml",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".properties": "properties",
        ".md": "markdown",
        ".txt": "text",
        ".sh": "shell"
    }

    return language_map.get(
        extension,
        "text"
    )

def create_github_document(
    content: str,
    owner: str,
    repo: str,
    file_path: str,
    commit_sha: str,
    file_sha: str
) -> Document:

    github_file_url = (
        f"https://github.com/"
        f"{owner}/{repo}/blob/"
        f"{commit_sha}/{file_path}"
    )

    return Document(
        page_content=content,
        metadata={
    "source": file_path,
    "source_type": "github",
    "repository": f"{owner}/{repo}",
    "owner": owner,
    "repo": repo,
    "file_path": file_path,
    "filename": Path(file_path).name,
    "language": get_language_from_extension(file_path),
    "github_url": github_file_url,
    "commit_sha": commit_sha,
    "file_sha": file_sha
}
    )

def load_github_repository(
    repo_url: str
) -> List[Document]:

    repo_data = get_useful_repository_files(
        repo_url
    )

    repository = repo_data["repository"]

    owner, repo = repository.split("/", 1)

    commit_sha = repo_data["commit_sha"]

    useful_files = repo_data["useful_files"]

    documents = []

    logger.info(
        f"Loading {len(useful_files)} files "
        f"from GitHub repository {repository}"
    )

    for file_info in useful_files:

        file_path = file_info["path"]
        file_sha = file_info["sha"]

        try:

            content = fetch_github_file_content(
                owner,
                repo,
                file_sha
            )

            if not content.strip():
                continue

            document = create_github_document(
                content=content,
                owner=owner,
                repo=repo,
                file_path=file_path,
                commit_sha=commit_sha,
                file_sha=file_sha
            )

            documents.append(document)

        except Exception as e:

            logger.warning(
                f"Could not load GitHub file "
                f"{file_path}: {e}"
            )

    logger.info(
        f"Loaded {len(documents)} GitHub "
        f"documents from {repository}"
    )

    return documents

def get_github_store_dir(
    owner: str,
    repo: str
) -> Path:

    repo_folder = f"{owner}__{repo}"

    return (
        settings.GITHUB_FAISS_STORE_DIR
        / repo_folder
    )

def build_github_vectorstore(
    repo_url: str
) -> FaissVectorStore:

    owner, repo = extract_github_repo(repo_url)

    logger.info(
        f"Preparing GitHub repository "
        f"{owner}/{repo} for indexing."
    )

    documents = load_github_repository(repo_url)

    if not documents:
        raise ValueError(
            "No useful GitHub files were found "
            "for indexing."
        )

    store_dir = get_github_store_dir(
        owner,
        repo
    )

    vectorstore = FaissVectorStore(
        persist_dir=store_dir,
        embedding_model=settings.EMBEDDING_MODEL
    )

    vectorstore.build_from_documents(
        documents
    )

    logger.info(
        f"GitHub repository {owner}/{repo} "
        f"successfully indexed."
    )

    return vectorstore

def load_github_manifest(
    store_dir: Path
) -> Dict[str, Any] | None:

    manifest_path = store_dir / "manifest.json"

    if not manifest_path.exists():
        return None

    try:
        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except Exception as e:
        logger.warning(
            f"Could not read GitHub manifest: {e}"
        )
        return None

def save_github_manifest(
    store_dir: Path,
    metadata: Dict[str, Any]
) -> None:

    store_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest = {
        "repository": metadata["full_name"],
        "branch": metadata["default_branch"],
        "commit_sha": metadata["commit_sha"],
        "indexed_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    manifest_path = (
        store_dir / "manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2
        )

def is_github_cache_valid(
    store_dir: Path,
    current_commit_sha: str
) -> bool:

    manifest = load_github_manifest(
        store_dir
    )

    if not manifest:
        return False

    cached_sha = manifest.get(
        "commit_sha"
    )

    faiss_path = (
        store_dir / "faiss.index"
    )

    metadata_path = (
        store_dir / "metadata.pkl"
    )

    if not faiss_path.exists():
        return False

    if not metadata_path.exists():
        return False

    return cached_sha == current_commit_sha

def get_or_create_github_vectorstore(
    repo_url: str
) -> FaissVectorStore:

    metadata = get_github_repo_metadata(
        repo_url
    )

    owner = metadata["owner"]
    repo = metadata["repo"]

    current_commit_sha = metadata[
        "commit_sha"
    ]

    store_dir = get_github_store_dir(
        owner,
        repo
    )

    # CACHE HIT
    if is_github_cache_valid(
        store_dir,
        current_commit_sha
    ):

        logger.info(
            f"Reusing cached GitHub FAISS "
            f"index for {owner}/{repo}"
        )

        vectorstore = FaissVectorStore(
            persist_dir=store_dir,
            embedding_model=
                settings.EMBEDDING_MODEL
        )

        vectorstore.load()

        return vectorstore

    # CACHE MISS
    logger.info(
        f"GitHub repository {owner}/{repo} "
        f"is new or has changed. "
        f"Rebuilding FAISS index."
    )

    vectorstore = build_github_vectorstore(
        repo_url
    )

    save_github_manifest(
        store_dir,
        metadata
    )

    return vectorstore