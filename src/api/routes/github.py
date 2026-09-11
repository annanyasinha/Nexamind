from fastapi import APIRouter, HTTPException , Request

from api.schemas import (
    GitHubIndexRequest,
    GitHubIndexResponse,
    GitHubQueryRequest,
    GitHubQueryResponse
)
from api.rate_limit import (
    limiter,
    GITHUB_INDEX_RATE_LIMIT,
    GITHUB_QUERY_RATE_LIMIT
)
from core.github_loader import (
    extract_github_repo,
    get_github_repo_metadata,
    get_github_store_dir,
    get_or_create_github_vectorstore,
    is_github_cache_valid
)

from core.tools import GitHubRAGTool
from utils.logger import logger


router = APIRouter( tags=["GitHub RAG"])

@router.post("/github/index",response_model=GitHubIndexResponse)
@limiter.limit(GITHUB_INDEX_RATE_LIMIT)
def index_github_repository(request: Request,req: GitHubIndexRequest):
    """
    Index or load a public GitHub repository
    into a repository-specific FAISS store.
    """

    try:
        metadata = get_github_repo_metadata(
            req.repo_url
        )

        owner = metadata["owner"]
        repo = metadata["repo"]

        store_dir = get_github_store_dir(
            owner,
            repo
        )

        was_cached = is_github_cache_valid(
            store_dir,
            metadata["commit_sha"]
        )

        vectorstore = (
            get_or_create_github_vectorstore(
                req.repo_url
            )
        )

        total_vectors = (
            vectorstore.index.ntotal
            if vectorstore.index is not None
            else 0
        )

        return GitHubIndexResponse(
            repository=metadata["full_name"],
            commit_sha=metadata["commit_sha"],
            total_vectors=total_vectors,
            cached=was_cached
        )

    except ValueError as e:

        logger.warning(
            f"GitHub indexing validation error: {e}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        logger.error(
            f"GitHub repository indexing failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"GitHub indexing failed: {e!s}"
        )

@router.post(
    "/github/query",
    response_model=GitHubQueryResponse
)
@limiter.limit(GITHUB_QUERY_RATE_LIMIT)
def query_github_repository(
    request: Request,
    req: GitHubQueryRequest
):

    """
    Perform semantic search over a GitHub repository.
    """

    try:
        owner, repo = extract_github_repo(
            req.repo_url
        )

        github_tool = GitHubRAGTool()

        result = github_tool.run(
            query=req.query,
            repo_url=req.repo_url,
            top_k=req.top_k
        )

        return GitHubQueryResponse(
            repository=f"{owner}/{repo}",
            query=req.query,
            output=result["output"],
            sources=result.get(
                "sources",
                []
            ),
            execution_time_ms=result[
                "execution_time_ms"
            ]
        )

    except ValueError as e:

        logger.warning(
            f"GitHub query validation error: {e}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        logger.error(
            f"GitHub repository query failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"GitHub query failed: {e!s}"
        )