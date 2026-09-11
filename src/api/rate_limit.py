from slowapi import Limiter
from slowapi.util import get_remote_address


limiter = Limiter(
    key_func=get_remote_address
)


AGENT_RATE_LIMIT = "15/minute"
RAG_QUERY_RATE_LIMIT = "20/minute"
RAG_STREAM_RATE_LIMIT = "10/minute"
VECTOR_SEARCH_RATE_LIMIT = "30/minute"

GITHUB_INDEX_RATE_LIMIT = "3/minute"
GITHUB_QUERY_RATE_LIMIT = "20/minute"

YOUTUBE_RATE_LIMIT = "5/minute"