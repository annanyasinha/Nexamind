"""
MongoDB Atlas connection module for NexaMind session & message persistence.
"""
from pymongo import MongoClient
from pymongo.server_api import ServerApi

from config import settings
from utils.logger import logger


class MongoDatabase:
    """
    Manages lazy connection to MongoDB Atlas database and access to sessions/messages collections.
    """
    def __init__(self):
        self._client = None
        self._db = None
        self._sessions = None
        self._messages = None

    @property
    def client(self):
        if self._client is None:
            self._connect()
        return self._client

    @property
    def db(self):
        if self._db is None:
            self._connect()
        return self._db

    @property
    def sessions(self):
        if self._sessions is None:
            self._connect()
        return self._sessions

    @property
    def messages(self):
        if self._messages is None:
            self._connect()
        return self._messages

    def set_client(self, custom_client):
        """Allows injecting a custom or mock MongoClient (e.g. mongomock) for testing."""
        self._client = custom_client
        self._db = custom_client[settings.MONGODB_DB_NAME]
        self._sessions = self._db["sessions"]
        self._messages = self._db["messages"]

    def _connect(self):
        if not settings.MONGODB_URI:
            logger.warning("MONGODB_URI is not set. MongoDB operations require a valid URI.")
            raise RuntimeError("MONGODB_URI is not configured.")

        try:
            logger.info("Connecting to MongoDB Atlas instance...")
            if settings.MONGODB_URI.startswith("mongodb+srv://") or settings.MONGODB_URI.startswith("mongodb://"):
                self._client = MongoClient(
                    settings.MONGODB_URI,
                    server_api=ServerApi("1"),
                    serverSelectionTimeoutMS=5000
                )
            else:
                self._client = MongoClient(
                    settings.MONGODB_URI,
                    serverSelectionTimeoutMS=5000
                )
            self._db = self._client[settings.MONGODB_DB_NAME]
            self._sessions = self._db["sessions"]
            self._messages = self._db["messages"]
            logger.info(f"Successfully connected to MongoDB database: '{settings.MONGODB_DB_NAME}'")
        except Exception as e:
            logger.error(f"Failed connecting to MongoDB: {e}")
            raise RuntimeError(f"MongoDB connection error: {e!s}")

    def ping(self) -> bool:
        """Pings the MongoDB admin interface to verify database health."""
        try:
            self.client.admin.command("ping")
            return True
        except Exception as e:
            logger.warning(f"MongoDB ping failed: {e}")
            return False


mongo_db = MongoDatabase()
