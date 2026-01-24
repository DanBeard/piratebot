"""
Voice Line Database using ChromaDB for semantic search.

Stores pre-generated voice lines with embeddings for fast semantic lookup.
Supports category filtering, recently-used exclusion, and fallback hierarchy.
"""

import json
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VoiceLine:
    """A single voice line with metadata."""

    id: str
    text: str
    category: str
    subcategory: str = ""
    tags: list[str] = field(default_factory=list)
    emotion: str = "neutral"
    audio_file: Optional[str] = None
    viseme_file: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for ChromaDB metadata."""
        return {
            "text": self.text,
            "category": self.category,
            "subcategory": self.subcategory,
            "tags": ",".join(self.tags),
            "emotion": self.emotion,
            "audio_file": self.audio_file or "",
            "viseme_file": self.viseme_file or "",
        }

    @classmethod
    def from_db_result(cls, id: str, metadata: dict, document: str) -> "VoiceLine":
        """Create VoiceLine from ChromaDB query result."""
        return cls(
            id=id,
            text=document,
            category=metadata.get("category", ""),
            subcategory=metadata.get("subcategory", ""),
            tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
            emotion=metadata.get("emotion", "neutral"),
            audio_file=metadata.get("audio_file") or None,
            viseme_file=metadata.get("viseme_file") or None,
        )


@dataclass
class SearchResult:
    """Result from voice line search."""

    line: VoiceLine
    similarity_score: float

    @property
    def is_good_match(self) -> bool:
        """Check if this is a good semantic match (>= 0.65)."""
        return self.similarity_score >= 0.65

    @property
    def is_excellent_match(self) -> bool:
        """Check if this is an excellent semantic match (>= 0.85)."""
        return self.similarity_score >= 0.85


class VoiceLineDB:
    """
    Voice line database with semantic search capabilities.

    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    Supports:
    - Semantic search by text similarity
    - Category and tag filtering
    - Recently-used line exclusion
    - Fallback hierarchy for graceful degradation
    """

    # Similarity thresholds
    EXCELLENT_MATCH_THRESHOLD = 0.85
    GOOD_MATCH_THRESHOLD = 0.65

    # Category priority for fallbacks
    FALLBACK_PRIORITY = [
        "costume_reactions/generic",
        "fallbacks/positive",
        "fallbacks/interjections",
    ]

    def __init__(
        self,
        db_path: str = "data/voice_line_db",
        collection_name: str = "voice_lines",
        embedding_model: str = "all-MiniLM-L6-v2",
        recently_used_maxlen: int = 20,
        recently_used_cooldown: float = 300.0,  # 5 minutes
    ):
        """
        Initialize voice line database.

        Args:
            db_path: Path to ChromaDB persistent storage
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
            recently_used_maxlen: Max number of recently used lines to track
            recently_used_cooldown: Seconds before a line can be reused
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.recently_used_cooldown = recently_used_cooldown

        # Recently used tracking: (line_id, timestamp)
        self._recently_used: deque[tuple[str, float]] = deque(maxlen=recently_used_maxlen)

        # Lazy-loaded components
        self._client = None
        self._collection = None
        self._embedding_fn = None

        logger.info(f"VoiceLineDB initialized: path={db_path}, model={embedding_model}")

    def _ensure_loaded(self) -> None:
        """Lazy-load ChromaDB and embedding model."""
        if self._collection is not None:
            return

        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )

        # Create persistent client
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.db_path))

        # Create embedding function
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB loaded: {self._collection.count()} voice lines in collection"
        )

    def add_line(self, line: VoiceLine) -> None:
        """
        Add a voice line to the database.

        Args:
            line: VoiceLine to add
        """
        self._ensure_loaded()

        self._collection.upsert(
            ids=[line.id],
            documents=[line.text],
            metadatas=[line.to_dict()],
        )

        logger.debug(f"Added voice line: {line.id}")

    def add_lines(self, lines: list[VoiceLine]) -> None:
        """
        Add multiple voice lines to the database.

        Args:
            lines: List of VoiceLines to add
        """
        if not lines:
            return

        self._ensure_loaded()

        self._collection.upsert(
            ids=[line.id for line in lines],
            documents=[line.text for line in lines],
            metadatas=[line.to_dict() for line in lines],
        )

        logger.info(f"Added {len(lines)} voice lines to database")

    def get_line(self, line_id: str) -> Optional[VoiceLine]:
        """
        Get a specific voice line by ID.

        Args:
            line_id: The voice line ID

        Returns:
            VoiceLine if found, None otherwise
        """
        self._ensure_loaded()

        result = self._collection.get(ids=[line_id])

        if not result["ids"]:
            return None

        return VoiceLine.from_db_result(
            id=result["ids"][0],
            metadata=result["metadatas"][0],
            document=result["documents"][0],
        )

    def search(
        self,
        query_text: str,
        n_results: int = 5,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        tags: Optional[list[str]] = None,
        exclude_recently_used: bool = True,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search for voice lines by semantic similarity.

        Args:
            query_text: Text to search for
            n_results: Maximum number of results to return
            category: Filter by category (e.g., "costume_reactions")
            subcategory: Filter by subcategory (e.g., "scary")
            tags: Filter by tags (any match)
            exclude_recently_used: Exclude lines used within cooldown period
            min_score: Minimum similarity score to include

        Returns:
            List of SearchResult sorted by similarity (highest first)
        """
        self._ensure_loaded()

        # Build where filter (ChromaDB requires $and for multiple conditions)
        where_filter = None
        if category and subcategory:
            where_filter = {
                "$and": [
                    {"category": {"$eq": category}},
                    {"subcategory": {"$eq": subcategory}},
                ]
            }
        elif category:
            where_filter = {"category": {"$eq": category}}
        elif subcategory:
            where_filter = {"subcategory": {"$eq": subcategory}}

        # Get IDs to exclude
        exclude_ids = set()
        if exclude_recently_used:
            current_time = time.time()
            for line_id, used_time in self._recently_used:
                if current_time - used_time < self.recently_used_cooldown:
                    exclude_ids.add(line_id)

        # Query more results than needed to account for filtering
        query_n = n_results + len(exclude_ids) + 10

        # Perform search
        results = self._collection.query(
            query_texts=[query_text],
            n_results=query_n,
            where=where_filter if where_filter else None,
        )

        # Process results
        search_results = []

        if not results["ids"] or not results["ids"][0]:
            return search_results

        for i, line_id in enumerate(results["ids"][0]):
            # Skip excluded lines
            if line_id in exclude_ids:
                continue

            # ChromaDB returns distances, convert to similarity
            # For cosine distance: similarity = 1 - distance
            distance = results["distances"][0][i]
            similarity = 1.0 - distance

            # Skip below minimum score
            if similarity < min_score:
                continue

            # Check tag filter if specified
            if tags:
                line_tags = results["metadatas"][0][i].get("tags", "").split(",")
                if not any(tag in line_tags for tag in tags):
                    continue

            line = VoiceLine.from_db_result(
                id=line_id,
                metadata=results["metadatas"][0][i],
                document=results["documents"][0][i],
            )

            search_results.append(SearchResult(line=line, similarity_score=similarity))

            if len(search_results) >= n_results:
                break

        return search_results

    def search_with_fallback(
        self,
        query_text: str,
        category_hint: Optional[str] = None,
        costume_type: Optional[str] = None,
    ) -> Optional[SearchResult]:
        """
        Search with automatic fallback hierarchy.

        Tries progressively broader searches until a match is found:
        1. Exact semantic match in specific category (>= 0.85)
        2. Good match in category (>= 0.65)
        3. Generic costume reaction
        4. Universal positive fallback
        5. Simple interjection

        Args:
            query_text: What we want to say
            category_hint: Suggested category to search
            costume_type: Type of costume (for tag boosting)

        Returns:
            Best matching SearchResult or None if database is empty
        """
        self._ensure_loaded()

        # Stage 1: Try specific category with high threshold
        if category_hint:
            results = self.search(
                query_text,
                n_results=3,
                category=category_hint,
                tags=[costume_type] if costume_type else None,
                min_score=self.EXCELLENT_MATCH_THRESHOLD,
            )
            if results:
                logger.debug(f"Found excellent match in {category_hint}")
                return results[0]

        # Stage 2: Try specific category with lower threshold
        if category_hint:
            results = self.search(
                query_text,
                n_results=3,
                category=category_hint,
                min_score=self.GOOD_MATCH_THRESHOLD,
            )
            if results:
                logger.debug(f"Found good match in {category_hint}")
                return results[0]

        # Stage 3: Try generic costume reactions
        results = self.search(
            query_text,
            n_results=5,
            category="costume_reactions",
            subcategory="generic",
        )
        if results:
            # Pick a random one from top results to add variety
            result = random.choice(results[:min(3, len(results))])
            logger.debug("Using generic costume reaction")
            return result

        # Stage 4: Try positive fallbacks
        results = self.search(
            query_text,
            n_results=5,
            category="fallbacks",
            subcategory="positive",
        )
        if results:
            result = random.choice(results[:min(3, len(results))])
            logger.debug("Using positive fallback")
            return result

        # Stage 5: Try any fallback
        results = self.search(
            query_text,
            n_results=5,
            category="fallbacks",
        )
        if results:
            result = random.choice(results)
            logger.debug("Using generic fallback")
            return result

        # Last resort: get any line at all
        all_results = self._collection.get(limit=10)
        if all_results["ids"]:
            idx = random.randrange(len(all_results["ids"]))
            line = VoiceLine.from_db_result(
                id=all_results["ids"][idx],
                metadata=all_results["metadatas"][idx],
                document=all_results["documents"][idx],
            )
            logger.warning("Using random line as last resort")
            return SearchResult(line=line, similarity_score=0.0)

        logger.error("No voice lines in database!")
        return None

    def mark_used(self, line_id: str) -> None:
        """
        Mark a voice line as recently used.

        Args:
            line_id: ID of the line that was used
        """
        self._recently_used.append((line_id, time.time()))
        logger.debug(f"Marked line as used: {line_id}")

    def get_random_from_category(
        self,
        category: str,
        subcategory: Optional[str] = None,
        exclude_recently_used: bool = True,
    ) -> Optional[VoiceLine]:
        """
        Get a random voice line from a category.

        Args:
            category: Category to sample from
            subcategory: Optional subcategory filter
            exclude_recently_used: Exclude recently used lines

        Returns:
            Random VoiceLine or None if category is empty
        """
        self._ensure_loaded()

        # Build where filter (ChromaDB requires $and for multiple conditions)
        if subcategory:
            where_filter = {
                "$and": [
                    {"category": {"$eq": category}},
                    {"subcategory": {"$eq": subcategory}},
                ]
            }
        else:
            where_filter = {"category": {"$eq": category}}

        # Get all matching lines
        results = self._collection.get(
            where=where_filter,
            limit=100,
        )

        if not results["ids"]:
            return None

        # Filter recently used if requested
        candidates = []
        exclude_ids = set()

        if exclude_recently_used:
            current_time = time.time()
            for line_id, used_time in self._recently_used:
                if current_time - used_time < self.recently_used_cooldown:
                    exclude_ids.add(line_id)

        for i, line_id in enumerate(results["ids"]):
            if line_id not in exclude_ids:
                candidates.append((
                    line_id,
                    results["metadatas"][i],
                    results["documents"][i],
                ))

        if not candidates:
            # All lines recently used, pick any
            idx = random.randrange(len(results["ids"]))
            candidates = [(
                results["ids"][idx],
                results["metadatas"][idx],
                results["documents"][idx],
            )]

        # Pick random candidate
        line_id, metadata, document = random.choice(candidates)
        return VoiceLine.from_db_result(line_id, metadata, document)

    def count(self) -> int:
        """Return total number of voice lines in database."""
        self._ensure_loaded()
        return self._collection.count()

    def get_categories(self) -> dict[str, int]:
        """Get all categories with their line counts."""
        self._ensure_loaded()

        # Get all metadata
        results = self._collection.get()

        categories = {}
        for metadata in results["metadatas"]:
            cat = metadata.get("category", "unknown")
            subcat = metadata.get("subcategory", "")
            key = f"{cat}/{subcat}" if subcat else cat
            categories[key] = categories.get(key, 0) + 1

        return categories

    def clear(self) -> None:
        """Clear all voice lines from the database."""
        self._ensure_loaded()

        # Delete all
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

        self._recently_used.clear()
        logger.info("Cleared all voice lines from database")

    def load_from_yaml(self, yaml_path: str) -> int:
        """
        Load voice lines from a YAML file.

        Args:
            yaml_path: Path to voice_lines.yaml

        Returns:
            Number of lines loaded
        """
        import yaml

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        lines = []

        for category, subcategories in data.get("voice_lines", {}).items():
            for subcategory, entries in subcategories.items():
                for entry in entries:
                    if isinstance(entry, str):
                        # Simple string entry
                        line_id = f"{category}_{subcategory}_{len(lines):03d}"
                        lines.append(VoiceLine(
                            id=line_id,
                            text=entry,
                            category=category,
                            subcategory=subcategory,
                        ))
                    elif isinstance(entry, dict):
                        # Full entry with metadata
                        line_id = entry.get("id", f"{category}_{subcategory}_{len(lines):03d}")
                        lines.append(VoiceLine(
                            id=line_id,
                            text=entry["text"],
                            category=category,
                            subcategory=subcategory,
                            tags=entry.get("tags", []),
                            emotion=entry.get("emotion", "neutral"),
                            audio_file=entry.get("audio_file"),
                            viseme_file=entry.get("viseme_file"),
                        ))

        self.add_lines(lines)
        return len(lines)

    def warmup(self) -> None:
        """Pre-load database and embedding model."""
        self._ensure_loaded()

        # Do a test query to warm up the embedding model
        self._collection.query(
            query_texts=["test warmup query"],
            n_results=1,
        )

        logger.info(f"VoiceLineDB warmup complete: {self.count()} lines loaded")

    def get_info(self) -> dict:
        """Return database information."""
        self._ensure_loaded()

        return {
            "db_path": str(self.db_path),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "total_lines": self.count(),
            "categories": self.get_categories(),
            "recently_used_count": len(self._recently_used),
        }


# Testing function
def test_voice_line_db():
    """Test VoiceLineDB functionality."""
    import tempfile

    # Create temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        db = VoiceLineDB(db_path=tmpdir)

        # Add some test lines
        test_lines = [
            VoiceLine(
                id="greet_001",
                text="Ahoy there, me hearty! Welcome to the spookiest night on the seven seas!",
                category="greetings",
                subcategory="generic",
                tags=["welcome", "ahoy"],
                emotion="happy",
            ),
            VoiceLine(
                id="greet_002",
                text="Arrr, who goes there? A brave soul approaches!",
                category="greetings",
                subcategory="generic",
                tags=["challenge"],
                emotion="curious",
            ),
            VoiceLine(
                id="costume_scary_001",
                text="Shiver me timbers! A fearsome vampire! Don't be bitin' me crew!",
                category="costume_reactions",
                subcategory="scary",
                tags=["vampire", "scary"],
                emotion="surprised",
            ),
            VoiceLine(
                id="costume_generic_001",
                text="Now that be a costume worthy of the high seas!",
                category="costume_reactions",
                subcategory="generic",
                tags=["praise"],
                emotion="impressed",
            ),
            VoiceLine(
                id="fallback_001",
                text="Arrr!",
                category="fallbacks",
                subcategory="interjections",
            ),
        ]

        db.add_lines(test_lines)
        print(f"Added {db.count()} lines")
        print(f"Categories: {db.get_categories()}")

        # Test search
        print("\n--- Search Tests ---")

        query = "That vampire costume is terrifying!"
        results = db.search(query, n_results=3)
        print(f"\nQuery: {query}")
        for r in results:
            print(f"  [{r.similarity_score:.3f}] {r.line.id}: {r.line.text[:50]}...")

        # Test search with fallback
        print("\n--- Fallback Search Test ---")
        result = db.search_with_fallback(
            "What a great zombie costume!",
            category_hint="costume_reactions",
            costume_type="zombie",
        )
        if result:
            print(f"Result: [{result.similarity_score:.3f}] {result.line.text}")

        # Test random from category
        print("\n--- Random Category Test ---")
        line = db.get_random_from_category("greetings")
        if line:
            print(f"Random greeting: {line.text}")

        print("\nDatabase info:", db.get_info())


if __name__ == "__main__":
    test_voice_line_db()
