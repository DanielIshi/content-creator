# Modulare Architektur und Wiederverwendbarkeits-Guide

## Überblick

Die YouTube Video Aggregation Pipeline ist nach dem Prinzip der modularen Architektur entwickelt. Jede Komponente ist eigenständig, testbar und wiederverwendbar. Dieser Guide erklärt, wie Sie die einzelnen Module nutzen, erweitern und in eigenen Projekten wiederverwenden können.

## Architektur-Prinzipien

### 1. Separation of Concerns
Jedes Modul hat eine klar definierte Verantwortlichkeit:
- **YouTubeAPIClient**: Nur Metadaten-Abruf
- **AIClassifier**: Nur Relevanz-Bewertung
- **TranscriptExtractor**: Nur Transkript-Extraktion
- **TextProcessor**: Nur Text-Verarbeitung
- **EmbeddingGenerator**: Nur Vektor-Generierung
- **SupabaseIndexer**: Nur Datenbank-Operationen

### 2. Dependency Injection
Alle Module sind über Konfiguration steuerbar und können ausgetauscht werden.

### 3. Async-First Design
Alle I/O-Operationen sind asynchron für maximale Performance.

### 4. Error Resilience
Jedes Modul behandelt Fehler graceful mit Fallback-Strategien.

## Modul-Dokumentation

### YouTubeAPIClient

**Zweck**: Abstraktion der YouTube Data API v3

#### Standalone-Verwendung

```python
from src.pipeline import YouTubeAPIClient
import asyncio

async def example_usage():
    client = YouTubeAPIClient("YOUR_API_KEY")
    
    # Einzelne Video-ID extrahieren
    video_id = client.extract_video_id("https://youtube.com/watch?v=dQw4w9WgXcQ")
    
    # Metadaten für mehrere Videos abrufen
    metadata = await client.get_video_metadata([video_id])
    
    for video in metadata:
        print(f"Titel: {video.title}")
        print(f"Views: {video.view_count}")

asyncio.run(example_usage())
```

#### Erweiterungsmöglichkeiten

```python
class CustomYouTubeClient(YouTubeAPIClient):
    """Erweiterte Version mit zusätzlichen Features"""
    
    async def get_channel_videos(self, channel_id: str, max_results: int = 50):
        """Holt alle Videos eines Kanals"""
        # Implementation hier
        pass
    
    async def get_video_comments(self, video_id: str, max_results: int = 100):
        """Holt Kommentare zu einem Video"""
        # Implementation hier
        pass
```

#### Konfiguration

```yaml
youtube_api:
  api_key: "YOUR_KEY"
  max_results: 50
  quota_limit: 10000
  rate_limit: 100  # Requests per minute
```

### AIClassifier

**Zweck**: KI-basierte Inhalts-Klassifizierung

#### Standalone-Verwendung

```python
from src.pipeline import AIClassifier, VideoMetadata
import asyncio

async def classify_content():
    classifier = AIClassifier("YOUR_OPENAI_KEY")
    
    # Mock-Video für Demo
    video = VideoMetadata(
        video_id="test123",
        title="Machine Learning Tutorial",
        description="Learn ML basics...",
        # ... weitere Felder
    )
    
    # Klassifizierung
    results = await classifier.classify_videos([video])
    
    for video, score, is_relevant in results:
        print(f"Video: {video.title}")
        print(f"Score: {score:.2f}")
        print(f"Relevant: {is_relevant}")

asyncio.run(classify_content())
```

#### Custom Classifier

```python
class DomainSpecificClassifier(AIClassifier):
    """Spezialisierter Klassifizierer für bestimmte Domänen"""
    
    def __init__(self, api_key: str, domain: str = "technology"):
        super().__init__(api_key)
        self.domain = domain
        self.domain_keywords = self._load_domain_keywords(domain)
    
    def _load_domain_keywords(self, domain: str) -> List[str]:
        """Lädt domänen-spezifische Keywords"""
        keywords_map = {
            "technology": ["AI", "ML", "programming", "software"],
            "science": ["research", "study", "experiment", "analysis"],
            "education": ["tutorial", "course", "lesson", "learn"]
        }
        return keywords_map.get(domain, [])
    
    async def _classify_single_video(self, video: VideoMetadata):
        """Überschreibt Standard-Klassifizierung"""
        # Custom Prompt für spezifische Domäne
        prompt = f"""
        Bewerte dieses Video für die Domäne '{self.domain}':
        Titel: {video.title}
        Beschreibung: {video.description[:300]}
        
        Relevanz-Score (0.0-1.0):
        """
        
        # Rest der Implementation...
```

### TranscriptExtractor

**Zweck**: Extraktion von Video-Transkripten

#### Standalone-Verwendung

```python
from src.pipeline import TranscriptExtractor
import asyncio

async def extract_transcripts():
    extractor = TranscriptExtractor()
    
    video_ids = ["dQw4w9WgXcQ", "aircAruvnKk"]
    transcripts = await extractor.extract_transcripts(video_ids)
    
    for video_id, transcript in transcripts.items():
        print(f"Video {video_id}: {transcript[:100]}...")

asyncio.run(extract_transcripts())
```

#### Multi-Source Extractor

```python
class MultiSourceTranscriptExtractor(TranscriptExtractor):
    """Erweiterte Version mit mehreren Quellen"""
    
    def __init__(self):
        super().__init__()
        self.sources = ['youtube', 'whisper', 'custom']
    
    async def extract_with_whisper(self, video_url: str) -> str:
        """Fallback mit Whisper AI"""
        # Implementation für Whisper API
        pass
    
    async def _extract_single_transcript(self, video_id: str) -> str:
        """Erweiterte Extraktion mit Fallbacks"""
        # Versuche YouTube API
        try:
            return await super()._extract_single_transcript(video_id)
        except Exception:
            # Fallback zu Whisper
            video_url = f"https://youtube.com/watch?v={video_id}"
            return await self.extract_with_whisper(video_url)
```

### TextProcessor

**Zweck**: Text-Verarbeitung und semantisches Chunking

#### Standalone-Verwendung

```python
from src.pipeline import TextProcessor

processor = TextProcessor(chunk_size=500, chunk_overlap=100)

text = "Your long text here..."
chunks = processor.process_and_chunk_text(text, "video123")

for chunk in chunks:
    print(f"Chunk {chunk['chunk_index']}: {chunk['content'][:50]}...")
    print(f"Tokens: {chunk['token_count']}")
```

#### Custom Text Processor

```python
class SemanticTextProcessor(TextProcessor):
    """Erweiterte Version mit semantischem Chunking"""
    
    def __init__(self, chunk_size: int = 1000, use_semantic_chunking: bool = True):
        super().__init__(chunk_size)
        self.use_semantic_chunking = use_semantic_chunking
        
        if use_semantic_chunking:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
    
    def _create_semantic_chunks(self, text: str) -> List[str]:
        """Erstellt semantisch kohärente Chunks"""
        doc = self.nlp(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sent in doc.sents:
            sent_tokens = len(self.tokenizer.encode(sent.text))
            
            if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sent.text
                current_tokens = sent_tokens
            else:
                current_chunk += " " + sent.text
                current_tokens += sent_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _create_chunks(self, text: str) -> List[str]:
        """Überschreibt Standard-Chunking"""
        if self.use_semantic_chunking:
            return self._create_semantic_chunks(text)
        else:
            return super()._create_chunks(text)
```

### EmbeddingGenerator

**Zweck**: Vektor-Embeddings für semantische Suche

#### Standalone-Verwendung

```python
from src.pipeline import EmbeddingGenerator
import asyncio

async def generate_embeddings():
    generator = EmbeddingGenerator("YOUR_OPENAI_KEY")
    
    chunks = [
        {"content": "Machine learning is...", "video_id": "test1"},
        {"content": "Neural networks are...", "video_id": "test1"}
    ]
    
    embedded_chunks = await generator.generate_embeddings(chunks)
    
    for chunk in embedded_chunks:
        print(f"Content: {chunk['content'][:50]}...")
        print(f"Embedding dimension: {len(chunk['embedding'])}")

asyncio.run(generate_embeddings())
```

#### Multi-Model Generator

```python
class MultiModelEmbeddingGenerator(EmbeddingGenerator):
    """Unterstützt verschiedene Embedding-Modelle"""
    
    def __init__(self, api_key: str, models: List[str] = None):
        super().__init__(api_key)
        self.models = models or ["text-embedding-ada-002", "text-embedding-3-small"]
    
    async def generate_multi_model_embeddings(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """Generiert Embeddings mit mehreren Modellen"""
        results = {}
        
        for model in self.models:
            self.model = model
            embedded_chunks = await self.generate_embeddings(chunks)
            results[model] = embedded_chunks
        
        return results
    
    async def compare_model_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Vergleicht Ähnlichkeit zwischen verschiedenen Modellen"""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        similarities = {}
        
        for model in self.models:
            self.model = model
            
            # Embeddings generieren
            chunks = [{"content": text1}, {"content": text2}]
            embedded = await self.generate_embeddings(chunks)
            
            # Ähnlichkeit berechnen
            emb1 = np.array(embedded[0]['embedding']).reshape(1, -1)
            emb2 = np.array(embedded[1]['embedding']).reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            similarities[model] = float(similarity)
        
        return similarities
```

### SupabaseIndexer

**Zweck**: Datenbank-Operationen und Vektor-Suche

#### Standalone-Verwendung

```python
from src.pipeline import SupabaseIndexer
import asyncio

async def index_data():
    indexer = SupabaseIndexer("SUPABASE_URL", "SUPABASE_KEY")
    
    # Mock-Daten
    videos = [...]  # VideoMetadata Objekte
    chunks = [...]  # Chunks mit Embeddings
    
    videos_indexed, chunks_indexed = await indexer.index_videos_and_chunks(videos, chunks)
    
    print(f"Indexed {videos_indexed} videos and {chunks_indexed} chunks")

asyncio.run(index_data())
```

#### Extended Indexer

```python
class ExtendedSupabaseIndexer(SupabaseIndexer):
    """Erweiterte Version mit zusätzlichen Features"""
    
    async def semantic_search(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Semantische Suche in der Datenbank"""
        try:
            result = await asyncio.to_thread(
                self.client.rpc,
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.7,
                    'match_count': limit
                }
            )
            return result.data
        except Exception as e:
            logging.error(f"Semantic search error: {e}")
            return []
    
    async def get_similar_videos(self, video_id: str, limit: int = 5) -> List[Dict]:
        """Findet ähnliche Videos basierend auf Embeddings"""
        # Hole Embeddings des Referenz-Videos
        chunks = await asyncio.to_thread(
            self.client.table('video_chunks')
            .select('embedding')
            .eq('video_id', video_id)
            .limit(1)
            .execute
        )
        
        if not chunks.data:
            return []
        
        # Verwende erstes Embedding für Suche
        query_embedding = chunks.data[0]['embedding']
        return await self.semantic_search(query_embedding, limit)
    
    async def get_trending_topics(self, days: int = 30) -> List[Dict]:
        """Analysiert Trending Topics"""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        result = await asyncio.to_thread(
            self.client.rpc,
            'get_trending_topics',
            {
                'since_date': cutoff_date.isoformat(),
                'min_videos': 3
            }
        )
        
        return result.data
```

## Pipeline-Komposition

### Custom Pipeline Builder

```python
class PipelineBuilder:
    """Builder Pattern für Custom Pipelines"""
    
    def __init__(self):
        self.components = {}
        self.config = {}
    
    def with_youtube_client(self, api_key: str, **kwargs):
        self.components['youtube'] = YouTubeAPIClient(api_key, **kwargs)
        return self
    
    def with_classifier(self, classifier_class, **kwargs):
        self.components['classifier'] = classifier_class(**kwargs)
        return self
    
    def with_transcript_extractor(self, extractor_class=TranscriptExtractor):
        self.components['transcript'] = extractor_class()
        return self
    
    def with_text_processor(self, processor_class=TextProcessor, **kwargs):
        self.components['text_processor'] = processor_class(**kwargs)
        return self
    
    def with_embedding_generator(self, generator_class=EmbeddingGenerator, **kwargs):
        self.components['embeddings'] = generator_class(**kwargs)
        return self
    
    def with_indexer(self, indexer_class=SupabaseIndexer, **kwargs):
        self.components['indexer'] = indexer_class(**kwargs)
        return self
    
    def build(self) -> 'CustomPipeline':
        return CustomPipeline(self.components, self.config)

# Verwendung
pipeline = (PipelineBuilder()
    .with_youtube_client("API_KEY")
    .with_classifier(DomainSpecificClassifier, domain="technology")
    .with_text_processor(SemanticTextProcessor, use_semantic_chunking=True)
    .with_embedding_generator(MultiModelEmbeddingGenerator)
    .with_indexer(ExtendedSupabaseIndexer)
    .build())
```

### Micro-Pipelines

```python
class MetadataOnlyPipeline:
    """Pipeline nur für Metadaten-Extraktion"""
    
    def __init__(self, youtube_client: YouTubeAPIClient, classifier: AIClassifier):
        self.youtube_client = youtube_client
        self.classifier = classifier
    
    async def process(self, urls: List[str]) -> List[Tuple[VideoMetadata, float, bool]]:
        # Video-IDs extrahieren
        video_ids = [self.youtube_client.extract_video_id(url) for url in urls]
        
        # Metadaten abrufen
        videos = await self.youtube_client.get_video_metadata(video_ids)
        
        # Klassifizieren
        classified = await self.classifier.classify_videos(videos)
        
        return classified

class TranscriptOnlyPipeline:
    """Pipeline nur für Transkript-Verarbeitung"""
    
    def __init__(self, extractor: TranscriptExtractor, processor: TextProcessor):
        self.extractor = extractor
        self.processor = processor
    
    async def process(self, video_ids: List[str]) -> Dict[str, List[Dict]]:
        # Transkripte extrahieren
        transcripts = await self.extractor.extract_transcripts(video_ids)
        
        # Text verarbeiten
        all_chunks = {}
        for video_id, transcript in transcripts.items():
            chunks = self.processor.process_and_chunk_text(transcript, video_id)
            all_chunks[video_id] = chunks
        
        return all_chunks
```

## Testing und Mocking

### Mock-Implementierungen

```python
class MockYouTubeClient(YouTubeAPIClient):
    """Mock für Testing"""
    
    def __init__(self):
        # Kein API-Key erforderlich
        pass
    
    async def get_video_metadata(self, video_ids: List[str]) -> List[VideoMetadata]:
        # Generiere Mock-Daten
        return [self._generate_mock_video(vid) for vid in video_ids]
    
    def _generate_mock_video(self, video_id: str) -> VideoMetadata:
        return VideoMetadata(
            video_id=video_id,
            title=f"Mock Video {video_id}",
            description="Mock description",
            channel_id="mock_channel",
            channel_title="Mock Channel",
            published_at="2024-01-01T00:00:00Z",
            duration=600,
            view_count=1000,
            like_count=100,
            comment_count=10,
            tags=["mock", "test"],
            category_id=28,
            language="en"
        )

class MockClassifier(AIClassifier):
    """Mock Classifier für Testing"""
    
    def __init__(self):
        pass
    
    async def classify_videos(self, videos: List[VideoMetadata]) -> List[Tuple[VideoMetadata, float, bool]]:
        results = []
        for video in videos:
            # Einfache Mock-Logik
            score = 0.8 if "AI" in video.title or "ML" in video.title else 0.3
            is_relevant = score >= 0.7
            results.append((video, score, is_relevant))
        return results
```

### Unit Tests

```python
import pytest
import asyncio
from src.pipeline import YouTubeAPIClient, AIClassifier

class TestYouTubeAPIClient:
    
    @pytest.fixture
    def client(self):
        return MockYouTubeClient()
    
    @pytest.mark.asyncio
    async def test_extract_video_id(self, client):
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = client.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    @pytest.mark.asyncio
    async def test_get_metadata(self, client):
        video_ids = ["test1", "test2"]
        metadata = await client.get_video_metadata(video_ids)
        
        assert len(metadata) == 2
        assert metadata[0].video_id == "test1"
        assert metadata[1].video_id == "test2"

class TestAIClassifier:
    
    @pytest.fixture
    def classifier(self):
        return MockClassifier()
    
    @pytest.mark.asyncio
    async def test_classification(self, classifier):
        from src.pipeline import VideoMetadata
        
        videos = [
            VideoMetadata(
                video_id="ai_video",
                title="AI Tutorial",
                description="Learn AI",
                # ... weitere Mock-Daten
            ),
            VideoMetadata(
                video_id="cooking_video", 
                title="Cooking Recipe",
                description="How to cook",
                # ... weitere Mock-Daten
            )
        ]
        
        results = await classifier.classify_videos(videos)
        
        assert len(results) == 2
        assert results[0][2] == True  # AI video ist relevant
        assert results[1][2] == False  # Cooking video ist nicht relevant
```

## Deployment-Strategien

### Docker-Containerisierung

```dockerfile
# Dockerfile für einzelne Module
FROM python:3.11-slim

WORKDIR /app

# Nur spezifische Dependencies installieren
COPY requirements-minimal.txt .
RUN pip install -r requirements-minimal.txt

# Nur benötigte Module kopieren
COPY src/pipeline.py .
COPY src/__init__.py .

# Modul-spezifischer Entrypoint
CMD ["python", "-m", "src.pipeline"]
```

### Microservice-Architektur

```python
# YouTube Metadata Service
from fastapi import FastAPI
from src.pipeline import YouTubeAPIClient

app = FastAPI()
client = YouTubeAPIClient(os.getenv("YOUTUBE_API_KEY"))

@app.post("/metadata")
async def get_metadata(video_ids: List[str]):
    metadata = await client.get_video_metadata(video_ids)
    return [asdict(video) for video in metadata]

# AI Classification Service
from fastapi import FastAPI
from src.pipeline import AIClassifier

app = FastAPI()
classifier = AIClassifier(os.getenv("OPENAI_API_KEY"))

@app.post("/classify")
async def classify_videos(videos: List[Dict]):
    video_objects = [VideoMetadata(**video) for video in videos]
    results = await classifier.classify_videos(video_objects)
    return [(asdict(video), score, relevant) for video, score, relevant in results]
```

### Serverless Functions

```python
# AWS Lambda Handler für Embedding-Generierung
import json
from src.pipeline import EmbeddingGenerator

def lambda_handler(event, context):
    generator = EmbeddingGenerator(os.environ['OPENAI_API_KEY'])
    
    chunks = json.loads(event['body'])['chunks']
    
    # Synchrone Wrapper für Lambda
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        embedded_chunks = loop.run_until_complete(
            generator.generate_embeddings(chunks)
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps(embedded_chunks)
        }
    finally:
        loop.close()
```

## Performance-Optimierung

### Caching-Strategien

```python
from functools import lru_cache
import redis
import json

class CachedYouTubeClient(YouTubeAPIClient):
    """YouTube Client mit Redis-Caching"""
    
    def __init__(self, api_key: str, redis_url: str = "redis://localhost:6379"):
        super().__init__(api_key)
        self.redis = redis.from_url(redis_url)
        self.cache_ttl = 3600  # 1 Stunde
    
    async def get_video_metadata(self, video_ids: List[str]) -> List[VideoMetadata]:
        # Prüfe Cache
        cached_videos = []
        uncached_ids = []
        
        for video_id in video_ids:
            cache_key = f"video_metadata:{video_id}"
            cached_data = self.redis.get(cache_key)
            
            if cached_data:
                video_dict = json.loads(cached_data)
                cached_videos.append(VideoMetadata(**video_dict))
            else:
                uncached_ids.append(video_id)
        
        # Hole nur uncached Videos
        if uncached_ids:
            new_videos = await super().get_video_metadata(uncached_ids)
            
            # Cache neue Videos
            for video in new_videos:
                cache_key = f"video_metadata:{video.video_id}"
                self.redis.setex(
                    cache_key, 
                    self.cache_ttl, 
                    json.dumps(asdict(video))
                )
            
            cached_videos.extend(new_videos)
        
        return cached_videos
```

### Batch-Optimierung

```python
class BatchOptimizedPipeline:
    """Pipeline mit optimierter Batch-Verarbeitung"""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    async def process_large_dataset(self, urls: List[str]) -> AsyncGenerator[Dict, None]:
        """Verarbeitet große Datensätze in Batches"""
        
        for i in range(0, len(urls), self.batch_size):
            batch = urls[i:i + self.batch_size]
            
            # Verarbeite Batch
            results = await self.process_batch(batch)
            
            # Yield Ergebnisse
            for result in results:
                yield result
            
            # Optional: Pause zwischen Batches
            await asyncio.sleep(0.1)
    
    async def process_batch(self, urls: List[str]) -> List[Dict]:
        """Verarbeitet einen einzelnen Batch"""
        # Implementation hier
        pass
```

## Monitoring und Observability

### Metriken-Integration

```python
from prometheus_client import Counter, Histogram, Gauge
import time

class InstrumentedYouTubeClient(YouTubeAPIClient):
    """YouTube Client mit Prometheus-Metriken"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        
        # Metriken definieren
        self.api_calls = Counter('youtube_api_calls_total', 'Total API calls')
        self.api_errors = Counter('youtube_api_errors_total', 'Total API errors')
        self.api_duration = Histogram('youtube_api_duration_seconds', 'API call duration')
        self.quota_usage = Gauge('youtube_quota_usage', 'Current quota usage')
    
    async def get_video_metadata(self, video_ids: List[str]) -> List[VideoMetadata]:
        start_time = time.time()
        
        try:
            self.api_calls.inc()
            result = await super().get_video_metadata(video_ids)
            
            # Update Quota (geschätzt)
            estimated_quota = len(video_ids) * 1  # 1 Quota pro Video
            self.quota_usage.inc(estimated_quota)
            
            return result
            
        except Exception as e:
            self.api_errors.inc()
            raise
        
        finally:
            duration = time.time() - start_time
            self.api_duration.observe(duration)
```

## Fazit

Die modulare Architektur der YouTube Video Aggregation Pipeline ermöglicht:

1. **Flexibilität**: Einzelne Module können ausgetauscht werden
2. **Testbarkeit**: Jedes Modul kann isoliert getestet werden
3. **Skalierbarkeit**: Module können unabhängig skaliert werden
4. **Wiederverwendbarkeit**: Module können in anderen Projekten genutzt werden
5. **Wartbarkeit**: Änderungen sind auf einzelne Module beschränkt

Durch die Verwendung von Dependency Injection, async/await und klaren Interfaces ist die Pipeline sowohl performant als auch erweiterbar.

---

**Nächste Schritte**: Experimentieren Sie mit den verschiedenen Modulen und erstellen Sie Ihre eigenen Erweiterungen basierend auf Ihren spezifischen Anforderungen.