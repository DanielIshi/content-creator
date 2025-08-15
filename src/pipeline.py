#!/usr/bin/env python3
"""
YouTube Video Aggregation Pipeline
Modulare, parametrisierbare Pipeline für Video-Metadaten, KI-Klassifizierung,
Transkripte, Chunking und Embedding-Indexierung in Supabase.
"""

import os
import sys
import json
import yaml
import csv
import time
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Third-party imports
import openai
from supabase import create_client, Client
from youtube_transcript_api import YouTubeTranscriptApi
import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class VideoMetadata:
    """Datenklasse für YouTube-Video-Metadaten"""
    video_id: str
    title: str
    description: str
    channel_id: str
    channel_title: str
    published_at: str
    duration: int
    view_count: int
    like_count: int
    comment_count: int
    tags: List[str]
    category_id: int
    language: str

@dataclass
class ProcessingStats:
    """Statistiken für Pipeline-Verarbeitung"""
    total_urls: int = 0
    metadata_fetched: int = 0
    classified_relevant: int = 0
    transcripts_extracted: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    indexed_in_supabase: int = 0
    errors: int = 0
    processing_time: float = 0.0
    error_details: List[str] = None

    def __post_init__(self):
        if self.error_details is None:
            self.error_details = []

class YouTubeAPIClient:
    """YouTube API Client für Metadaten-Abruf"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extrahiert Video-ID aus YouTube-URL"""
        import re
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    async def get_video_metadata(self, video_ids: List[str]) -> List[VideoMetadata]:
        """Holt Metadaten für mehrere Videos parallel"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            # Batch-Verarbeitung in 50er-Gruppen (YouTube API Limit)
            for i in range(0, len(video_ids), 50):
                batch = video_ids[i:i+50]
                tasks.append(self._fetch_batch_metadata(session, batch))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_metadata = []
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Fehler beim Abrufen der Metadaten: {result}")
                    continue
                all_metadata.extend(result)
            
            return all_metadata
    
    async def _fetch_batch_metadata(self, session: aiohttp.ClientSession, video_ids: List[str]) -> List[VideoMetadata]:
        """Holt Metadaten für einen Batch von Videos"""
        url = f"{self.base_url}/videos"
        params = {
            'key': self.api_key,
            'id': ','.join(video_ids),
            'part': 'snippet,statistics,contentDetails'
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_video_data(data)
                else:
                    logging.error(f"YouTube API Fehler: {response.status}")
                    return []
        except Exception as e:
            logging.error(f"Fehler beim API-Aufruf: {e}")
            return []
    
    def _parse_video_data(self, data: Dict) -> List[VideoMetadata]:
        """Parst YouTube API Response zu VideoMetadata"""
        videos = []
        
        for item in data.get('items', []):
            try:
                snippet = item['snippet']
                statistics = item['statistics']
                content_details = item['contentDetails']
                
                # Duration von ISO 8601 zu Sekunden konvertieren
                duration = self._parse_duration(content_details.get('duration', 'PT0S'))
                
                video = VideoMetadata(
                    video_id=item['id'],
                    title=snippet.get('title', ''),
                    description=snippet.get('description', ''),
                    channel_id=snippet.get('channelId', ''),
                    channel_title=snippet.get('channelTitle', ''),
                    published_at=snippet.get('publishedAt', ''),
                    duration=duration,
                    view_count=int(statistics.get('viewCount', 0)),
                    like_count=int(statistics.get('likeCount', 0)),
                    comment_count=int(statistics.get('commentCount', 0)),
                    tags=snippet.get('tags', []),
                    category_id=int(snippet.get('categoryId', 0)),
                    language=snippet.get('defaultLanguage', 'unknown')
                )
                videos.append(video)
                
            except Exception as e:
                logging.error(f"Fehler beim Parsen der Video-Daten: {e}")
                continue
        
        return videos
    
    def _parse_duration(self, duration_str: str) -> int:
        """Konvertiert ISO 8601 Duration zu Sekunden"""
        import re
        
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_str)
        
        if not match:
            return 0
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds

class AIClassifier:
    """KI-basierte Klassifizierung für Video-Relevanz"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        
    async def classify_videos(self, videos: List[VideoMetadata]) -> List[Tuple[VideoMetadata, float, bool]]:
        """Klassifiziert Videos parallel"""
        tasks = []
        
        for video in videos:
            tasks.append(self._classify_single_video(video))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        classified_videos = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Klassifizierungsfehler für Video {videos[i].video_id}: {result}")
                # Fallback: niedrige Relevanz
                classified_videos.append((videos[i], 0.1, False))
            else:
                classified_videos.append(result)
        
        return classified_videos
    
    async def _classify_single_video(self, video: VideoMetadata) -> Tuple[VideoMetadata, float, bool]:
        """Klassifiziert ein einzelnes Video"""
        
        # Prompt für KI-Klassifizierung
        prompt = f"""
        Analysiere dieses YouTube-Video und bewerte seine Relevanz für AI/ML/Data Science Themen.
        
        Titel: {video.title}
        Beschreibung: {video.description[:500]}...
        Tags: {', '.join(video.tags[:10])}
        Kanal: {video.channel_title}
        
        Bewerte die Relevanz auf einer Skala von 0.0 bis 1.0:
        - 0.9-1.0: Hochrelevant (direkt AI/ML/Data Science)
        - 0.7-0.8: Relevant (verwandte Themen wie Python, Statistik)
        - 0.5-0.6: Teilweise relevant (allgemeine Tech-Themen)
        - 0.0-0.4: Nicht relevant
        
        Antworte nur mit einer Zahl zwischen 0.0 und 1.0.
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            score = max(0.0, min(1.0, score))  # Clamp zwischen 0 und 1
            
            is_relevant = score >= 0.7
            
            return (video, score, is_relevant)
            
        except Exception as e:
            logging.error(f"Fehler bei KI-Klassifizierung: {e}")
            # Fallback: einfache Keyword-basierte Klassifizierung
            return self._fallback_classification(video)
    
    def _fallback_classification(self, video: VideoMetadata) -> Tuple[VideoMetadata, float, bool]:
        """Fallback-Klassifizierung basierend auf Keywords"""
        ai_keywords = [
            'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'data science', 'python', 'tensorflow',
            'pytorch', 'ai', 'ml', 'nlp', 'computer vision', 'algorithm'
        ]
        
        text_to_check = f"{video.title} {video.description}".lower()
        
        score = 0.0
        for keyword in ai_keywords:
            if keyword in text_to_check:
                score += 0.1
        
        # Zusätzliche Punkte für Tags
        for tag in video.tags:
            if tag.lower() in ai_keywords:
                score += 0.05
        
        score = min(1.0, score)
        is_relevant = score >= 0.7
        
        return (video, score, is_relevant)

class TranscriptExtractor:
    """Extraktor für YouTube-Transkripte"""
    
    def __init__(self):
        self.api = YouTubeTranscriptApi
    
    async def extract_transcripts(self, video_ids: List[str]) -> Dict[str, str]:
        """Extrahiert Transkripte für mehrere Videos parallel"""
        tasks = []
        
        for video_id in video_ids:
            tasks.append(self._extract_single_transcript(video_id))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        transcripts = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.warning(f"Kein Transkript für Video {video_ids[i]}: {result}")
            else:
                transcripts[video_ids[i]] = result
        
        return transcripts
    
    async def _extract_single_transcript(self, video_id: str) -> str:
        """Extrahiert Transkript für ein einzelnes Video"""
        try:
            # Versuche verschiedene Sprachen
            languages = ['de', 'en', 'auto']
            
            for lang in languages:
                try:
                    transcript_list = await asyncio.to_thread(
                        self.api.get_transcript,
                        video_id,
                        languages=[lang]
                    )
                    
                    # Kombiniere alle Textteile
                    full_text = ' '.join([entry['text'] for entry in transcript_list])
                    return full_text
                    
                except Exception:
                    continue
            
            raise Exception("Kein Transkript in unterstützten Sprachen gefunden")
            
        except Exception as e:
            raise Exception(f"Transkript-Extraktion fehlgeschlagen: {e}")

class TextProcessor:
    """Text-Verarbeitung und Chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def process_and_chunk_text(self, text: str, video_id: str) -> List[Dict[str, Any]]:
        """Verarbeitet Text und erstellt Chunks"""
        
        # Text bereinigen
        cleaned_text = self._clean_text(text)
        
        # In Chunks aufteilen
        chunks = self._create_chunks(cleaned_text)
        
        # Chunk-Metadaten hinzufügen
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'video_id': video_id,
                'chunk_index': i,
                'content': chunk,
                'token_count': len(self.tokenizer.encode(chunk)),
                'char_count': len(chunk)
            })
        
        return processed_chunks
    
    def _clean_text(self, text: str) -> str:
        """Bereinigt Text von unerwünschten Zeichen"""
        import re
        
        # Entferne übermäßige Whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Entferne spezielle YouTube-Artefakte
        text = re.sub(r'\[.*?\]', '', text)  # [Musik], [Applaus], etc.
        text = re.sub(r'\(.*?\)', '', text)  # (unverständlich), etc.
        
        return text.strip()
    
    def _create_chunks(self, text: str) -> List[str]:
        """Erstellt überlappende Text-Chunks"""
        
        # Tokenize den Text
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Ende des aktuellen Chunks
            end = min(start + self.chunk_size, len(tokens))
            
            # Chunk-Tokens extrahieren
            chunk_tokens = tokens[start:end]
            
            # Zurück zu Text dekodieren
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Nächster Start mit Überlappung
            start = end - self.chunk_overlap
            
            # Verhindere Endlosschleife
            if start >= end:
                break
        
        return chunks

class EmbeddingGenerator:
    """Generiert Embeddings für Text-Chunks"""
    
    def __init__(self, openai_api_key: str, model: str = "text-embedding-ada-002"):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.batch_size = 100  # OpenAI Batch-Limit
    
    async def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generiert Embeddings für alle Chunks"""
        
        # Verarbeite in Batches
        embedded_chunks = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_embeddings = await self._generate_batch_embeddings(batch)
            embedded_chunks.extend(batch_embeddings)
        
        return embedded_chunks
    
    async def _generate_batch_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generiert Embeddings für einen Batch von Chunks"""
        
        try:
            # Extrahiere Texte
            texts = [chunk['content'] for chunk in chunks]
            
            # API-Aufruf
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                model=self.model,
                input=texts
            )
            
            # Embeddings zu Chunks hinzufügen
            embedded_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_copy = chunk.copy()
                chunk_copy['embedding'] = response.data[i].embedding
                embedded_chunks.append(chunk_copy)
            
            return embedded_chunks
            
        except Exception as e:
            logging.error(f"Fehler bei Embedding-Generierung: {e}")
            # Fallback: leere Embeddings
            for chunk in chunks:
                chunk['embedding'] = [0.0] * 1536  # Ada-002 Dimension
            return chunks

class SupabaseIndexer:
    """Indexiert Daten in Supabase"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
    
    async def index_videos_and_chunks(self, 
                                    videos: List[VideoMetadata], 
                                    chunks: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Indexiert Videos und Chunks in Supabase"""
        
        videos_indexed = await self._index_videos(videos)
        chunks_indexed = await self._index_chunks(chunks)
        
        return videos_indexed, chunks_indexed
    
    async def _index_videos(self, videos: List[VideoMetadata]) -> int:
        """Indexiert Video-Metadaten"""
        
        try:
            # Konvertiere zu Dictionaries
            video_dicts = []
            for video in videos:
                video_dict = {
                    'video_id': video.video_id,
                    'title': video.title,
                    'description': video.description,
                    'channel_id': video.channel_id,
                    'channel_title': video.channel_title,
                    'published_at': video.published_at,
                    'duration': video.duration,
                    'view_count': video.view_count,
                    'like_count': video.like_count,
                    'comment_count': video.comment_count,
                    'tags': video.tags,
                    'category_id': video.category_id,
                    'language': video.language,
                    'indexed_at': datetime.now().isoformat()
                }
                video_dicts.append(video_dict)
            
            # Batch-Insert
            result = await asyncio.to_thread(
                self.client.table('videos').upsert(video_dicts).execute
            )
            
            return len(result.data)
            
        except Exception as e:
            logging.error(f"Fehler beim Indexieren der Videos: {e}")
            return 0
    
    async def _index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Indexiert Text-Chunks mit Embeddings"""
        
        try:
            # Bereite Chunks für Supabase vor
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = {
                    'video_id': chunk['video_id'],
                    'chunk_index': chunk['chunk_index'],
                    'content': chunk['content'],
                    'token_count': chunk['token_count'],
                    'char_count': chunk['char_count'],
                    'embedding': chunk['embedding'],
                    'indexed_at': datetime.now().isoformat()
                }
                chunk_dicts.append(chunk_dict)
            
            # Batch-Insert in kleineren Gruppen (Supabase Limit)
            batch_size = 100
            total_indexed = 0
            
            for i in range(0, len(chunk_dicts), batch_size):
                batch = chunk_dicts[i:i + batch_size]
                
                result = await asyncio.to_thread(
                    self.client.table('video_chunks').upsert(batch).execute
                )
                
                total_indexed += len(result.data)
            
            return total_indexed
            
        except Exception as e:
            logging.error(f"Fehler beim Indexieren der Chunks: {e}")
            return 0

class YouTubePipeline:
    """Hauptpipeline für YouTube-Video-Aggregation"""
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        
        # Konfiguration laden
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            raise ValueError("Entweder config_path oder config muss angegeben werden")
        
        # Logging konfigurieren
        self._setup_logging()
        
        # Komponenten initialisieren
        self.youtube_client = YouTubeAPIClient(self.config['youtube_api']['api_key'])
        self.ai_classifier = AIClassifier(self.config['openai']['api_key'])
        self.transcript_extractor = TranscriptExtractor()
        self.text_processor = TextProcessor(
            chunk_size=self.config['processing']['chunk_size'],
            chunk_overlap=self.config['processing']['chunk_overlap']
        )
        self.embedding_generator = EmbeddingGenerator(self.config['openai']['api_key'])
        self.supabase_indexer = SupabaseIndexer(
            self.config['supabase']['url'],
            self.config['supabase']['service_role_key']
        )
        
        # Statistiken
        self.stats = ProcessingStats()
    
    def _load_config(self, config_path: str) -> Dict:
        """Lädt Konfiguration aus YAML-Datei"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Fehler beim Laden der Konfiguration: {e}")
    
    def _setup_logging(self):
        """Konfiguriert Logging"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('pipeline.log')
            ]
        )
    
    async def process_urls(self, urls: List[str]) -> ProcessingStats:
        """Verarbeitet URLs durch die komplette Pipeline"""
        
        start_time = time.time()
        self.stats = ProcessingStats()
        self.stats.total_urls = len(urls)
        
        logging.info(f"🚀 Starte Pipeline für {len(urls)} URLs")
        
        try:
            # Schritt 1: Video-IDs extrahieren
            video_ids = []
            for url in urls:
                video_id = self.youtube_client.extract_video_id(url)
                if video_id:
                    video_ids.append(video_id)
                else:
                    self.stats.errors += 1
                    self.stats.error_details.append(f"Ungültige URL: {url}")
            
            logging.info(f"📋 Video-IDs extrahiert: {len(video_ids)}")
            
            # Schritt 2: Metadaten abrufen
            videos_metadata = await self.youtube_client.get_video_metadata(video_ids)
            self.stats.metadata_fetched = len(videos_metadata)
            logging.info(f"📊 Metadaten abgerufen: {len(videos_metadata)}")
            
            # Schritt 3: KI-Klassifizierung
            classified_videos = await self.ai_classifier.classify_videos(videos_metadata)
            relevant_videos = [(video, score, relevant) for video, score, relevant in classified_videos if relevant]
            self.stats.classified_relevant = len(relevant_videos)
            logging.info(f"🤖 Klassifizierung: {len(relevant_videos)}/{len(videos_metadata)} als relevant")
            
            # Schritt 4: Transkripte extrahieren
            relevant_video_ids = [video.video_id for video, _, _ in relevant_videos]
            transcripts = await self.transcript_extractor.extract_transcripts(relevant_video_ids)
            self.stats.transcripts_extracted = len(transcripts)
            logging.info(f"📝 Transkripte extrahiert: {len(transcripts)}")
            
            # Schritt 5: Text-Verarbeitung und Chunking
            all_chunks = []
            for video_id, transcript in transcripts.items():
                chunks = self.text_processor.process_and_chunk_text(transcript, video_id)
                all_chunks.extend(chunks)
            
            self.stats.chunks_created = len(all_chunks)
            logging.info(f"✂️ Text-Chunks erstellt: {len(all_chunks)}")
            
            # Schritt 6: Embeddings generieren
            embedded_chunks = await self.embedding_generator.generate_embeddings(all_chunks)
            self.stats.embeddings_generated = len(embedded_chunks)
            logging.info(f"🧠 Embeddings generiert: {len(embedded_chunks)}")
            
            # Schritt 7: In Supabase indexieren
            relevant_video_metadata = [video for video, _, _ in relevant_videos]
            videos_indexed, chunks_indexed = await self.supabase_indexer.index_videos_and_chunks(
                relevant_video_metadata, embedded_chunks
            )
            self.stats.indexed_in_supabase = videos_indexed
            logging.info(f"💾 Supabase-Indexierung: {videos_indexed} Videos, {chunks_indexed} Chunks")
            
        except Exception as e:
            self.stats.errors += 1
            self.stats.error_details.append(str(e))
            logging.error(f"Pipeline-Fehler: {e}")
        
        finally:
            self.stats.processing_time = time.time() - start_time
            logging.info(f"⏱️ Pipeline abgeschlossen in {self.stats.processing_time:.2f} Sekunden")
        
        return self.stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Pipeline-Statistiken zurück"""
        return {
            'total_urls': self.stats.total_urls,
            'metadata_fetched': self.stats.metadata_fetched,
            'classified_relevant': self.stats.classified_relevant,
            'transcripts_extracted': self.stats.transcripts_extracted,
            'chunks_created': self.stats.chunks_created,
            'embeddings_generated': self.stats.embeddings_generated,
            'indexed_in_supabase': self.stats.indexed_in_supabase,
            'errors': self.stats.errors,
            'processing_time': self.stats.processing_time,
            'error_details': self.stats.error_details
        }
    
    def save_statistics(self, output_path: str):
        """Speichert Statistiken als JSON"""
        stats_data = {
            'pipeline_stats': self.get_statistics(),
            'timestamp': datetime.now().isoformat(),
            'config_summary': {
                'chunk_size': self.config['processing']['chunk_size'],
                'chunk_overlap': self.config['processing']['chunk_overlap'],
                'embedding_model': self.config.get('openai', {}).get('embedding_model', 'text-embedding-ada-002')
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)

def load_urls_from_csv(csv_path: str) -> List[str]:
    """Lädt URLs aus CSV-Datei"""
    urls = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'url' in row:
                    urls.append(row['url'])
    except Exception as e:
        logging.error(f"Fehler beim Laden der CSV-Datei: {e}")
    
    return urls

async def main():
    """Hauptfunktion für CLI-Verwendung"""
    parser = argparse.ArgumentParser(description='YouTube Video Aggregation Pipeline')
    parser.add_argument('--config', required=True, help='Pfad zur Konfigurationsdatei')
    parser.add_argument('--urls-csv', help='CSV-Datei mit URLs')
    parser.add_argument('--urls', nargs='+', help='Direkte URL-Angabe')
    parser.add_argument('--output', default='stats.json', help='Output-Datei für Statistiken')
    
    args = parser.parse_args()
    
    # URLs laden
    if args.urls_csv:
        urls = load_urls_from_csv(args.urls_csv)
    elif args.urls:
        urls = args.urls
    else:
        print("❌ Keine URLs angegeben. Verwende --urls-csv oder --urls")
        return
    
    if not urls:
        print("❌ Keine gültigen URLs gefunden")
        return
    
    # Pipeline ausführen
    pipeline = YouTubePipeline(config_path=args.config)
    stats = await pipeline.process_urls(urls)
    
    # Statistiken speichern
    pipeline.save_statistics(args.output)
    
    # Zusammenfassung ausgeben
    print("\n" + "="*60)
    print("📈 PIPELINE-STATISTIKEN")
    print("="*60)
    print(f"📋 Verarbeitete URLs: {stats.total_urls}")
    print(f"📊 Metadaten abgerufen: {stats.metadata_fetched}")
    print(f"🤖 Als relevant klassifiziert: {stats.classified_relevant}")
    print(f"📝 Transkripte extrahiert: {stats.transcripts_extracted}")
    print(f"✂️ Text-Chunks erstellt: {stats.chunks_created}")
    print(f"🧠 Embeddings generiert: {stats.embeddings_generated}")
    print(f"💾 In Supabase indexiert: {stats.indexed_in_supabase}")
    print(f"⚠️ Fehler: {stats.errors}")
    print(f"⏱️ Verarbeitungszeit: {stats.processing_time:.2f} Sekunden")
    
    if stats.error_details:
        print(f"\n🔍 Fehlerdetails:")
        for error in stats.error_details:
            print(f"  - {error}")

if __name__ == "__main__":
    asyncio.run(main())
