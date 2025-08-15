#!/usr/bin/env python3
"""
YouTube Video Aggregation Pipeline - Demo Version
Demonstriert die Pipeline-Funktionalität mit Mock-Daten
"""

import json
import time
import random
from datetime import datetime
from typing import List, Dict, Any
import csv

class MockPipelineDemo:
    """Demo-Version der Pipeline mit Mock-Daten"""
    
    def __init__(self):
        self.stats = {
            'total_urls': 0,
            'metadata_fetched': 0,
            'classified_relevant': 0,
            'transcripts_extracted': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'indexed_in_supabase': 0,
            'errors': 0,
            'processing_time': 0.0,
            'error_details': []
        }
    
    def extract_video_id(self, url: str) -> str:
        """Extrahiert Video-ID aus YouTube-URL"""
        import re
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return url.split('=')[-1] if '=' in url else 'unknown'
    
    def generate_mock_metadata(self, video_id: str) -> Dict[str, Any]:
        """Generiert Mock-Metadaten für ein Video"""
        titles = [
            "Introduction to Machine Learning with Python",
            "Advanced AI Techniques for Developers", 
            "Building Neural Networks from Scratch",
            "Deep Learning Best Practices",
            "Computer Vision with TensorFlow",
            "Natural Language Processing Tutorial",
            "Reinforcement Learning Explained",
            "Data Science Project Walkthrough",
            "Python Programming for AI",
            "Modern Web Development with React"
        ]
        
        channels = [
            "TechEdu Channel", "AI Academy", "CodeMaster", "DataScience Pro",
            "ML Tutorials", "Python Guru", "Tech Insights", "Developer Hub"
        ]
        
        return {
            'video_id': video_id,
            'title': random.choice(titles),
            'description': f"Comprehensive tutorial covering advanced topics in AI and machine learning. This video provides practical examples and hands-on coding demonstrations.",
            'channel_id': f"UC{random.randint(100000, 999999)}",
            'channel_title': random.choice(channels),
            'published_at': '2024-01-15T10:00:00Z',
            'duration': random.randint(600, 3600),  # 10-60 minutes
            'view_count': random.randint(1000, 100000),
            'like_count': random.randint(50, 5000),
            'comment_count': random.randint(10, 500),
            'tags': ['AI', 'Machine Learning', 'Python', 'Tutorial', 'Programming'],
            'category_id': 28,  # Science & Technology
            'language': 'en'
        }
    
    def classify_video(self, metadata: Dict[str, Any]) -> tuple:
        """Mock-Klassifizierung basierend auf Titel und Tags"""
        ai_keywords = ['AI', 'machine learning', 'neural', 'deep learning', 'data science', 'python', 'programming']
        
        title_lower = metadata['title'].lower()
        score = 0.0
        
        for keyword in ai_keywords:
            if keyword.lower() in title_lower:
                score += 0.2
        
        # Zusätzliche Punkte für Tags
        for tag in metadata.get('tags', []):
            if tag.lower() in [k.lower() for k in ai_keywords]:
                score += 0.1
        
        score = min(1.0, score)  # Maximal 1.0
        is_relevant = score >= 0.7
        
        return score, is_relevant
    
    def generate_mock_transcript(self, video_id: str) -> str:
        """Generiert Mock-Transkript"""
        transcript_templates = [
            "Welcome to this comprehensive tutorial on machine learning. Today we'll explore the fundamentals of neural networks and how to implement them using Python. Machine learning has revolutionized the way we approach data analysis and prediction tasks. In this video, we'll start with the basics and gradually move to more advanced concepts. First, let's understand what artificial intelligence really means and how it differs from traditional programming approaches.",
            
            "In this session, we're going to dive deep into artificial intelligence and its practical applications. We'll cover everything from basic algorithms to advanced deep learning techniques. The field of AI has grown tremendously in recent years, with applications spanning from computer vision to natural language processing. Let's begin by setting up our development environment and installing the necessary libraries.",
            
            "Data science is becoming increasingly important in today's digital world. In this tutorial, we'll explore how to use Python for data analysis and machine learning. We'll work with real datasets and learn how to clean, process, and analyze data effectively. By the end of this video, you'll have a solid understanding of the data science workflow and be able to apply these techniques to your own projects."
        ]
        
        return random.choice(transcript_templates)
    
    def process_text_and_create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Verarbeitet Text und erstellt Chunks"""
        # Einfache Chunk-Erstellung basierend auf Sätzen
        sentences = text.split('. ')
        chunks = []
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 500:  # Max 500 Zeichen pro Chunk
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append({
                        'chunk_index': chunk_index,
                        'content': current_chunk.strip(),
                        'token_count': len(current_chunk.split()),
                        'start_time': chunk_index * 30,  # Mock-Zeitstempel
                        'end_time': (chunk_index + 1) * 30
                    })
                    chunk_index += 1
                current_chunk = sentence + ". "
        
        # Letzten Chunk hinzufügen
        if current_chunk.strip():
            chunks.append({
                'chunk_index': chunk_index,
                'content': current_chunk.strip(),
                'token_count': len(current_chunk.split()),
                'start_time': chunk_index * 30,
                'end_time': (chunk_index + 1) * 30
            })
        
        return chunks
    
    def generate_mock_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generiert Mock-Embeddings"""
        for chunk in chunks:
            # Mock-Embedding als Liste von 1536 Zufallswerten (wie OpenAI ada-002)
            chunk['embedding'] = [random.uniform(-1, 1) for _ in range(1536)]
        
        return chunks
    
    def simulate_supabase_indexing(self, videos_data: List[Dict], chunks_data: Dict[str, List[Dict]]) -> tuple:
        """Simuliert Supabase-Indexierung"""
        videos_indexed = len(videos_data)
        total_chunks = sum(len(chunks) for chunks in chunks_data.values())
        
        # Simuliere Datenbankoperationen
        time.sleep(0.1)  # Kurze Verzögerung für Realismus
        
        return videos_indexed, total_chunks
    
    def process_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Verarbeitet URLs durch die komplette Mock-Pipeline"""
        start_time = time.time()
        
        self.stats['total_urls'] = len(urls)
        print(f"🚀 Starte Mock-Pipeline für {len(urls)} URLs")
        
        # Schritt 1: Video-IDs extrahieren
        video_ids = [self.extract_video_id(url) for url in urls]
        print(f"📋 Video-IDs extrahiert: {len(video_ids)}")
        
        # Schritt 2: Mock-Metadaten generieren
        videos_metadata = []
        for video_id in video_ids:
            metadata = self.generate_mock_metadata(video_id)
            videos_metadata.append(metadata)
        
        self.stats['metadata_fetched'] = len(videos_metadata)
        print(f"📊 Metadaten generiert: {len(videos_metadata)}")
        
        # Schritt 3: KI-Klassifizierung
        classified_videos = []
        relevant_count = 0
        
        for metadata in videos_metadata:
            score, is_relevant = self.classify_video(metadata)
            classified_videos.append((metadata, score, is_relevant))
            if is_relevant:
                relevant_count += 1
        
        self.stats['classified_relevant'] = relevant_count
        print(f"🤖 Klassifizierung abgeschlossen: {relevant_count}/{len(videos_metadata)} als relevant eingestuft")
        
        # Schritt 4: Mock-Transkripte generieren
        transcripts = {}
        transcript_count = 0
        
        for metadata, score, is_relevant in classified_videos:
            if is_relevant:
                transcript = self.generate_mock_transcript(metadata['video_id'])
                transcripts[metadata['video_id']] = transcript
                transcript_count += 1
        
        self.stats['transcripts_extracted'] = transcript_count
        print(f"📝 Transkripte generiert: {transcript_count}")
        
        # Schritt 5: Text-Verarbeitung und Chunking
        all_chunks = {}
        total_chunks = 0
        
        for video_id, transcript in transcripts.items():
            chunks = self.process_text_and_create_chunks(transcript)
            all_chunks[video_id] = chunks
            total_chunks += len(chunks)
        
        self.stats['chunks_created'] = total_chunks
        print(f"✂️ Text-Chunks erstellt: {total_chunks}")
        
        # Schritt 6: Mock-Embeddings generieren
        embeddings_count = 0
        for video_id, chunks in all_chunks.items():
            embedded_chunks = self.generate_mock_embeddings(chunks)
            all_chunks[video_id] = embedded_chunks
            embeddings_count += len(embedded_chunks)
        
        self.stats['embeddings_generated'] = embeddings_count
        print(f"🧠 Embeddings generiert: {embeddings_count}")
        
        # Schritt 7: Mock-Supabase-Indexierung
        relevant_videos = [metadata for metadata, score, is_relevant in classified_videos if is_relevant]
        videos_indexed, chunks_indexed = self.simulate_supabase_indexing(relevant_videos, all_chunks)
        
        self.stats['indexed_in_supabase'] = videos_indexed
        print(f"💾 Supabase-Indexierung simuliert: {videos_indexed} Videos, {chunks_indexed} Chunks")
        
        # Pipeline-Statistiken finalisieren
        self.stats['processing_time'] = time.time() - start_time
        
        # Detaillierte Ergebnisse für Demo
        detailed_results = {
            'videos': relevant_videos,
            'chunks_sample': list(all_chunks.values())[0][:2] if all_chunks else [],  # Erste 2 Chunks als Beispiel
            'classification_scores': [(v['title'], score) for v, score, _ in classified_videos]
        }
        
        return self.stats, detailed_results

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
        print(f"❌ Fehler beim Laden der CSV-Datei: {e}")
    
    return urls

def save_stats(stats: Dict[str, Any], detailed_results: Dict[str, Any], output_path: str):
    """Speichert Statistiken und Ergebnisse als JSON"""
    output_data = {
        'pipeline_stats': stats,
        'detailed_results': detailed_results,
        'timestamp': datetime.now().isoformat(),
        'demo_mode': True
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

def main():
    """Hauptfunktion für Demo-Pipeline"""
    print("🎬 YouTube Video Aggregation Pipeline - Demo Mode")
    print("=" * 60)
    
    # URLs aus CSV laden
    urls = load_urls_from_csv('youtube_links.csv')
    if not urls:
        print("❌ Keine URLs in CSV-Datei gefunden")
        return
    
    # Nur erste 10 URLs verarbeiten
    urls = urls[:10]
    
    # Pipeline ausführen
    pipeline = MockPipelineDemo()
    stats, detailed_results = pipeline.process_urls(urls)
    
    # Ergebnisse speichern
    save_stats(stats, detailed_results, 'stats.json')
    
    # Zusammenfassung ausgeben
    print("\n" + "=" * 60)
    print("📈 PIPELINE-STATISTIKEN")
    print("=" * 60)
    print(f"📋 Verarbeitete URLs: {stats['total_urls']}")
    print(f"📊 Metadaten abgerufen: {stats['metadata_fetched']}")
    print(f"🤖 Als relevant klassifiziert: {stats['classified_relevant']}")
    print(f"📝 Transkripte extrahiert: {stats['transcripts_extracted']}")
    print(f"✂️ Text-Chunks erstellt: {stats['chunks_created']}")
    print(f"🧠 Embeddings generiert: {stats['embeddings_generated']}")
    print(f"💾 In Supabase indexiert: {stats['indexed_in_supabase']}")
    print(f"⚠️ Fehler: {stats['errors']}")
    print(f"⏱️ Verarbeitungszeit: {stats['processing_time']:.2f} Sekunden")
    
    # Beispiel-Ergebnisse anzeigen
    print("\n" + "=" * 60)
    print("🎯 KLASSIFIZIERUNGS-ERGEBNISSE")
    print("=" * 60)
    for title, score in detailed_results['classification_scores']:
        status = "✅ RELEVANT" if score >= 0.7 else "❌ NICHT RELEVANT"
        print(f"{status} | Score: {score:.2f} | {title}")
    
    if detailed_results['chunks_sample']:
        print("\n" + "=" * 60)
        print("📄 BEISPIEL TEXT-CHUNKS")
        print("=" * 60)
        for i, chunk in enumerate(detailed_results['chunks_sample']):
            print(f"Chunk {i+1}:")
            print(f"  📝 Inhalt: {chunk['content'][:100]}...")
            print(f"  🔢 Token: {chunk['token_count']}")
            print(f"  ⏰ Zeit: {chunk['start_time']}-{chunk['end_time']}s")
            print()
    
    print("\n✅ Demo-Pipeline erfolgreich abgeschlossen!")
    print(f"📊 Statistiken gespeichert in: stats.json")

if __name__ == "__main__":
    main()