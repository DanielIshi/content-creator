# YouTube Video Aggregation Pipeline

Eine vollständige, produktionsbereite Pipeline zur Aggregation und Analyse von YouTube-Video-Daten mit erweiterten Funktionen für Content Creator und Datenanalysten.

## 🚀 Features

- **Vollautomatisierte Datenextraktion** von YouTube-Videos über die YouTube Data API
- **Intelligente Datenverarbeitung** mit Sentiment-Analyse und Engagement-Metriken
- **Flexible Konfiguration** über YAML-Dateien
- **Robuste Fehlerbehandlung** mit Retry-Mechanismen und Logging
- **Skalierbare Architektur** mit modularem Design
- **Demo-Modus** mit Mock-Daten für Testing
- **Umfassende Dokumentation** und Beispiele

## 📋 Inhaltsverzeichnis

- [Installation](#installation)
- [Schnellstart](#schnellstart)
- [Konfiguration](#konfiguration)
- [Verwendung](#verwendung)
- [Projektstruktur](#projektstruktur)
- [Dokumentation](#dokumentation)
- [Beispiele](#beispiele)
- [Beitragen](#beitragen)
- [Lizenz](#lizenz)

## 🛠 Installation

### Voraussetzungen

- Python 3.8+
- YouTube Data API v3 Key
- Internetverbindung

### Setup

1. Repository klonen:
```bash
git clone https://github.com/DanielIshi/content-creator.git
cd content-creator
```

2. Virtuelle Umgebung erstellen:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows
```

3. Dependencies installieren:
```bash
pip install -r requirements.txt
```

4. Konfiguration anpassen:
```bash
cp config/config.yaml config/config_local.yaml
# Bearbeiten Sie config_local.yaml mit Ihren API-Keys
```

## ⚡ Schnellstart

### Demo-Modus (ohne API-Key)

```bash
python src/pipeline_demo.py
```

### Produktions-Modus

```bash
# Konfiguration anpassen
nano config/config.yaml

# Pipeline ausführen
python src/pipeline.py
```

## ⚙️ Konfiguration

Die Pipeline wird über `config/config.yaml` konfiguriert:

```yaml
# YouTube API Einstellungen
youtube_api:
  api_key: "YOUR_API_KEY_HERE"
  max_results: 50
  
# Datenbank Einstellungen
database:
  type: "sqlite"
  path: "data/youtube_data.db"
  
# Verarbeitung
processing:
  enable_sentiment_analysis: true
  batch_size: 100
  retry_attempts: 3
```

Siehe [Konfigurationsdokumentation](docs/configuration.md) für alle verfügbaren Optionen.

## 🎯 Verwendung

### Grundlegende Verwendung

```python
from src.pipeline import YouTubePipeline

# Pipeline initialisieren
pipeline = YouTubePipeline('config/config.yaml')

# URLs verarbeiten
urls = ['https://youtube.com/watch?v=...']
results = pipeline.process_urls(urls)

# Statistiken abrufen
stats = pipeline.get_statistics()
```

### Erweiterte Verwendung

```python
# Custom Konfiguration
config = {
    'youtube_api': {'api_key': 'your_key'},
    'processing': {'enable_sentiment_analysis': True}
}

pipeline = YouTubePipeline(config=config)

# Batch-Verarbeitung
pipeline.process_batch('examples/youtube_links.csv')

# Ergebnisse exportieren
pipeline.export_results('output.json')
```

## 📁 Projektstruktur

```
content-creator/
├── README.md                 # Dieses Dokument
├── requirements.txt          # Python Dependencies
├── .gitignore               # Git Ignore Regeln
├── src/                     # Hauptquellcode
│   ├── __init__.py
│   ├── pipeline.py          # Hauptpipeline (Produktion)
│   └── pipeline_demo.py     # Demo-Version mit Mock-Daten
├── config/                  # Konfigurationsdateien
│   └── config.yaml          # Standard-Konfiguration
├── docs/                    # Dokumentation
│   ├── schema_doc.md        # Datenbankschema
│   ├── modules.md           # Wiederverwendbarkeits-Guide
│   └── PIPELINE_SUMMARY.md  # Projektzusammenfassung
├── examples/                # Beispiele und Testdaten
│   ├── youtube_links.csv    # Beispiel-URLs
│   └── stats.json          # Beispiel-Statistiken
└── data/                   # Datenverzeichnis (wird erstellt)
    └── .gitkeep
```

## 📚 Dokumentation

- **[Pipeline-Übersicht](docs/PIPELINE_SUMMARY.md)** - Vollständige Projektbeschreibung
- **[Datenbankschema](docs/schema_doc.md)** - Detaillierte Schema-Dokumentation
- **[Modulare Architektur](docs/modules.md)** - Wiederverwendbarkeits-Guide
- **[API-Referenz](docs/api_reference.md)** - Vollständige API-Dokumentation

## 🔧 Beispiele

### Einfache Video-Analyse

```python
from src.pipeline import YouTubePipeline

pipeline = YouTubePipeline('config/config.yaml')
result = pipeline.analyze_video('https://youtube.com/watch?v=dQw4w9WgXcQ')

print(f"Titel: {result['title']}")
print(f"Views: {result['view_count']}")
print(f"Sentiment: {result['sentiment_score']}")
```

### Batch-Verarbeitung

```python
# CSV-Datei mit URLs verarbeiten
pipeline.process_csv_file('examples/youtube_links.csv')

# Ergebnisse in verschiedenen Formaten exportieren
pipeline.export_to_json('results.json')
pipeline.export_to_csv('results.csv')
```

## 🤝 Beitragen

Beiträge sind willkommen! Bitte lesen Sie unsere [Beitragsrichtlinien](CONTRIBUTING.md).

1. Fork das Repository
2. Erstellen Sie einen Feature-Branch (`git checkout -b feature/AmazingFeature`)
3. Committen Sie Ihre Änderungen (`git commit -m 'Add some AmazingFeature'`)
4. Push zum Branch (`git push origin feature/AmazingFeature`)
5. Öffnen Sie eine Pull Request

## 📊 Status

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Last Commit](https://img.shields.io/github/last-commit/DanielIshi/content-creator)

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe [LICENSE](LICENSE) für Details.

## 🙏 Danksagungen

- YouTube Data API v3
- Alle Mitwirkenden und Tester
- Open Source Community

## 📞 Support

Bei Fragen oder Problemen:

- Öffnen Sie ein [Issue](https://github.com/DanielIshi/content-creator/issues)
- Kontaktieren Sie uns über [Discussions](https://github.com/DanielIshi/content-creator/discussions)

---

**Entwickelt mit ❤️ für Content Creator und Datenanalysten**