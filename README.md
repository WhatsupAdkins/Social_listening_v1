# AI Powered Social Listening Program

This repository contains Python scripts for analyzing social media comments from various sources (YouTube, Twitch, etc.) using AI-powered sentiment analysis and classification.

## Example Output

### Card System Analysis Dashboard
See `examples/example_dashboard.html` for a sample card system analysis dashboard with English text. This demonstrates:
- Interactive statistics cards
- Visual charts showing sentiment distribution by playtime segments
- Detailed main points with supporting comments
- Organized by playtime segments (Early, Mid, Late, Veteran)

The dashboard uses sample data from `examples/example_data.json` to show the expected output format.

### YouTube Insights Dashboard
See `examples/example_insights_dashboard.html` for a sample YouTube insights dashboard (matching the style of `src/generate_insights_webpage.py`). This demonstrates:
- Interactive filtering by Theme, Sentiment, and Priority
- Bar chart showing sentiment distribution by theme
- Two tabs: "Key Insights" and "Priority Themes"
- Insight cards with badges (module, sentiment, importance)
- Supporting comments for each insight
- Real-time filter updates

![Insights Dashboard Screenshot](examples/example.png)

*Example dashboard showing interactive filters, statistics, and key insights with supporting comments*

The dashboard uses sample data from `examples/example_insights_data.json` to show the expected output format.

### Real Dashboard Examples
Check out `examples/dashboards/` for real dashboard outputs:
- `S3PVE_insights_11.10.html` - PVE mode insights analysis
- `card_system_evaluation_20251205_192807.html` - Card system evaluation dashboard

## Project Structure

```
.
├── src/                    # Python scripts
│   ├── analyze_card_text_ai.py
│   ├── generate_insights_webpage.py
│   ├── theme_analyzer.py
│   ├── theme_insight_generator.py
│   ├── twitch_sentiment.py
│   ├── rag_knowledge.py
│   └── rag_knowledge_enhanced.py
├── data/                  # Sample datasets
│   ├── fragpunk_steam_reviews_20251201_202726.csv
│   └── youtube_comments_processed_20251110_105015.csv
├── examples/               # Example dashboards and sample data
│   ├── example_dashboard.html
│   ├── example_insights_dashboard.html
│   ├── example_data.json
│   ├── example_insights_data.json
│   ├── example.png
│   └── dashboards/        # Real dashboard examples
│       ├── S3PVE_insights_11.10.html
│       └── card_system_evaluation_20251205_192807.html
├── docs/                  # Documentation
│   ├── CHANGELOG.md
│   ├── CONTRIBUTING.md
│   ├── PROJECT_CHECKLIST.md
│   └── PROJECT_STRUCTURE.md
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── .gitignore            # Git ignore rules
```

## Scripts Overview

### 1. `src/analyze_card_text_ai.py`
Analyzes card system text opinions by different playtime segments using AI.
- Uses Claude Haiku 4.5 for card-related content extraction and sentiment classification
- Uses Claude Sonnet 4.5 for theme analysis
- Generates interactive HTML dashboards with visualizations
- **Requires**: `ANTHROPIC_API_KEY` environment variable

### 2. `src/generate_insights_webpage.py`
Generates an interactive HTML webpage to visualize insights.
- Supports filtering by module (theme), sentiment, and priority
- Loads insights from JSON files
- Creates interactive dashboard with charts and filters

### 3. `src/theme_insight_generator.py`
Aggregates comments by category (module), sentiment, and generates theme insights.
- Uses Claude Haiku 4.5 to map comments to candidate themes
- Uses Claude Sonnet 4.5 to reduce and generate insights
- **Requires**: `ANTHROPIC_API_KEY` environment variable

### 4. `src/theme_analyzer.py`
YouTube comments theme-based analyzer with denoising, deduplication, and hierarchical classification.
- Classifies comments into 4 layers: module, sub-module, dimension, and sentiment
- Supports both hierarchical (modules) and legacy (themes) classification structures
- Optional RAG (Retrieval-Augmented Generation) integration for domain knowledge enhancement
- Language detection and near-duplicate removal using embeddings
- **Requires**: `OPENAI_API_KEY` environment variable (for GPT classification)
- **Optional**: RAG system (`rag_knowledge.py` or `rag_knowledge_enhanced.py`) for enhanced classification accuracy
- **Optional Dependencies**: `langdetect`, `sentence-transformers` (for advanced features)

### 5. `src/rag_knowledge.py` & `src/rag_knowledge_enhanced.py`
RAG (Retrieval-Augmented Generation) knowledge base systems for game comments.
- `rag_knowledge.py`: Basic RAG system for FragPunk domain knowledge
- `rag_knowledge_enhanced.py`: Enhanced RAG with hybrid retrieval (semantic + keyword matching)
- Supports both DOCX and JSONL knowledge sources
- **Used by**: `theme_analyzer.py` (optional dependency for enhanced classification)
- **Requires**: `OPENAI_API_KEY` environment variable

### 6. `src/twitch_sentiment.py`
Twitch comment sentiment analysis and marketing funnel classification tool.
- Two-step process: lightweight filtering + OpenAI GPT-4o classification
- Classifies comments into 5 marketing funnel categories
- **Requires**: `OPENAI_API_KEY` environment variable

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Set the following environment variables based on which scripts you want to use:

```bash
# For Anthropic Claude API (analyze_card_text_ai.py, theme_insight_generator.py)
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For OpenAI API (theme_analyzer.py, rag_knowledge.py, twitch_sentiment.py)
export OPENAI_API_KEY="your-openai-api-key"
```

### File Paths

Some scripts have hardcoded paths that you may need to update:

- `src/twitch_sentiment.py`: Update `REPLAY_DIR` and `OUTPUT_DIR` variables
- `src/analyze_card_text_ai.py`: Update the card file path in `main()` function
- `src/theme_analyzer.py`: Update JSONL file paths in the RAG initialization section (lines 96-100) if using RAG
- `src/rag_knowledge.py` / `src/rag_knowledge_enhanced.py`: Update document paths in test sections

## Usage Examples

### Analyze Card System Reviews
```bash
python src/analyze_card_text_ai.py
```

### Generate Theme Insights
```bash
python src/theme_insight_generator.py processed_comments.csv
```

### Analyze YouTube Comments with Theme Classification
```bash
# Basic usage with GPT classification
python src/theme_analyzer.py comments.csv

# With RAG enhancement for better accuracy
python src/theme_analyzer.py comments.csv --use-rag --rag-docx "/path/to/glossary.docx"

# Use single API call mode (faster, cheaper)
python src/theme_analyzer.py comments.csv --single-call

# Keyword-based classification (no API cost)
python src/theme_analyzer.py comments.csv --no-gpt
```

### Generate Insights Webpage
```bash
python src/generate_insights_webpage.py --results-dir insights_results --output dashboard.html
```

### Analyze Twitch Comments
```bash
python src/twitch_sentiment.py
```

## Dependencies

See `requirements.txt` for the complete list. Main dependencies include:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `anthropic` - Claude API client
- `openai` - OpenAI API client
- `plotly` - Interactive visualizations
- `python-docx` - Word document processing
- `tiktoken` - Token counting
- `tqdm` - Progress bars
- `pyyaml` - YAML configuration parsing

### Optional Dependencies

Some scripts have optional dependencies for enhanced features:
- `langdetect` - Language detection (for `theme_analyzer.py`)
- `sentence-transformers` - Near-duplicate detection (for `theme_analyzer.py`)
- `scikit-learn` - Cosine similarity calculations (for `theme_analyzer.py`)

### Script Dependencies

- **`theme_analyzer.py`** optionally uses **`rag_knowledge.py`** or **`rag_knowledge_enhanced.py`** for domain knowledge enhancement. When RAG is enabled, it provides game-specific context to improve classification accuracy. The RAG system retrieves relevant game knowledge (characters, weapons, modes, cosmetics) to help classify comments more accurately.

## Notes

- All scripts have been translated to English
- Some scripts may require specific input file formats (CSV, JSON, DOCX)
- API rate limits may apply - scripts include rate limiting delays
- Output directories are created automatically if they don't exist

## Contributing

Contributions are welcome! Please see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<<<<<<< HEAD
=======
## Changelog

See [docs/CHANGELOG.md](docs/CHANGELOG.md) for a list of changes and version history.
>>>>>>> 25a53cb (Add data folder and reorganize README: move Example Output section to top)

