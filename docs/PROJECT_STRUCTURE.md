# Project Structure

```
code/
├── README.md                    # Main project documentation
├── CONTRIBUTING.md              # Contribution guidelines
├── CHANGELOG.md                 # Version history
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── .env.example                 # Environment variable template
│
├── analyze_card_text_ai.py     # Card system analysis with AI
├── generate_insights_webpage.py # Interactive insights dashboard generator
├── theme_insight_generator.py   # Theme-based insight generation
├── rag_knowledge.py             # Basic RAG knowledge base
├── rag_knowledge_enhanced.py   # Enhanced RAG with hybrid retrieval
└── twitch_sentiment.py         # Twitch comment sentiment analysis
```

## Directory Structure (Created at Runtime)

```
output/
├── insights/                    # AI analysis results (JSON)
└── dashboards/                  # Generated HTML dashboards

insights_results/                # Theme insight results (JSON)

[Script-specific output directories]
```

## Data Flow

1. **Input**: CSV files with comments/reviews
2. **Processing**: 
   - Filtering and preprocessing
   - AI-powered analysis (Claude/OpenAI)
   - Classification and sentiment analysis
3. **Output**: 
   - JSON files with structured insights
   - HTML dashboards for visualization
   - CSV reports with summaries

## Key Dependencies

- **AI APIs**: Anthropic Claude, OpenAI GPT
- **Data Processing**: pandas, numpy
- **Visualization**: plotly
- **Document Processing**: python-docx
- **Utilities**: tqdm, pyyaml, tiktoken

