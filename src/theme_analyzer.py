#!/usr/bin/env python3
"""
YouTube Comments Theme-Based Analyzer
Denoise, deduplicate, and classify comments into predefined business themes
With optional RAG (Retrieval-Augmented Generation) for domain knowledge
"""

import os
import re
import yaml
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Progress bar
from tqdm import tqdm

# Language detection
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except:
    LANGDETECT_AVAILABLE = False
    print("⚠️ langdetect not installed. Install with: pip install langdetect")

# For near-duplicate detection
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Install with: pip install sentence-transformers")

from sklearn.metrics.pairwise import cosine_similarity

# OpenAI for GPT classification
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not installed. Install with: pip install openai")

# RAG for domain knowledge enhancement
try:
    from rag_knowledge_enhanced import FragPunkRAGEnhanced
    RAG_AVAILABLE = True
    RAG_ENHANCED = True
except:
    try:
        from rag_knowledge import FragPunkRAG
        RAG_AVAILABLE = True
        RAG_ENHANCED = False
    except:
        RAG_AVAILABLE = False
        RAG_ENHANCED = False


class ThemeAnalyzer:
    def __init__(self, csv_file, config_file="themes_config.yaml", dedupe_threshold=0.85, use_gpt=True, api_key=None, outdir=None, limit=None, 
                 use_rag=False, rag_docx=None, rag_top_k=2, single_call_mode=False):
        self.csv_file = csv_file
        self.config_file = config_file
        self.dedupe_threshold = dedupe_threshold
        self.use_gpt = use_gpt and OPENAI_AVAILABLE
        self.limit = limit
        self.single_call_mode = single_call_mode  # NEW: Use single API call (faster, cheaper)
        self.outdir = (outdir.rstrip('/') + '/') if outdir else ""
        if self.outdir and not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)
        
        # OpenAI setup
        if self.use_gpt:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not self.api_key:
                print("⚠️ OPENAI_API_KEY not set. Falling back to keyword-based classification.")
                self.use_gpt = False
            else:
                self.openai_client = OpenAI(api_key=self.api_key)
                self.gpt_model = "gpt-4o"
        
        # RAG setup
        self.use_rag = use_rag and self.use_gpt and RAG_AVAILABLE
        self.rag_system = None
        self.rag_top_k = rag_top_k
        
        if self.use_rag:
            # Check for JSONL files first (new format)
            jsonl_files = [
                '/Users/zizhengwan/Desktop/lancers.jsonl',
                '/Users/zizhengwan/Desktop/weapons.jsonl',
                '/Users/zizhengwan/Desktop/modes.jsonl',
                '/Users/zizhengwan/Desktop/skins.jsonl',
                '/Users/zizhengwan/Desktop/fragpunk_cosmetics_combined.jsonl'
            ]
            jsonl_paths = [f for f in jsonl_files if os.path.exists(f)]
            
            if jsonl_paths:
                # Use JSONL files
                rag_type = "Enhanced (Hybrid Retrieval)" if RAG_ENHANCED else "Standard"
                print(f"🔧 Initializing {rag_type} RAG system with {len(jsonl_paths)} JSONL files...")
                try:
                    # For now, use standard FragPunkRAG for JSONL (Enhanced RAG needs updates)
                    from rag_knowledge import FragPunkRAG
                    self.rag_system = FragPunkRAG(
                        jsonl_paths=jsonl_paths,
                        api_key=self.api_key
                    )
                    self.rag_system.build_knowledge_base()
                    print(f"✅ RAG system ready!")
                except Exception as e:
                    print(f"⚠️ RAG initialization failed: {str(e)[:100]}. Continuing without RAG.")
                    self.use_rag = False
            elif rag_docx and os.path.exists(rag_docx):
                # Fallback to DOCX (legacy)
                rag_type = "Enhanced (Hybrid Retrieval)" if RAG_ENHANCED else "Standard"
                print(f"🔧 Initializing {rag_type} RAG system with: {rag_docx}")
                try:
                    if RAG_ENHANCED:
                        self.rag_system = FragPunkRAGEnhanced(
                            docx_path=rag_docx,
                            api_key=self.api_key
                        )
                    else:
                        self.rag_system = FragPunkRAG(
                            docx_path=rag_docx,
                            api_key=self.api_key
                        )
                    self.rag_system.build_knowledge_base()
                    print(f"✅ RAG system ready!")
                except Exception as e:
                    print(f"⚠️ RAG initialization failed: {str(e)[:100]}. Continuing without RAG.")
                    self.use_rag = False
            else:
                print(f"⚠️ RAG files not found. RAG disabled.")
                self.use_rag = False
        
        self.df = None
        self.config = None
        self.preserve_slang = set()
        self.embedding_model = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_config(self):
        """Load theme configuration from YAML"""
        print(f"📋 Loading theme configuration from {self.config_file}...")
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preserve_slang = set(term.lower() for term in self.config.get('preserve_slang', []))
        
        # Support both old 'themes' format and new 'modules' format
        if 'modules' in self.config:
            module_count = len(self.config['modules'])
            submodule_count = sum(len(m.get('sub_modules', {})) for m in self.config['modules'].values())
            print(f"   ✅ Loaded {module_count} modules with {submodule_count} total sub-modules")
        elif 'themes' in self.config:
            theme_count = len(self.config['themes'])
            print(f"   ✅ Loaded {theme_count} themes (legacy format)")
        
        print(f"   ✅ Loaded {len(self.preserve_slang)} slang terms to preserve")
    
    def load_data(self):
        """Load comments from CSV"""
        print(f"\n📂 Loading comments from {self.csv_file}...")
        self.df = pd.read_csv(self.csv_file, lineterminator="\n")
        
        # Apply limit if specified
        if self.limit and self.limit > 0:
            original_count = len(self.df)
            self.df = self.df.head(self.limit)
            print(f"   ⚠️  Limiting to first {self.limit} comments (out of {original_count} total)")
        
        print(f"   ✅ Loaded {len(self.df)} comments")
        
        # Find text column
        text_cols = ["comment_text", "text", "content", "comment", "body"]
        self.text_col = None
        for col in text_cols:
            if col in self.df.columns:
                self.text_col = col
                break
        
        if not self.text_col:
            # Heuristic
            for col in self.df.columns:
                if self.df[col].dtype == object and self.df[col].str.len().mean() > 20:
                    self.text_col = col
                    break
        
        # Find like column
        self.like_col = None
        for col in self.df.columns:
            if 'like' in col.lower():
                self.like_col = col
                break
        
        print(f"   📝 Text column: {self.text_col}")
        print(f"   ❤️ Like column: {self.like_col if self.like_col else 'Not found'}")
    
    def clean_text(self, text: str) -> str:
        """
        Advanced text cleaning while preserving gamer slang
        - Lowercase
        - Strip URLs, mentions, hashtags
        - Remove repeated punctuation
        - Remove pure emoji
        - Keep gamer slang
        """
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        original = text
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (but keep the word)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove repeated punctuation (!! -> !, ??? -> ?)
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Remove pure emoji lines/segments
        # This is a simple approach - remove emoji unicode ranges
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub('', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        if not LANGDETECT_AVAILABLE:
            return 'en'  # Default to English if langdetect not available
        
        if not text:
            return 'en'
        
        text_stripped = text.strip()
        
        # Very short text - assume English for gaming context
        if len(text_stripped) < 3:
            return 'en'
        
        # Check if mostly emoji or special chars
        alpha_chars = sum(c.isalpha() for c in text_stripped)
        if alpha_chars < 3:
            return 'en'  # Gaming context default
        
        try:
            lang = detect(str(text_stripped))
            # Validate the result is a proper language code
            if lang and len(lang) >= 2 and lang != 'no':
                return lang
            return 'en'
        except (LangDetectException, Exception):
            # Most gaming comments are English, default to 'en' if detection fails
            return 'en'
    
    def denoise_and_bucket(self):
        """Apply denoising and language bucketing"""
        print(f"\n🧹 Denoising and bucketing...")
        
        # Clean text
        self.df['_text_original'] = self.df[self.text_col].astype(str)
        print(f"   Cleaning {len(self.df)} comments...")
        tqdm.pandas(desc="   Cleaning text", ncols=100)
        self.df['_text_cleaned'] = self.df['_text_original'].progress_apply(self.clean_text)
        
        # Remove empty after cleaning
        before = len(self.df)
        self.df = self.df[self.df['_text_cleaned'].str.len() >= 3].reset_index(drop=True)
        removed = before - len(self.df)
        print(f"   ✅ Removed {removed} empty/too short comments after cleaning")
        
        # Language detection
        if LANGDETECT_AVAILABLE:
            print(f"   🌐 Detecting languages...")
            tqdm.pandas(desc="   Language detection", ncols=100)
            self.df['_language'] = self.df['_text_cleaned'].progress_apply(self.detect_language)
            
            lang_dist = self.df['_language'].value_counts()
            print(f"   Languages detected:")
            for lang, count in lang_dist.head(5).items():
                pct = count / len(self.df) * 100
                print(f"      {lang}: {count} ({pct:.1f}%)")
            
            # Mark non-English
            self.df['_is_english'] = self.df['_language'] == 'en'
            non_english = (~self.df['_is_english']).sum()
            print(f"   ⚠️ {non_english} non-English comments will be bucketed separately")
        else:
            self.df['_language'] = 'unknown'
            self.df['_is_english'] = True
        
        print(f"   ✅ {len(self.df)} comments after denoising")
    
    def load_embedding_model(self):
        """Load sentence transformer for near-duplicate detection"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("   ⚠️ Skipping near-duplicate detection (sentence-transformers not available)")
            return
        
        print(f"\n🔢 Loading embedding model for duplicate detection...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight
        print(f"   ✅ Model loaded")
    
    def remove_near_duplicates(self):
        """Remove near-duplicate comments using embeddings"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.embedding_model is None:
            print("\n   ⚠️ Skipping duplicate removal")
            self.df['_is_duplicate'] = False
            return
        
        print(f"\n🔍 Removing near-duplicates (threshold={self.dedupe_threshold})...")
        
        # Get embeddings
        texts = self.df['_text_cleaned'].tolist()
        print(f"   Computing embeddings for {len(texts)} comments...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False, batch_size=128)
        
        # Compute pairwise similarity
        print(f"   Computing similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find duplicates
        duplicates = set()
        for i in range(len(similarity_matrix)):
            if i in duplicates:
                continue
            for j in range(i + 1, len(similarity_matrix)):
                if j in duplicates:
                    continue
                if similarity_matrix[i][j] >= self.dedupe_threshold:
                    # Keep the one with higher engagement
                    like_i = self.df.iloc[i][self.like_col] if self.like_col else 0
                    like_j = self.df.iloc[j][self.like_col] if self.like_col else 0
                    
                    if like_i >= like_j:
                        duplicates.add(j)
                    else:
                        duplicates.add(i)
        
        self.df['_is_duplicate'] = False
        self.df.loc[list(duplicates), '_is_duplicate'] = True
        
        print(f"   ✅ Found {len(duplicates)} near-duplicates ({len(duplicates)/len(self.df)*100:.1f}%)")
        print(f"   ✅ Keeping {len(self.df) - len(duplicates)} unique comments")
    
    def classify_with_gpt_single_call(self, text: str, video_title: str = '', parent_comment: str = '', comment_type: str = 'top_level') -> Tuple[str, str, str, str]:
        """Use GPT to classify a comment into 4 layers in ONE API call (faster, cheaper)"""
        max_retries = 3
        retry_delay = 1
        
        # Build context information
        context_info = ""
        if video_title:
            context_info += f"\n\nVIDEO TITLE: {video_title}"
        if parent_comment and comment_type == 'reply':
            context_info += f"\n\nPARENT COMMENT (this is a reply to): {parent_comment}"
        
        # Get RAG context if enabled
        rag_context = ""
        should_use_rag = True
        if self.use_rag and self.rag_system:
            text_lower = text.lower()
            non_game_indicators = [
                'great video', 'nice video', 'good video', 'appreciate it',
                'thanks for the video', 'thank you', 'subscribed', 'subscribe',
                'first', 'early', 'lol', 'lmao', 'haha', 'nice', 'cool',
                'watching', 'viewer', 'content creator', 'youtube',
                'channel', 'stream', 'streamer'
            ]
            if len(text.split()) <= 5:
                for indicator in non_game_indicators:
                    if indicator in text_lower:
                        should_use_rag = False
                        break
        
        if self.use_rag and self.rag_system and should_use_rag:
            try:
                rag_context = self.rag_system.get_context_for_comment(text, top_k=self.rag_top_k)
                if rag_context:
                    rag_context = f"\n\nGAME KNOWLEDGE (FragPunk context):\n{rag_context}\n"
            except Exception as e:
                pass
        
        # Build classification structure description
        structure_desc = "CLASSIFICATION STRUCTURE:\n"
        for module_id, module_data in self.config['modules'].items():
            module_name = module_data.get('name', module_data.get('name_en', ''))
            structure_desc += f"\n{module_id}: {module_name}\n"
            sub_modules = module_data.get('sub_modules', {})
            if sub_modules:
                for sub_id, sub_data in sub_modules.items():
                    sub_name = sub_data.get('name', sub_data.get('name_en', ''))
                    dimensions = sub_data.get('dimensions', [])
                    dim_str = f" (dimensions: {', '.join(dimensions)})" if dimensions else ""
                    structure_desc += f"  └─ {sub_id}: {sub_name}{dim_str}\n"
        
        sentiment_options = self.config.get('sentiment_options', [])
        sentiment_str = ', '.join(sentiment_options)
        
        for attempt in range(max_retries):
            try:
                prompt = f"""Classify this gaming comment into ALL 4 layers in ONE response.

CLASSIFICATION PRIORITY:
1) Comment content (primary)
2) Parent comment context (if reply)
3) GAME KNOWLEDGE from RAG (if available)
4) Video title (lowest, only when unclear)

IMPORTANT RULES:
- Population Health ("dead game", player count) → M2_mode_experience
- Technical/Platform/Network (ping, FPS, bugs, patch notes) → M5_technical_performance
- Main Mode (Shard Clash, ranked, competitive) → M2_mode_experience > Main Mode
- PVE content (clearing levels, stages, missions, co-op) → M2_mode_experience > PVE
- Other Modes (battle rotation, arcade, events) → M2_mode_experience > Others
- Cosmetics/Visual (skins, animations, effects) → M3_monetization > S1_skin
- Battle Pass content → M3_monetization > S2_battle_pass
- General game praise/statements → M1_game_related > S5_other

{structure_desc}

SENTIMENT OPTIONS: {sentiment_str}

SENTIMENT GUIDELINES:
- Positive: Emotional positive words (❤, 😍, 🔥), anticipation, excitement, defending game
- Negative: Emotional negative words, criticism, complaints, stating game problems
- Neutral: Objective discussion WITHOUT emotional words (discussing mechanics, balance, etc.)

{rag_context}

---
COMMENT TO CLASSIFY:
"{text}"
{context_info}
---

OUTPUT FORMAT (JSON only, no explanation):
{{
  "module": "M1_game_related",
  "submodule": "S1_lancer",
  "dimension": "平衡",
  "sentiment": "Neutral"
}}

RULES:
- Use "none" for submodule/dimension if not applicable
- Module/submodule must be valid IDs from the structure above
- Dimension must be from the list (or "none")
- Sentiment must be exactly one of: {sentiment_str}"""

                api_params = {
                    "model": self.gpt_model,
                    "messages": [
                        {"role": "system", "content": "You are a game feedback classifier. Output ONLY valid JSON with the 4 classification layers. No explanations, no markdown code blocks, just the JSON object."},
                        {"role": "user", "content": prompt}
                    ]
                }
                
                if not self.gpt_model.startswith('o1'):
                    api_params["max_tokens"] = 150
                    api_params["temperature"] = 0
                
                response = self.openai_client.chat.completions.create(**api_params)
                result = response.choices[0].message.content.strip()
                
                # Remove markdown code blocks if present
                result = re.sub(r'^```json\s*', '', result)
                result = re.sub(r'^```\s*', '', result)
                result = re.sub(r'\s*```$', '', result)
                result = result.strip()
                
                # Parse JSON
                try:
                    classification = json.loads(result)
                except json.JSONDecodeError as e:
                    print(f"\n   ⚠️ JSON parse error: {str(e)[:50]} for text: '{text[:40]}...'")
                    print(f"   Raw output: {result[:100]}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    raise
                
                # Validate and extract values
                module = classification.get('module', '')
                submodule = classification.get('submodule', 'none')
                dimension = classification.get('dimension', 'none')
                sentiment = classification.get('sentiment', 'none')
                
                # Validate module exists
                if module not in self.config['modules']:
                    print(f"\n   ⚠️ Invalid module '{module}' for text: '{text[:40]}...'")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    # Fallback to keyword matching
                    return self._keyword_match_hierarchical(text)
                
                # Validate submodule if not 'none'
                if submodule != 'none':
                    sub_modules = self.config['modules'][module].get('sub_modules', {})
                    if submodule not in sub_modules:
                        print(f"\n   ⚠️ Invalid submodule '{submodule}' for module '{module}'")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        # Use first submodule as fallback
                        if sub_modules:
                            submodule = list(sub_modules.keys())[0]
                        else:
                            submodule = 'none'
                
                # Validate sentiment
                if sentiment not in sentiment_options and sentiment != 'none':
                    print(f"\n   ⚠️ Invalid sentiment '{sentiment}' (expected one of {sentiment_options})")
                    # Try to match
                    sentiment_lower = sentiment.lower()
                    matched = False
                    for opt in sentiment_options:
                        if opt.lower() in sentiment_lower or sentiment_lower in opt.lower():
                            sentiment = opt
                            matched = True
                            break
                    if not matched:
                        sentiment = 'Neutral' if 'Neutral' in sentiment_options else sentiment_options[0]
                
                return module, submodule, dimension, sentiment
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n   ⚠️ Single-call GPT error: {str(e)[:100]}")
                    # Fall back to keyword matching
                    return self._keyword_match_hierarchical(text)
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
        
        # Final fallback
        return self._keyword_match_hierarchical(text)
    
    def classify_with_gpt5_hierarchical(self, text: str, video_title: str = '', parent_comment: str = '', comment_type: str = 'top_level') -> Tuple[str, str, str, str]:
        """Use GPT to classify a comment into 4 layers: module, sub-module, dimension, sentiment (with optional RAG and context)"""
        max_retries = 3
        retry_delay = 1
        
        # Build context information
        context_info = ""
        if video_title:
            context_info += f"\n\nVIDEO TITLE: {video_title}"
        if parent_comment and comment_type == 'reply':
            context_info += f"\n\nPARENT COMMENT (this is a reply to): {parent_comment}"
        
        # Quick check: Skip RAG for obviously non-game comments
        should_use_rag = True
        if self.use_rag and self.rag_system:
            text_lower = text.lower()
            # Skip RAG for clearly non-game comments
            non_game_indicators = [
                'great video', 'nice video', 'good video', 'appreciate it',
                'thanks for the video', 'thank you', 'subscribed', 'subscribe',
                'first', 'early', 'lol', 'lmao', 'haha', 'nice', 'cool',
                'watching', 'viewer', 'content creator', 'youtube',
                'channel', 'stream', 'streamer'
            ]
            # If comment is very short and contains only non-game indicators, skip RAG
            if len(text.split()) <= 5:
                for indicator in non_game_indicators:
                    if indicator in text_lower:
                        should_use_rag = False
                        break
        
        # Get RAG context if enabled and comment is likely game-related
        rag_context = ""
        if self.use_rag and self.rag_system and should_use_rag:
            try:
                rag_context = self.rag_system.get_context_for_comment(text, top_k=self.rag_top_k)
                if rag_context:
                    rag_context = f"\n\nGAME KNOWLEDGE (FragPunk context):\n{rag_context}\n"
            except Exception as e:
                # Fail silently, continue without RAG for this comment
                pass
        
        # Build module descriptions
        module_desc_lines = []
        for module_id, module_data in self.config['modules'].items():
            name = module_data.get('name', module_data.get('name_en', ''))
            intent = module_data.get('business_intent', '')
            seeds = ', '.join(module_data.get('seed_terms', [])[:5])
            module_desc_lines.append(f"{module_id}: {name} - {intent} (keywords: {seeds})")
        
        for attempt in range(max_retries):
            try:
                # Step 1: Classify into module
                if rag_context:
                    # Enhanced prompt when RAG context is available
                    module_prompt = f"""Classify this gaming comment into ONE module ID from the list below.

IMPORTANT: 
- If PARENT COMMENT is provided, this is a reply. The reply should be classified based on the topic/context of the parent comment, not just the reply text itself.
- Classification priority: 1) Comment content, 2) Parent comment context, 3) GAME KNOWLEDGE, 4) Video title (lowest, only when unclear)

CLASSIFICATION RULES (apply in order):
1. Population Health: Comments about "dead game", "dying", "wish the game wasn't dead", player population, player count, playerbase, community health, comparing player counts to other games → M2 (模式体验) > Main Mode - these are about population health affecting main mode (queue times, matchmaking)
2. Technical/Platform/Network: Comments about ping, latency, hit registration, platform differences (PS5 vs PC), crossplay, FPS, lag, bugs, crashes, performance, patch notes, updates → M5 (技术性能). These are about technical/performance issues.
3. Main Mode Experience: Comments about Shard Clash, ranked mode, competitive mode, matchmaking, queue times, ranked matches, MMR, ladder → M2 (模式体验) > Main Mode
4. PVE Mode Experience: Comments about clearing/completing levels, stages, missions, or PVE content (e.g., "cleared 1-1", "beat nightmare", "completed stage", "finished mission", "co-op mission", difficulty levels like "nightmare", "survival", "investigation", co-op gameplay) → M2 (模式体验) > PVE. These are about PVE gameplay experience, not maps or skins.
5. Other Modes: Comments about battle rotation, arcade, arcade rotation, event modes, limited modes, rotating modes, special events → M2 (模式体验) > Others
6. Cosmetics/Visual Effects: Comments about kill animations, death animations, finishers, visual effects, skins, cosmetics, weapon skins, character appearance, outfit → M3 (Monetization) > 皮肤. ANY visual/cosmetic element including animations and effects.
7. GAME KNOWLEDGE - Entity Type: If GAME KNOWLEDGE identifies mentioned items as cosmetic/skin → M3 (Monetization) > 皮肤. If mentions characters/weapons/abilities → classify by entity type.
8. Activities & Events: Comments about tournaments, events, battle pass, rewards, points, grinding, "recruit friends", "incentivize", progress, tiers, leveling → M3 (Battle Pass) or M4 (活动/赛事)
9. Content Modules: Map/character/mode/skin specific feedback → M1-M4
10. General Game Statements: Broad/general statements about the game without specific topic → M1 (游戏相关) > S5_other (其他游戏相关)
11. Other: ONLY non-game topics (YouTube content, test servers, personal chat unrelated to game, off-topic links)

CONTEXT CLUES:
- "dead game", "dying", "wish the game wasn't dead", "game is dead", "player count", "population", "queue time", "matchmaking" → M2 (模式体验) > Main Mode - population health affects main mode
- "shard clash", "ranked", "competitive", "ranked match", "MMR", "ladder" → M2 (模式体验) > Main Mode
- "pve", "pve mode", "co-op", "mission", "level", "stage", "cleared", "completed", "beat nightmare", "co-op mission" → M2 (模式体验) > PVE
- "battle rotation", "arcade", "arcade rotation", "event mode", "limited mode", "rotation" → M2 (模式体验) > Others
- "currency", "coins", "bundle", "shop", "purchase rewards" → M3 (Monetization)
- "I prefer X" / "I would take X over Y" → M3 (皮肤) if comparing visual items
- "wasted whole day", "getting to tier/level X", "progress", "grind" → M3 (Battle Pass)
- "patch notes", "update", "hotfix", "changelog" → M5 (技术性能)
- "ping", "ms", "hit reg", "PS5 vs PC", "crossplay", platform differences → M5 (技术性能)
- "fps", "lag", "bug", "crash", "performance" → M5 (技术性能)
- General broad game statements → M1 (游戏相关) > S5_other
- "test server", "beta access" → M6 (其他)
- ONLY non-game topics (YouTube, personal chat) → M6 (其他)

MODULES:
{chr(10).join(module_desc_lines)}

{rag_context}

---
COMMENT TO CLASSIFY:
"{text}"
{context_info}
---

Output ONLY the module ID (e.g., M1 or M2 or M7). Nothing else."""
                else:
                    # Original prompt when no RAG
                    module_prompt = f"""Classify this gaming comment into ONE module ID from the list below.

IMPORTANT: 
- If PARENT COMMENT is provided, this is a reply. The reply should be classified based on the topic/context of the parent comment, not just the reply text itself.
- Classification priority: 1) Comment content, 2) Parent comment context, 3) GAME KNOWLEDGE, 4) Video title (lowest, only when unclear)

CLASSIFICATION RULES (apply in order):
1. Population Health: "dead game", "dying", "game is dead", player population, player count, playerbase, comparing player counts → M2 (模式体验) > Main Mode - population health affects main mode
2. Technical/Platform/Network: Ping, latency, hit registration, platform differences (PS5 vs PC), crossplay, FPS, lag, bugs, crashes, performance, patch notes, updates → M5 (技术性能)
3. Main Mode Experience: Shard Clash, ranked mode, competitive mode, matchmaking, queue times, ranked matches, MMR, ladder → M2 (模式体验) > Main Mode
4. PVE Mode Experience: Comments about clearing/completing levels, stages, missions, co-op missions, PVE content, "cleared 1-1", "beat nightmare", "completed stage", co-op gameplay → M2 (模式体验) > PVE
5. Other Modes: Battle rotation, arcade, arcade rotation, event modes, limited modes, rotating modes, special events → M2 (模式体验) > Others
6. Cosmetics/Visual: Kill animations, death animations, finishers, visual effects, skins, cosmetics → M3 (Monetization) > 皮肤
7. GAME KNOWLEDGE - Entity Type: If GAME KNOWLEDGE identifies mentioned items as cosmetic/skin → M3 (Monetization) > 皮肤. If mentions characters/weapons/abilities → classify by entity type.
8. Activities & Events: tournaments, events, battle pass, rewards, grinding, progress → M3 (Battle Pass) or M4 (活动/赛事)
9. Content: Maps, characters, modes, skins → appropriate module (M1-M4)
10. General Game Statements: Broad statements about the game without specific topic → M1 (游戏相关) > S5_other
11. Other: ONLY non-game topics (YouTube content, test servers, personal chat unrelated to game, off-topic links)

CONTEXT CLUES:
- "dead game", "dying", "X has more players", "queue time", "matchmaking" → M2 (模式体验) > Main Mode - population health
- "shard clash", "ranked", "competitive", "ranked match", "MMR" → M2 (模式体验) > Main Mode
- "pve", "pve mode", "co-op", "mission", "level", "stage", "cleared", "completed", "co-op mission" → M2 (模式体验) > PVE
- "battle rotation", "arcade", "arcade rotation", "event mode", "rotation" → M2 (模式体验) > Others
- "currency", "coins", "bundle", "shop", "purchase rewards" → M3 (Monetization)
- "kill animation", "finisher", "effect" → M3 (皮肤)
- "patch notes", "update", "hotfix", "changelog" → M5 (技术性能)
- "ping", "ms", "hit reg", "PS5 vs PC", "crossplay", platform differences → M5 (技术性能)
- "I prefer X" / "take X over Y" → M3 (皮肤) if comparing visual items
- "wasted whole day", "getting to level/tier X", "grind" → M3 (Battle Pass)
- "fps", "lag", "bug", "crash", "performance" → M5 (技术性能)
- General broad game statements → M1 (游戏相关) > S5_other
- "test server", "beta" → M6 (其他)

MODULES:
{chr(10).join(module_desc_lines)}

---
COMMENT TO CLASSIFY:
"{text}"
{context_info}
---

Output ONLY the module ID (e.g., M1 or M2 or M7). Nothing else."""

                api_params = {
                    "model": self.gpt_model,
                    "messages": [
                        {"role": "system", "content": "You are a game feedback classifier. Use the provided game knowledge to identify game entities mentioned in comments. Output ONLY a module ID like M1, M2, M3, etc. No explanations."},
                        {"role": "user", "content": module_prompt}
                    ]
                }
                
                # Add parameters for models that support them (not needed for o1 series)
                if not self.gpt_model.startswith('o1'):
                    api_params["max_tokens"] = 10
                    api_params["temperature"] = 0
                
                response = self.openai_client.chat.completions.create(**api_params)
                module_result = response.choices[0].message.content.strip().upper()
                
                # Extract module ID
                module_match = re.search(r'M(\d+)', module_result)
                if not module_match:
                    if module_result.isdigit():
                        module_id = f"M{module_result}"
                    else:
                        print(f"\n   ⚠️ Invalid module format: '{module_result}' for text: '{text[:40]}...'")
                        raise ValueError(f"Invalid module format: {module_result}")
                else:
                    module_id = f"M{module_match.group(1)}"
                
                # Find the full module key
                module_key = None
                for key in self.config['modules'].keys():
                    if key.upper().startswith(module_id.upper() + '_'):
                        module_key = key
                        break
                
                if not module_key:
                    print(f"\n   ⚠️ Module {module_id} not found for text: '{text[:40]}...'")
                    raise ValueError(f"Module {module_id} not found")
                
                # Step 2: Classify into sub-module (if sub-modules exist)
                sub_modules = self.config['modules'][module_key].get('sub_modules', {})
                
                if not sub_modules:
                    # No sub-modules (e.g., M6_other, M5_technical_performance)
                    # Still need to classify intent even without sub-modules
                    sub_key = 'none'
                    sub_module_data = self.config['modules'][module_key]  # Use module data instead
                    dimension = 'none'
                else:
                    sub_key = None  # Will be set below
                
                # Build sub-module descriptions
                sub_desc_lines = []
                for sub_id, sub_data in sub_modules.items():
                    sub_name = sub_data.get('name', sub_data.get('name_en', ''))
                    sub_seeds = ', '.join(sub_data.get('seed_terms', [])[:5])
                    sub_desc_lines.append(f"{sub_id}: {sub_name} (keywords: {sub_seeds})")
                
                # Build classification rules based on module type
                if module_key == 'M2_mode_experience':
                    # Rules for Mode Experience module
                    classification_rules = """CLASSIFICATION RULES (apply in priority order):
1. Main Mode: Comments about Shard Clash, ranked mode, competitive mode, matchmaking, queue times, ranked matches, MMR, ladder, "dead game" (population health affecting main mode) → S1 (Main Mode)
2. PVE: Comments about PVE mode, co-op, missions, levels, stages, clearing/completing levels, "cleared 1-1", "beat nightmare", "completed stage", co-op missions, co-op gameplay, difficulty levels (nightmare, survival, investigation) → S2 (PVE)
3. Other Modes: Comments about battle rotation, arcade, arcade rotation, event modes, limited modes, rotating modes, special events → S3 (Others)
4. Only if comment context is unclear, use VIDEO TITLE as reference"""
                elif module_key == 'M3_monetization':
                    # Rules for Monetization module
                    classification_rules = """CLASSIFICATION RULES (apply in priority order):
1. If GAME KNOWLEDGE identifies mentioned items as cosmetic/skin/arm_ornament → S1 (皮肤) - RAG has highest priority
2. If comment directly mentions kill animation, death animation, finisher, visual effects, skins, cosmetics, appearance → S1 (皮肤)
3. If comment mentions battle pass, bp, tier, level, pass rewards → S2 (Battle Pass)
4. If comment mentions currency, coins, bundle, shop, store, gacha, pop can, loot box → S3 (其他付费活动)
5. Only if comment context is unclear, use VIDEO TITLE as reference"""
                else:
                    # Generic rules for other modules
                    classification_rules = """CLASSIFICATION RULES (apply in priority order):
1. Match comment content to the most relevant sub-topic based on keywords and context
2. Use GAME KNOWLEDGE (if available) to identify mentioned entities
3. Only if comment context is unclear, use VIDEO TITLE as reference"""

                sub_prompt = f"""The comment is about {self.config['modules'][module_key]['name']}. Now classify it into ONE sub-topic from the list below.

SUB-TOPICS:
{chr(10).join(sub_desc_lines)}

{rag_context}

{classification_rules}

---
COMMENT TO CLASSIFY:
"{text}"
{context_info}
---

Output ONLY the sub-topic ID (e.g., S1 or S2 or S3). Nothing else."""

                api_params["messages"] = [
                    {"role": "system", "content": "You are a classifier. You must output ONLY a sub-topic ID like S1, S2, S3, etc. No explanations, no extra text."},
                    {"role": "user", "content": sub_prompt}
                ]
                
                response = self.openai_client.chat.completions.create(**api_params)
                sub_result = response.choices[0].message.content.strip().upper()
                
                # Extract sub-module ID
                sub_match = re.search(r'S(\d+)', sub_result)
                if not sub_match:
                    if sub_result.isdigit():
                        sub_id = f"S{sub_result}"
                    else:
                        # Default to first sub-module if invalid
                        print(f"\n   ⚠️ Invalid sub-module result: '{sub_result}' for module {module_key}, text: '{text[:40]}...'")
                        sub_id = list(sub_modules.keys())[0]
                else:
                    sub_id = f"S{sub_match.group(1)}"
                
                # Find the full sub-module key
                sub_key = None
                for key in sub_modules.keys():
                    if key.upper().startswith(sub_id.upper() + '_'):
                        sub_key = key
                        break
                
                if not sub_key:
                    if not sub_modules:
                        # Already handled above, skip to intent
                        sub_key = 'none'
                    else:
                        # Default to first sub-module if not found
                        print(f"\n   ⚠️ Sub-module {sub_id} not found in {module_key}, using first available")
                        sub_key = list(sub_modules.keys())[0]
                        sub_module_data = sub_modules[sub_key]
                
                # Step 3: Classify into dimension (if dimensions exist and sub-module exists)
                if sub_key != 'none':
                    sub_module_data = sub_modules[sub_key]
                else:
                    # Use module data when no sub-modules
                    sub_module_data = self.config['modules'][module_key]
                
                dimensions = sub_module_data.get('dimensions', [])
                
                if not dimensions:
                    # No dimensions for this sub-module
                    dimension = 'none'
                else:
                    # Build dimension descriptions
                    dim_desc_lines = []
                    for dim in dimensions:
                        dim_desc_lines.append(f"- {dim}")
                    
                    dim_prompt = f"""The comment is about {sub_module_data.get('name', sub_module_data.get('name_en', ''))}. Now classify which dimension/aspect it focuses on.

NOTE: VIDEO TITLE is only for reference when the comment context is unclear. Focus on the comment content and GAME KNOWLEDGE.

DIMENSIONS:
{chr(10).join(dim_desc_lines)}

{rag_context}

---
COMMENT TO CLASSIFY:
"{text}"
{context_info}
---

Output ONLY the dimension name exactly as shown above (e.g., "外观" or "平衡" or "可玩性"). Nothing else."""

                    api_params["messages"] = [
                        {"role": "system", "content": "You are a classifier. You must output ONLY the dimension name exactly as shown. No explanations, no extra text."},
                        {"role": "user", "content": dim_prompt}
                    ]
                    
                    response = self.openai_client.chat.completions.create(**api_params)
                    dimension_result = response.choices[0].message.content.strip()
                    
                    # Validate dimension exists in list
                    if dimension_result in dimensions:
                        dimension = dimension_result
                    else:
                        # Try to find closest match
                        dimension_lower = dimension_result.lower()
                        matched = False
                        for dim in dimensions:
                            if dim.lower() in dimension_lower or dimension_lower in dim.lower():
                                dimension = dim
                                matched = True
                                break
                        if not matched:
                            # Default to first dimension
                            print(f"\n   ⚠️ Unexpected dimension: '{dimension_result}' (expected one of {dimensions}), using first")
                            dimension = dimensions[0]
                
                # Step 4: Classify into sentiment (use global sentiment_options)
                sentiment_options = self.config.get('sentiment_options', [])
                
                if not sentiment_options:
                    # No sentiment options available at all
                    sentiment = 'none'
                else:
                    # Build sentiment descriptions
                    sentiment_desc_lines = []
                    for sentiment_opt in sentiment_options:
                        sentiment_desc_lines.append(f"- {sentiment_opt}")
                    
                    sentiment_prompt = f"""Classify the sentiment of this comment about {sub_module_data.get('name', sub_module_data.get('name_en', ''))}.

NOTE: VIDEO TITLE is only for reference when the comment context is unclear. Focus on the comment content, GAME KNOWLEDGE, and PARENT COMMENT.

{rag_context}

SENTIMENT OPTIONS:
{chr(10).join(sentiment_desc_lines)}

GUIDELINES:
- Core Principle: Classify based on presence of emotional words and tone. Objective discussion about Lancers, mechanics, or game features WITHOUT emotional words = Neutral. Only classify as Positive/Negative when clear emotional language is present.

- Sentiment Target: Ensure the sentiment is towards the GAME, not towards other players/commenters.

- Reply Context: If this is a reply to a parent comment:
  * If parent comment is Positive, and reply shows agreement/support → Positive
  * If parent comment is Negative, and reply shows agreement → Negative
  * If reply disagrees or has different sentiment, classify based on the reply's own sentiment

- Positive: Positive emotions, anticipation, or defending the game:
  * Positive emotion words (including symbols like ❤, 😍, 🔥)
  * Asking about leaks or upcoming content (shows anticipation)
  * Defending the game or arguing against negative opinions
  * Expressing desire or excitement

- Negative: Negative emotions, critical comparisons, or discussing game problems:
  * Negative emotion words
  * Comparisons showing other games are doing better
  * Discussing game problems, issues, or shortcomings (whether directly stated or implied)
  * Critical statements about lack of updates
  * Quitting with negative emotion (asking questions after quitting = Neutral)

- Neutral: Objective discussion without emotion, anticipation, or criticism:
  * Objective analysis of mechanics, balance, abilities
  * Simple questions without excitement or criticism
  * Factual explanations and corrections
  * Discussing game features objectively without emotional tone

---
COMMENT TO CLASSIFY:
"{text}"
{context_info}
---

Output ONLY the sentiment option exactly as shown above (Positive, Negative, or Neutral). Nothing else."""

                    api_params["messages"] = [
                        {"role": "system", "content": "You are a classifier. You must output ONLY the sentiment option exactly as shown (Positive, Negative, or Neutral). No explanations, no extra text."},
                        {"role": "user", "content": sentiment_prompt}
                    ]
                    
                    response = self.openai_client.chat.completions.create(**api_params)
                    sentiment_result = response.choices[0].message.content.strip()
                    
                    # Validate sentiment exists in list
                    if sentiment_result in sentiment_options:
                        sentiment = sentiment_result
                    else:
                        # Try to find closest match
                        sentiment_lower = sentiment_result.lower()
                        matched = False
                        for sentiment_opt in sentiment_options:
                            if sentiment_opt.lower() in sentiment_lower or sentiment_lower in sentiment_opt.lower():
                                sentiment = sentiment_opt
                                matched = True
                                break
                        if not matched:
                            # Log unexpected result for debugging
                            print(f"\n   ⚠️ Unexpected sentiment result: '{sentiment_result}' for text: '{text[:50]}...'")
                            # Default to Neutral if exists, otherwise first option
                            if "Neutral" in sentiment_options:
                                sentiment = "Neutral"
                            else:
                                sentiment = sentiment_options[0]
                
                return module_key, sub_key, dimension, sentiment
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n   ⚠️  GPT API error: {str(e)[:100]}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    # After all retries failed, fall back to keyword
                    result = self._keyword_match_hierarchical(text)
                    # Return 4 values (module, submodule, dimension, intent)
                    if len(result) == 2:
                        return result[0], result[1], 'none', 'none'
                    return result
        
        result = self._keyword_match_hierarchical(text)
        # Return 4 values (module, submodule, dimension, intent)
        if len(result) == 2:
            return result[0], result[1], 'none', 'none'
        return result
    
    def classify_with_gpt5(self, text: str, batch_themes: str) -> str:
        """Use GPT to classify a comment into a theme (legacy support)"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Simplified, more direct prompt
                prompt = f"""Classify this comment into ONE theme ID from the list below.

THEMES:
{batch_themes}

COMMENT: "{text}"

Output ONLY the theme ID (e.g., T1 or T2 or T12). Nothing else."""

                # Build API parameters
                api_params = {
                    "model": self.gpt_model,
                    "messages": [
                        {"role": "system", "content": "You are a classifier. You must output ONLY a theme ID like T1, T2, T3, etc. No explanations, no extra text."},
                        {"role": "user", "content": prompt}
                    ]
                }
                
                # Add parameters for models that support them (not needed for o1 series)
                if not self.gpt_model.startswith('o1'):
                    api_params["max_tokens"] = 10
                    api_params["temperature"] = 0
                
                response = self.openai_client.chat.completions.create(**api_params)

                result = response.choices[0].message.content.strip().upper()
                
                # Try to extract theme ID - be flexible with format
                theme_match = re.search(r'T(\d+)', result)
                if theme_match:
                    theme_num = theme_match.group(1)
                    theme_id = f"T{theme_num}"
                    
                    # Validate it's a real theme (T1-T12)
                    # Check if any theme key starts with this ID
                    for theme_key in self.config['themes'].keys():
                        if theme_key.upper().startswith(theme_id.upper() + '_'):
                            return theme_key  # Return the full key like T12_misc_small_talk
                
                # If no valid theme found, check if it's just a number
                if result.isdigit():
                    theme_id = f"T{result}"
                    for theme_key in self.config['themes'].keys():
                        if theme_key.upper().startswith(theme_id.upper() + '_'):
                            return theme_key
                
                # Log the issue for debugging (only on last attempt)
                if attempt == max_retries - 1:
                    print(f"\n   ⚠️  GPT returned invalid format: '{result}' for text: '{text[:50]}...'")
                
                # Retry if not last attempt
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                
                # Last resort: use keyword matching
                return self._keyword_match_theme(text)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n   ⚠️  GPT API error: {str(e)[:100]}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    # After all retries failed, fall back to keyword
                    return self._keyword_match_theme(text)
        
        return self._keyword_match_theme(text)
    
    def _keyword_match_hierarchical(self, text: str) -> Tuple[str, str, str, str]:
        """Fallback keyword-based hierarchical matching (returns 4 layers: module, sub-module, dimension, sentiment)"""
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        # First, match module
        module_scores = {}
        
        for module_id, module_data in self.config['modules'].items():
            score = 0
            
            for term in module_data.get('seed_terms', []):
                term_lower = term.lower()
                
                if term_lower in text_lower:
                    if ' ' in term_lower:
                        score += 3
                    else:
                        if term_lower in text_words:
                            score += 2
                        elif term_lower in text_lower:
                            score += 1
            
            if score > 0:
                module_scores[module_id] = score
        
        if not module_scores:
            # Default to "Other" module if exists, otherwise find any module with "other" in name
            other_module = None
            for module_id, module_data in self.config['modules'].items():
                if 'other' in module_id.lower() or 'other' in module_data.get('name', '').lower() or 'other' in module_data.get('name_en', '').lower():
                    other_module = module_id
                    break
            if other_module:
                return other_module, 'none', 'none', 'none'
            # Last resort: use last module in config
            return list(self.config['modules'].keys())[-1], 'none', 'none', 'none'
        
        # Get best module
        best_module = max(module_scores.items(), key=lambda x: x[1])[0]
        
        # Now match sub-module
        sub_modules = self.config['modules'][best_module].get('sub_modules', {})
        
        if not sub_modules:
            return best_module, 'none', 'none', 'none'
        
        sub_scores = {}
        
        for sub_id, sub_data in sub_modules.items():
            score = 0
            
            for term in sub_data.get('seed_terms', []):
                term_lower = term.lower()
                
                if term_lower in text_lower:
                    if ' ' in term_lower:
                        score += 3
                    else:
                        if term_lower in text_words:
                            score += 2
                        elif term_lower in text_lower:
                            score += 1
            
            if score > 0:
                sub_scores[sub_id] = score
        
        if sub_scores:
            best_sub = max(sub_scores.items(), key=lambda x: x[1])[0]
        else:
            # Default to first sub-module
            best_sub = list(sub_modules.keys())[0]
        
        sub_module_data = sub_modules[best_sub]
        
        # Try to match dimension (simple keyword matching)
        dimensions = sub_module_data.get('dimensions', [])
        dimension = 'none'
        if dimensions:
            # Simple keyword matching for dimensions
            dim_keywords = {
                '外观': ['appearance', 'look', 'visual', 'design', 'aesthetic', 'beautiful', 'ugly', 'cool'],
                '平衡': ['balance', 'balanced', 'unbalanced', 'fair', 'unfair', 'op', 'nerf', 'buff'],
                '可玩性': ['playability', 'fun', 'boring', 'interesting', 'depth', 'strategy', 'gameplay'],
                '体验': ['experience', 'feel', 'enjoyable', 'frustrating'],
                '难度': ['difficulty', 'hard', 'easy', 'challenging'],
                '游戏节奏与时长': ['pacing', 'duration', 'length', 'rhythm', 'speed', 'fast', 'slow'],
                '排队时长 (人口健康)': ['queue', 'wait', 'time', 'population', 'player count', 'matchmaking'],
                '外观设计': ['appearance', 'design', 'look', 'visual', 'aesthetic'],
                '购买意愿': ['buy', 'purchase', 'worth', 'price', 'cost', 'value'],
                '肝度': ['grind', 'grindy', 'time consuming', 'tedious'],
                '奖励量': ['reward', 'rewards', 'generous', 'stingy', 'free']
            }
            
            best_dim_score = 0
            for dim in dimensions:
                keywords = dim_keywords.get(dim, [])
                score = sum(1 for kw in keywords if kw in text_lower)
                if score > best_dim_score:
                    best_dim_score = score
                    dimension = dim
        
        # Try to match sentiment (simple keyword matching)
        sentiment_options = self.config.get('sentiment_options', [])
        
        sentiment = 'none'
        if sentiment_options:
            # Simple keyword matching for sentiment
            # Positive keywords: praise, tips, questions, suggestions (unless explicitly negative)
            # Negative keywords: complaints, criticism
            # Neutral: factual statements
            positive_keywords = ['love', 'amazing', 'great', 'best', 'perfect', 'awesome', 'excellent', 'good', 'nice', 'cool', 'fun', 'enjoy', 'appreciate', 'thank', 'tip', 'guide', 'help', 'how to', 'tutorial', 'should', 'could', 'add', 'improve', 'suggestion', 'idea', '?', 'how', 'what', 'when', 'where', 'why', 'can', 'does', 'is it', 'will it']
            negative_keywords = ['hate', 'worst', 'terrible', 'awful', 'bad', 'broken', 'trash', 'garbage', 'sucks', 'disappointing', 'frustrated', 'annoying', 'boring', 'useless', 'waste', 'fix', 'broken', 'unplayable']
            
            positive_score = sum(1 for kw in positive_keywords if kw in text_lower)
            negative_score = sum(1 for kw in negative_keywords if kw in text_lower)
            
            # Check if negative keywords are explicitly negative (e.g., "this is broken" vs "this should be fixed")
            if negative_score > 0:
                # Check for negative context
                negative_context = ['is broken', 'is bad', 'is terrible', 'is awful', 'is worst', 'sucks', 'hate', 'disappointing', 'frustrated']
                if any(ctx in text_lower for ctx in negative_context):
                    sentiment = 'Negative'
                elif 'fix' in text_lower and ('broken' in text_lower or 'bad' in text_lower or 'terrible' in text_lower):
                    sentiment = 'Negative'
                elif negative_score > positive_score:
                    sentiment = 'Negative'
                else:
                    # If positive keywords dominate, it's likely a suggestion/question/tip, which should be Positive
                    sentiment = 'Positive'
            elif positive_score > 0:
                sentiment = 'Positive'
            else:
                sentiment = 'Neutral'
            
            # Validate sentiment exists in options
            if sentiment not in sentiment_options:
                # Default to Neutral if exists, otherwise first option
                if 'Neutral' in sentiment_options:
                    sentiment = 'Neutral'
                else:
                    sentiment = sentiment_options[0]
        
        return best_module, best_sub, dimension, sentiment
    
    def _keyword_match_theme(self, text: str) -> str:
        """Fallback keyword-based theme matching with better logic (legacy support)"""
        text_lower = text.lower()
        text_words = set(text_lower.split())
        theme_scores = {}
        
        for theme_id, theme_data in self.config['themes'].items():
            score = 0
            
            for term in theme_data['seed_terms']:
                term_lower = term.lower()
                
                # Exact phrase match gets higher score
                if term_lower in text_lower:
                    # Multi-word terms get bonus
                    if ' ' in term_lower:
                        score += 3
                    else:
                        # Single word - check if it's a word boundary match
                        if term_lower in text_words:
                            score += 2
                        elif term_lower in text_lower:
                            # Substring match (weaker)
                            score += 1
            
            if score > 0:
                theme_scores[theme_id] = score
        
        if theme_scores:
            # Get theme with highest score
            best_theme = max(theme_scores.items(), key=lambda x: x[1])[0]
            return best_theme
        
        # If very short, default to misc
        if len(text_lower) < 15:
            return 'T12_misc_small_talk'
        
        # For longer comments with no matches, still classify as misc rather than unclassified
        if len(text_lower) < 50:
            return 'T12_misc_small_talk'
        
        return 'unclassified'
    
    def classify_themes(self):
        """Classify comments into business themes (supports both hierarchical and legacy)"""
        print(f"\n🎯 Classifying into business themes...")
        
        # Check if using new hierarchical structure
        use_hierarchical = 'modules' in self.config
        
        if use_hierarchical:
            self._classify_hierarchical()
        else:
            self._classify_legacy()
    
    def _classify_hierarchical(self):
        """Classify using hierarchical structure (modules + sub-modules)"""
        print(f"   Using hierarchical classification (modules + sub-modules)")
        
        if self.use_gpt:
            print(f"   Using GPT for intelligent classification...")
            print(f"   Model: {self.gpt_model}")
            print(f"   Total comments to classify: {len(self.df)}")
            
            # Classify with progress bar and rate limiting
            print(f"\n   🤖 Calling GPT API for {len(self.df)} comments...")
            print(f"   (This may take a few minutes, please be patient)")
            
            module_results = []
            submodule_results = []
            failed_count = 0
            
            # Use tqdm for progress bar
            dimension_results = []
            sentiment_results = []
            
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="   GPT classification", ncols=100):
                # Use original text, not cleaned (to preserve emojis and formatting)
                text = row['_text_original'] if '_text_original' in row else row[self.text_col]
                # Get context information if available
                video_title = row.get('video_title', '') if 'video_title' in self.df.columns else ''
                parent_comment = row.get('parent_comment_text', '') if 'parent_comment_text' in self.df.columns else ''
                comment_type = row.get('comment_type', 'top_level') if 'comment_type' in self.df.columns else 'top_level'
                
                # Choose classification method based on single_call_mode
                if self.single_call_mode:
                    module, submodule, dimension, sentiment = self.classify_with_gpt_single_call(text, video_title, parent_comment, comment_type)
                else:
                    module, submodule, dimension, sentiment = self.classify_with_gpt5_hierarchical(text, video_title, parent_comment, comment_type)
                # Check if module exists in config, if not try to find default
                if module not in self.config['modules']:
                    # Try to find "Other" module
                    other_module = None
                    for module_id in self.config['modules'].keys():
                        if 'other' in module_id.lower():
                            other_module = module_id
                            break
                    if other_module:
                        module = other_module
                        submodule = 'none'
                        dimension = 'none'
                        sentiment = 'none'
                    else:
                        # Use last module as fallback
                        module = list(self.config['modules'].keys())[-1]
                        submodule = 'none'
                        dimension = 'none'
                        sentiment = 'none'
                module_results.append(module)
                submodule_results.append(submodule)
                dimension_results.append(dimension)
                sentiment_results.append(sentiment)
                
                # Rate limiting: small delay every 10 requests
                if (idx + 1) % 10 == 0:
                    time.sleep(0.1)
            
            self.df['_module'] = module_results
            self.df['_submodule'] = submodule_results
            self.df['_dimension'] = dimension_results
            self.df['_sentiment'] = sentiment_results
            
            if failed_count > 0:
                print(f"\n   ⚠️  {failed_count} comments fell back to keyword matching")
        else:
            print(f"   Using keyword-based classification...")
            tqdm.pandas(desc="   Keyword matching", ncols=100)
            results = self.df['_text_cleaned'].progress_apply(self._keyword_match_hierarchical)
            self.df['_module'] = [r[0] if len(r) >= 1 else 'M6_other' for r in results]
            self.df['_submodule'] = [r[1] if len(r) >= 2 else 'none' for r in results]
            self.df['_dimension'] = [r[2] if len(r) >= 3 else 'none' for r in results]
            self.df['_sentiment'] = [r[3] if len(r) >= 4 else 'none' for r in results]
        
        # Add readable names
        module_names = {}
        submodule_names = {}
        dimension_names = {}
        sentiment_names = {}
        
        for module_id, module_data in self.config['modules'].items():
            module_names[module_id] = module_data.get('name', module_data.get('name_en', module_id))
            
            for sub_id, sub_data in module_data.get('sub_modules', {}).items():
                submodule_names[sub_id] = sub_data.get('name', sub_data.get('name_en', sub_id))
                
                # Store dimensions for this sub-module
                dimensions = sub_data.get('dimensions', [])
                for dim in dimensions:
                    dimension_names[dim] = dim
        
        # Get sentiment_options from global config
        sentiment_options = self.config.get('sentiment_options', [])
        for sentiment_opt in sentiment_options:
            sentiment_names[sentiment_opt] = sentiment_opt
        
        submodule_names['none'] = 'N/A'
        dimension_names['none'] = 'N/A'
        sentiment_names['none'] = 'N/A'
        
        self.df['_module_name'] = self.df['_module'].map(module_names)
        self.df['_submodule_name'] = self.df['_submodule'].map(submodule_names)
        self.df['_dimension_name'] = self.df['_dimension'].map(dimension_names)
        self.df['_sentiment_name'] = self.df['_sentiment'].map(sentiment_names)
        
        # Create combined label for easier analysis (4 layers)
        def create_classification(row):
            parts = [row['_module_name']]
            if row['_submodule'] != 'none':
                parts.append(row['_submodule_name'])
            if row['_dimension'] != 'none':
                parts.append(row['_dimension_name'])
            if row['_sentiment'] != 'none':
                parts.append(row['_sentiment_name'])
            return ' > '.join(parts)
        
        self.df['_classification'] = self.df.apply(create_classification, axis=1)
        
        # Report
        print(f"\n   Module distribution:")
        module_dist = self.df['_module'].value_counts()
        for module_id, count in module_dist.items():
            pct = count / len(self.df) * 100
            module_name = module_names.get(module_id, module_id)
            print(f"      {module_name}: {count} ({pct:.1f}%)")
        
        print(f"\n   Top classifications (module > sub-module > dimension > sentiment):")
        classification_dist = self.df['_classification'].value_counts().head(10)
        for classification, count in classification_dist.items():
            pct = count / len(self.df) * 100
            print(f"      {classification}: {count} ({pct:.1f}%)")
        
        # Show dimension distribution
        if '_dimension' in self.df.columns:
            dim_dist = self.df[self.df['_dimension'] != 'none']['_dimension'].value_counts()
            if len(dim_dist) > 0:
                print(f"\n   Dimension distribution:")
                for dim, count in dim_dist.head(5).items():
                    pct = count / len(self.df) * 100
                    print(f"      {dim}: {count} ({pct:.1f}%)")
        
        # Show sentiment distribution
        if '_sentiment' in self.df.columns:
            sentiment_dist = self.df[self.df['_sentiment'] != 'none']['_sentiment'].value_counts()
            if len(sentiment_dist) > 0:
                print(f"\n   Sentiment distribution:")
                for sentiment, count in sentiment_dist.items():
                    pct = count / len(self.df) * 100
                    print(f"      {sentiment}: {count} ({pct:.1f}%)")
    
    def _classify_legacy(self):
        """Classify using legacy theme structure"""
        themes = self.config['themes']
        
        if self.use_gpt:
            print(f"   Using GPT for intelligent classification...")
            print(f"   Model: {self.gpt_model}")
            print(f"   Total comments to classify: {len(self.df)}")
            
            # Build theme description for prompt
            theme_desc_lines = []
            for theme_id, theme_data in themes.items():
                name = theme_data['name']
                intent = theme_data['business_intent']
                seeds = ', '.join(theme_data['seed_terms'][:5])
                theme_desc_lines.append(f"{theme_id}: {name} - {intent} (e.g., {seeds})")
            batch_themes = '\n'.join(theme_desc_lines)
            
            # Classify with progress bar and rate limiting
            print(f"\n   🤖 Calling GPT API for {len(self.df)} comments...")
            print(f"   (This may take a few minutes, please be patient)")
            
            results = []
            failed_count = 0
            
            # Use tqdm for progress bar
            for idx, text in enumerate(tqdm(self.df['_text_cleaned'], desc="   GPT classification", ncols=100)):
                result = self.classify_with_gpt5(text, batch_themes)
                if result == 'unclassified':
                    failed_count += 1
                results.append(result)
                
                # Rate limiting: small delay every 10 requests
                if (idx + 1) % 10 == 0:
                    time.sleep(0.1)
            
            self.df['_theme'] = results
            
            if failed_count > 0:
                print(f"\n   ⚠️  {failed_count} comments fell back to keyword matching")
        else:
            print(f"   Using keyword-based classification...")
            tqdm.pandas(desc="   Keyword matching", ncols=100)
            self.df['_theme'] = self.df['_text_cleaned'].progress_apply(self._keyword_match_theme)
        
        # Add theme names
        theme_names = {tid: tdata['name'] for tid, tdata in themes.items()}
        theme_names['unclassified'] = 'Unclassified'
        self.df['_theme_name'] = self.df['_theme'].map(theme_names)
        
        # Report
        theme_dist = self.df['_theme'].value_counts()
        print(f"\n   Theme distribution:")
        for theme_id, count in theme_dist.items():
            pct = count / len(self.df) * 100
            theme_name = theme_names.get(theme_id, theme_id)
            print(f"      {theme_name}: {count} ({pct:.1f}%)")
    
    def generate_insights(self):
        """Generate comprehensive insights"""
        print(f"\n📊 Generating insights...")
        
        use_hierarchical = 'modules' in self.config
        
        insights = {
            'total_comments_raw': len(self.df),
            'total_comments_cleaned': (~self.df['_is_duplicate']).sum(),
            'duplicates_removed': self.df['_is_duplicate'].sum(),
            'non_english_count': (~self.df['_is_english']).sum() if '_is_english' in self.df.columns else 0,
            'themes': {} if not use_hierarchical else None,
            'modules': {} if use_hierarchical else None,
            'classifications': {} if use_hierarchical else None,
            'languages': {},
        }
        
        # Work with deduplicated data
        df_clean = self.df[~self.df['_is_duplicate']].copy()
        
        if use_hierarchical:
            # Module insights
            insights['modules'] = {}
            for module in df_clean['_module'].unique():
                module_df = df_clean[df_clean['_module'] == module]
                module_name = module_df['_module_name'].iloc[0] if len(module_df) > 0 else module
                
                insights['modules'][module_name] = {
                    'count': len(module_df),
                    'percentage': len(module_df) / len(df_clean) * 100,
                    'avg_likes': module_df[self.like_col].mean() if self.like_col else 0,
                    'total_likes': module_df[self.like_col].sum() if self.like_col else 0,
                }
            
            # Classification insights (module > sub-module)
            insights['classifications'] = {}
            for classification in df_clean['_classification'].unique():
                class_df = df_clean[df_clean['_classification'] == classification]
                
                insights['classifications'][classification] = {
                    'count': len(class_df),
                    'percentage': len(class_df) / len(df_clean) * 100,
                    'avg_likes': class_df[self.like_col].mean() if self.like_col else 0,
                    'total_likes': class_df[self.like_col].sum() if self.like_col else 0,
                }
        else:
            # Theme insights (legacy)
            insights['themes'] = {}
            for theme in df_clean['_theme'].unique():
                theme_df = df_clean[df_clean['_theme'] == theme]
                theme_name = theme_df['_theme_name'].iloc[0] if len(theme_df) > 0 else theme
                
                insights['themes'][theme_name] = {
                    'count': len(theme_df),
                    'percentage': len(theme_df) / len(df_clean) * 100,
                    'avg_likes': theme_df[self.like_col].mean() if self.like_col else 0,
                    'total_likes': theme_df[self.like_col].sum() if self.like_col else 0,
                }
        
        # Language insights
        if '_language' in df_clean.columns:
            for lang in df_clean['_language'].value_counts().head(10).index:
                lang_df = df_clean[df_clean['_language'] == lang]
                insights['languages'][lang] = {
                    'count': len(lang_df),
                    'percentage': len(lang_df) / len(df_clean) * 100,
                }
        
        return insights, df_clean
    
    def save_outputs(self, df_clean):
        """Save processed data and reports"""
        print(f"\n💾 Saving outputs...")
        
        use_hierarchical = 'modules' in self.config
        
        # Reorder columns for better readability
        # Put analysis columns at the end
        original_cols = [col for col in self.df.columns if not col.startswith('_')]
        analysis_cols = [col for col in self.df.columns if col.startswith('_')]
        
        # Make sure all columns exist in df_clean
        all_cols = original_cols + analysis_cols
        missing_in_clean = [col for col in all_cols if col not in df_clean.columns]
        if missing_in_clean:
            print(f"   [WARNING] Columns missing in df_clean: {missing_in_clean}")
            # Use only columns that exist in df_clean
            all_cols = [col for col in all_cols if col in df_clean.columns]
        
        # Create properly ordered dataframes
        df_full_ordered = self.df[all_cols].copy()
        df_clean_ordered = df_clean[all_cols].copy()
        
        # Save full processed dataset (use \r\n for Excel compatibility)
        full_output = f"{self.outdir}comments_processed_{self.timestamp}.csv"
        df_full_ordered.to_csv(full_output, index=False, encoding='utf-8-sig', lineterminator='\r\n')
        print(f"   • {full_output} (all data with flags)")
        
        # Save clean dataset (no duplicates)
        clean_output = f"{self.outdir}comments_clean_{self.timestamp}.csv"
        df_clean_ordered.to_csv(clean_output, index=False, encoding='utf-8-sig', lineterminator='\r\n')
        print(f"   • {clean_output} (deduplicated)")
        
        # Save summary
        if use_hierarchical:
            # Module summary
            if self.like_col:
                module_summary = df_clean.groupby(['_module', '_module_name']).agg({
                    '_text_cleaned': 'count',
                    self.like_col: ['sum', 'mean']
                }).reset_index()
                module_summary.columns = ['module_id', 'module_name', 'comment_count', 'total_likes', 'avg_likes']
            else:
                module_summary = df_clean.groupby(['_module', '_module_name']).agg({
                    '_text_cleaned': 'count'
                }).reset_index()
                module_summary.columns = ['module_id', 'module_name', 'comment_count']
            module_summary = module_summary.sort_values('comment_count', ascending=False)
            
            module_output = f"{self.outdir}module_summary_{self.timestamp}.csv"
            module_summary.to_csv(module_output, index=False, encoding='utf-8-sig', lineterminator='\r\n')
            print(f"   • {module_output} (module summary)")
            
            # Classification summary (module > sub-module > dimension > sentiment)
            groupby_cols = ['_module', '_module_name', '_submodule', '_submodule_name']
            if '_dimension' in df_clean.columns and '_dimension_name' in df_clean.columns:
                groupby_cols.extend(['_dimension', '_dimension_name'])
            if '_sentiment' in df_clean.columns and '_sentiment_name' in df_clean.columns:
                groupby_cols.extend(['_sentiment', '_sentiment_name'])
            groupby_cols.append('_classification')
            
            if self.like_col:
                classification_summary = df_clean.groupby(groupby_cols).agg({
                    '_text_cleaned': 'count',
                    self.like_col: ['sum', 'mean']
                }).reset_index()
                # Build column names
                col_names = ['module_id', 'module_name', 'submodule_id', 'submodule_name']
                if '_dimension' in df_clean.columns:
                    col_names.extend(['dimension', 'dimension_name'])
                if '_sentiment' in df_clean.columns:
                    col_names.extend(['sentiment', 'sentiment_name'])
                col_names.extend(['classification', 'comment_count', 'total_likes', 'avg_likes'])
                classification_summary.columns = col_names
            else:
                classification_summary = df_clean.groupby(groupby_cols).agg({
                    '_text_cleaned': 'count'
                }).reset_index()
                # Build column names
                col_names = ['module_id', 'module_name', 'submodule_id', 'submodule_name']
                if '_dimension' in df_clean.columns:
                    col_names.extend(['dimension', 'dimension_name'])
                if '_sentiment' in df_clean.columns:
                    col_names.extend(['sentiment', 'sentiment_name'])
                col_names.extend(['classification', 'comment_count'])
                classification_summary.columns = col_names
            classification_summary = classification_summary.sort_values('comment_count', ascending=False)
            
            classification_output = f"{self.outdir}classification_summary_{self.timestamp}.csv"
            classification_summary.to_csv(classification_output, index=False, encoding='utf-8-sig', lineterminator='\r\n')
            print(f"   • {classification_output} (classification summary)")
            
            return full_output, clean_output, module_output, classification_output
        else:
            # Theme summary (legacy)
            if self.like_col:
                theme_summary = df_clean.groupby(['_theme', '_theme_name']).agg({
                    '_text_cleaned': 'count',
                    self.like_col: ['sum', 'mean']
                }).reset_index()
                theme_summary.columns = ['theme_id', 'theme_name', 'comment_count', 'total_likes', 'avg_likes']
            else:
                theme_summary = df_clean.groupby(['_theme', '_theme_name']).agg({
                    '_text_cleaned': 'count'
                }).reset_index()
                theme_summary.columns = ['theme_id', 'theme_name', 'comment_count']
            theme_summary = theme_summary.sort_values('comment_count', ascending=False)
            
            theme_output = f"{self.outdir}theme_summary_{self.timestamp}.csv"
            theme_summary.to_csv(theme_output, index=False, encoding='utf-8-sig', lineterminator='\r\n')
            print(f"   • {theme_output} (theme summary)")
            
            return full_output, clean_output, theme_output
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("🎬 YouTube Comments Theme-Based Analysis")
        print("=========================================\n")
        print(f"⚙️ Configuration:")
        if self.use_gpt:
            api_mode = "Single API call (fast, cheap)" if self.single_call_mode else "Multi-step (4 calls, more thorough)"
            print(f"   Classification mode: GPT ({api_mode})")
        else:
            print(f"   Classification mode: Keyword-based")
        print(f"   Model: {self.gpt_model if self.use_gpt else 'N/A'}")
        print(f"   RAG Enhancement: {'✅ Enabled' if self.use_rag else '❌ Disabled'}")
        print(f"   Dedupe threshold: {self.dedupe_threshold}")
        print(f"   Config file: {self.config_file}")
        
        # Load
        self.load_config()
        self.load_data()
        
        # Denoise
        self.denoise_and_bucket()
        
        # Deduplicate
        self.load_embedding_model()
        self.remove_near_duplicates()
        
        # Classify
        self.classify_themes()
        
        # Insights
        insights, df_clean = self.generate_insights()
        
        # Save
        outputs = self.save_outputs(df_clean)
        
        # Print summary
        use_hierarchical = 'modules' in self.config
        
        print("\n" + "="*80)
        print("🎉 ANALYSIS COMPLETE")
        print("="*80)
        print(f"\n📊 Summary:")
        print(f"   Raw comments: {insights['total_comments_raw']:,}")
        print(f"   After deduplication: {insights['total_comments_cleaned']:,}")
        print(f"   Duplicates removed: {insights['duplicates_removed']:,}")
        
        if insights['languages']:
            print(f"\n🌐 Languages detected:")
            for lang, data in sorted(insights['languages'].items(), key=lambda x: x[1]['count'], reverse=True)[:5]:
                lang_name = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'pt': 'Portuguese', 
                            'ja': 'Japanese', 'ko': 'Korean', 'zh-cn': 'Chinese', 'ru': 'Russian'}.get(lang, lang)
                print(f"   {lang_name}: {data['count']:,} ({data['percentage']:.1f}%)")
        
        if use_hierarchical:
            print(f"\n🎯 Top Modules:")
            if insights['modules']:
                top_modules = sorted(insights['modules'].items(), key=lambda x: x[1]['count'], reverse=True)[:5]
                for module_name, data in top_modules:
                    print(f"   {module_name}: {data['count']:,} ({data['percentage']:.1f}%)")
            
            print(f"\n📋 Top Classifications (Module > Sub-module):")
            if insights['classifications']:
                top_classifications = sorted(insights['classifications'].items(), key=lambda x: x[1]['count'], reverse=True)[:10]
                for classification, data in top_classifications:
                    print(f"   {classification}: {data['count']:,} ({data['percentage']:.1f}%)")
        else:
            print(f"\n🎯 Top Themes:")
            if insights['themes']:
                top_themes = sorted(insights['themes'].items(), key=lambda x: x[1]['count'], reverse=True)[:5]
                for theme_name, data in top_themes:
                    print(f"   {theme_name}: {data['count']:,} ({data['percentage']:.1f}%)")
        
        print(f"\n📁 Output files:")
        for output_file in outputs:
            print(f"   • {output_file}")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Theme-based analysis of YouTube comments with GPT classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use GPT for classification (recommended)
  python theme_analyzer.py comments.csv
  
  # Use keyword-based (faster, no API cost)
  python theme_analyzer.py comments.csv --no-gpt
  
  # Custom settings
  python theme_analyzer.py comments.csv --dedupe-threshold 0.90 --outdir results/
  
  # Test with first 100 comments
  python theme_analyzer.py comments.csv --limit 100
  
  # Use RAG for domain knowledge enhancement
  python theme_analyzer.py comments.csv --use-rag --rag-docx "/path/to/glossary.docx"
        """
    )
    ap.add_argument("csv_file", nargs="?", help="Path to comments CSV")
    ap.add_argument("--config", type=str, default="themes_config.yaml", help="Theme configuration YAML")
    ap.add_argument("--dedupe-threshold", type=float, default=0.85, help="Cosine similarity threshold for near-duplicates (default: 0.85)")
    ap.add_argument("--no-gpt", action="store_true", help="Disable GPT, use keyword-based classification")
    ap.add_argument("--api-key", type=str, default=None, help="OpenAI API key (or use OPENAI_API_KEY env var)")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of comments to process (for testing)")
    ap.add_argument("--use-rag", action="store_true", help="Enable RAG (Retrieval-Augmented Generation) for domain knowledge")
    ap.add_argument("--rag-docx", type=str, default=None, help="Path to domain knowledge docx file for RAG")
    ap.add_argument("--rag-top-k", type=int, default=2, help="Number of context chunks to retrieve for RAG (default: 2)")
    ap.add_argument("--single-call", action="store_true", help="Use single API call per comment (faster, cheaper, ~75%% cost reduction)")
    args = ap.parse_args()
    
    if not args.csv_file:
        import glob
        csv_files = [f for f in glob.glob("youtube_comments_*.csv") if "analyzed" not in f and "processed" not in f and "clean" not in f]
        if csv_files:
            csv_files.sort(reverse=True)
            args.csv_file = csv_files[0]
            print(f"📂 Using most recent CSV: {args.csv_file}\n")
        else:
            print("❌ No CSV file found!")
            return
    
    analyzer = ThemeAnalyzer(
        args.csv_file,
        config_file=args.config,
        dedupe_threshold=args.dedupe_threshold,
        use_gpt=not args.no_gpt,
        api_key=args.api_key,
        outdir=args.outdir,
        limit=args.limit,
        use_rag=args.use_rag,
        rag_docx=args.rag_docx,
        rag_top_k=args.rag_top_k,
        single_call_mode=args.single_call
    )
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()

