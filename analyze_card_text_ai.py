#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze card system text opinions by different playtime segments using AI
- Use Haiku to classify card-related positive/negative sentiment for each comment
- Use Sonnet for theme analysis
Requires ANTHROPIC_API_KEY environment variable
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import os
import re
import html
from collections import Counter
import time


def get_api_key_from_file(file_path, pattern):
    """Extract API key from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(pattern, content)
            if match:
                return match.group(1)
    except:
        pass
    return None

def load_data(card_file: str):
    """Load data"""
    print('Loading card system reviews...')
    df = pd.read_csv(card_file, encoding='utf-8-sig')
    print(f'Loaded {len(df):,} card-related reviews')
    return df


def extract_card_content_with_haiku(reviews_text, api_key, batch_size=20):
    """Use Haiku to extract card-related paragraphs from all reviews (process all reviews, no sampling)"""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        model = "claude-haiku-4-5-20251001"
        
        all_extracts = []
        
        # Process all reviews in batches
        total_batches = (len(reviews_text) + batch_size - 1) // batch_size
        print(f"    Processing {len(reviews_text)} reviews in {total_batches} batches...")
        
        for i in range(0, len(reviews_text), batch_size):
            batch = reviews_text[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"    Batch {batch_num}/{total_batches} ({len(batch)} reviews)...", end=' ', flush=True)
            
            reviews_str = "\n\n---\n\n".join([f"Review {j+1}: {review}" for j, review in enumerate(batch)])
            
            prompt = f"""You are a game review analysis assistant. You have a set of game reviews, and you need to extract paragraphs related to the "card system" from each review.

**Task:**
1. Extract only sentences or paragraphs related to cards, decks, card drawing, card abilities, card balance, etc. from each review
2. If a review doesn't mention the card system, return an empty string
3. Only extract card-related content, ignore other game aspects (graphics, sound, matchmaking, etc.)

Review content:
{reviews_str}

Return in JSON format as follows:
{{
  "results": [
    {{"index": 1, "card_content": "extracted card-related content (or empty string if none)"}},
    {{"index": 2, "card_content": "extracted card-related content (or empty string if none)"}},
    ...
  ]
}}

Return only JSON, no other explanations."""
            
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                text = response.content[0].text.strip()
                
                # Clean JSON
                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                        if text.lstrip().startswith("json"):
                            text = text.lstrip()[4:]
                    text = text.strip()
                
                result = json.loads(text)
                batch_extracts = result.get("results", [])
                
                # Sort by index
                batch_extracts.sort(key=lambda x: x.get("index", 0))
                
                all_extracts.extend(batch_extracts)
                
                # Count extraction results
                extracted_count = sum(1 for e in batch_extracts if e.get("card_content", "").strip())
                print(f"✓ (extracted {extracted_count}/{len(batch)} card-related contents)")
                
                # Avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"✗ Error: {e}")
                # If error, add empty results
                for j in range(len(batch)):
                    all_extracts.append({
                        "index": i + j + 1,
                        "card_content": ""
                    })
        
        return all_extracts
        
    except Exception as e:
        print(f"⚠️  Error in Haiku extraction: {e}")
        return None


def classify_extracted_content_with_haiku(card_contents, api_key, batch_size=50):
    """Use Haiku to classify extracted card content as positive or negative"""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        model = "claude-haiku-4-5-20251001"
        
        # Filter out empty content
        non_empty = [(i, content) for i, content in enumerate(card_contents) if content and content.strip()]
        
        if len(non_empty) == 0:
            return []
        
        all_classifications = []
        
        # Process in batches
        total_batches = (len(non_empty) + batch_size - 1) // batch_size
        print(f"    Classifying {len(non_empty)} card-related contents in {total_batches} batches...")
        
        for i in range(0, len(non_empty), batch_size):
            batch = non_empty[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"    Batch {batch_num}/{total_batches}...", end=' ', flush=True)
            
            contents_str = "\n\n---\n\n".join([f"Content {j+1}: {content}" for j, (_, content) in enumerate(batch)])
            
            prompt = f"""You are a game review analysis assistant. You have a set of card system evaluation content. Please determine whether each evaluation is positive or negative SPECIFICALLY ABOUT THE CARD SYSTEM, not about other aspects of the game.

Important: Only judge the sentiment toward the card system itself. Ignore mentions of other game aspects (graphics, servers, matchmaking, etc.). Focus solely on whether the player's opinion about the card system is positive or negative.

Evaluation content:
{contents_str}

Return in JSON format as follows:
{{
  "results": [
    {{"index": 1, "sentiment": "positive" or "negative"}},
    {{"index": 2, "sentiment": "positive" or "negative"}},
    ...
  ]
}}

Return only JSON, no other explanations."""
            
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                text = response.content[0].text.strip()
                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                        if text.lstrip().startswith("json"):
                            text = text.lstrip()[4:]
                    text = text.strip()
                
                result = json.loads(text)
                batch_classifications = result.get("results", [])
                batch_classifications.sort(key=lambda x: x.get("index", 0))
                
                # Map back to original index
                for j, (orig_idx, _) in enumerate(batch):
                    if j < len(batch_classifications):
                        all_classifications.append({
                            "original_index": orig_idx,
                            "sentiment": batch_classifications[j].get("sentiment", "positive")
                        })
                
                print("✓")
                time.sleep(0.3)
                
            except Exception as e:
                print(f"✗ Error: {e}")
        
        return all_classifications
        
    except Exception as e:
        print(f"⚠️  Error in Haiku classification: {e}")
        return []


def analyze_themes_with_sonnet(card_positive_texts, card_negative_texts, category, api_key):
    """Use Sonnet for theme analysis - separately analyze positive and negative feedback"""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        model = "claude-sonnet-4-5-20250929"
        
        # Get category name (keeping English for now, can be localized later)
        category_map_cn = {
            'Early (0-2h)': 'Early (0-2h)',
            'Mid (2-10h)': 'Mid (2-10h)',
            'Late (10-50h)': 'Late (10-50h)',
            'Veteran (50h+)': 'Veteran (50h+)'
        }
        category_cn = category_map_cn.get(category, category)
        
        result = {
            'positive_points': [],
            'negative_points': []
        }
        
        # Analyze positive feedback separately
        if len(card_positive_texts) > 0:
            print(f"  Using Sonnet to analyze {len(card_positive_texts)} positive reviews...", end=' ', flush=True)
            
            # Limit to 300 for analysis if too many
            if len(card_positive_texts) > 300:
                texts_for_analysis = card_positive_texts[:300]
                note = f"(showing first 300 for analysis, total {len(card_positive_texts)})"
            else:
                texts_for_analysis = card_positive_texts
                note = ""
            
            texts_str = "\n\n---\n\n".join([f"Review {i+1}: {text}" for i, text in enumerate(texts_for_analysis)])
            
            prompt_positive = f"""You are a game review analysis expert. You have a set of **positive reviews** about the card system from players in the playtime segment {category_cn}. {note}

These reviews are all positive feedback about the card system. Please analyze these reviews and extract the main positive points players have about the card system.

Positive review content:
{texts_str}

Return in JSON format as follows:
{{
  "main_points": [
    {{
      "point": "Main positive point 1 about the card system (in English)",
      "description": "Detailed description of this point (optional)"
    }},
    {{
      "point": "Main positive point 2 about the card system (in English)",
      "description": "Detailed description of this point (optional)"
    }}
  ]
}}

**Important requirements:**
1. All points must be expressed in English, concise and clear, one sentence summary
2. description is optional, used to better describe the point
3. Generate 3-8 main positive points covering all aspects of the card system
4. Return only JSON, no other explanations
5. All special characters in strings (quotes, newlines, etc.) must be properly escaped"""
            
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt_positive}]
                )
                
                text = response.content[0].text.strip()
                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                        if text.lstrip().startswith("json"):
                            text = text.lstrip()[4:]
                    text = text.strip()
                
                positive_result = json.loads(text)
                result['positive_points'] = positive_result.get('main_points', [])
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        # Analyze negative feedback separately
        if len(card_negative_texts) > 0:
            print(f"  Using Sonnet to analyze {len(card_negative_texts)} negative reviews...", end=' ', flush=True)
            
            # Limit to 300 for analysis if too many
            if len(card_negative_texts) > 300:
                texts_for_analysis = card_negative_texts[:300]
                note = f"(showing first 300 for analysis, total {len(card_negative_texts)})"
            else:
                texts_for_analysis = card_negative_texts
                note = ""
            
            texts_str = "\n\n---\n\n".join([f"Review {i+1}: {text}" for i, text in enumerate(texts_for_analysis)])
            
            prompt_negative = f"""You are a game review analysis expert. You have a set of **negative reviews** about the card system from players in the playtime segment {category_cn}. {note}

These reviews are all negative feedback about the card system. Please analyze these reviews and extract the main negative points players have about the card system.

Negative review content:
{texts_str}

Return in JSON format as follows:
{{
  "main_points": [
    {{
      "point": "Main negative point 1 about the card system (in English)",
      "description": "Detailed description of this point (optional)"
    }},
    {{
      "point": "Main negative point 2 about the card system (in English)",
      "description": "Detailed description of this point (optional)"
    }}
  ]
}}

**Important requirements:**
1. All points must be expressed in English, concise and clear, one sentence summary
2. description is optional, used to better describe the point
3. Generate 3-8 main negative points covering all aspects of the card system
4. Return only JSON, no other explanations
5. All special characters in strings (quotes, newlines, etc.) must be properly escaped"""
            
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt_negative}]
                )
                
                text = response.content[0].text.strip()
                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                        if text.lstrip().startswith("json"):
                            text = text.lstrip()[4:]
                    text = text.strip()
                
                negative_result = json.loads(text)
                result['negative_points'] = negative_result.get('main_points', [])
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        return result
        
    except Exception as e:
        print(f"⚠️  Error in Sonnet analysis: {e}")
        return None


def map_comments_to_points_with_haiku(main_points, all_card_texts, api_key, batch_size=20):
    """Use Haiku to map comments to main points and calculate accurate volume and backup comments"""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        model = "claude-haiku-4-5-20251001"
        
        if not main_points or len(main_points) == 0:
            return None
        
        # Prepare points list
        points_list = []
        for point_data in main_points:
            if isinstance(point_data, dict):
                point_text = point_data.get('point', '')
                description = point_data.get('description', '')
                points_list.append({
                    'name': point_text,
                    'description': description
                })
            else:
                # Compatible with old format (pure string)
                points_list.append({
                    'name': str(point_data),
                    'description': ''
                })
        
        print(f"    Using Haiku to map {len(all_card_texts)} comments to {len(points_list)} points...", end=' ', flush=True)
        
        # Build points list text
        theme_lines = [f"{i+1}. Point: {p['name']}\n   Description: {p['description'] or '(no description)'}" 
                      for i, p in enumerate(points_list)]
        
        # Process comments in batches
        all_mappings = []
        total_batches = (len(all_card_texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(all_card_texts), batch_size):
            batch = all_card_texts[batch_idx:batch_idx+batch_size]
            batch_num = batch_idx // batch_size + 1
            
            comment_lines = [f"{i+1}. {comment}" for i, comment in enumerate(batch)]
            
            prompt = f"""You are a game user feedback analysis assistant. You have a set of comments about the card system and a set of organized main points.

Your task: Tag each comment below with one or more point labels.

[Main Points List] (only use these point names, do not invent new points):
{chr(10).join(theme_lines)}

[Comments to Tag]:
{chr(10).join(comment_lines)}

Tagging rules:
1. Each comment can correspond to 0-3 points, if it doesn't match at all, return empty list []
2. Try to cover the core issues in the comment, not just surface words
3. Only use the point names given above as labels
4. You can tag one comment with multiple points

Output format (JSON only, no explanations, no markdown code blocks):
{{
  "results": [
    {{"index": 1, "points": ["Point Name 1", "Point Name 2"]}},
    {{"index": 2, "points": []}},
    ...
  ]
}}"""
            
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                text = response.content[0].text.strip()
                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                        if text.lstrip().startswith("json"):
                            text = text.lstrip()[4:]
                    text = text.strip()
                
                result = json.loads(text)
                batch_results = result.get("results", [])
                batch_results_sorted = sorted(batch_results, key=lambda x: x.get("index", 0))
                
                # Validate point names and map to comments
                point_names = {p['name'] for p in points_list}
                for r in batch_results_sorted:
                    idx = r.get("index", 1) - 1  # Convert to 0-based index
                    if 0 <= idx < len(batch):
                        points = r.get("points", [])
                        cleaned_points = [p for p in points if p in point_names]
                        all_mappings.append({
                            'comment': batch[idx],
                            'points': cleaned_points
                        })
                    else:
                        # Index out of range, add empty mapping
                        all_mappings.append({
                            'comment': '',
                            'points': []
                        })
                
                time.sleep(0.3)  # Avoid rate limiting
                
            except Exception as e:
                print(f"✗ Batch {batch_num} error: {e}")
                # Add empty mappings
                all_mappings.extend([{'comment': comment, 'points': []} for comment in batch])
        
        # Calculate volume and backup comments for each point
        point_stats = {}
        for point_data in points_list:
            point_name = point_data['name']
            matching_comments = []
            
            for mapping in all_mappings:
                if point_name in mapping['points']:
                    matching_comments.append(mapping['comment'])
            
            point_stats[point_name] = {
                'comment_count': len(matching_comments),
                'backup_comments': matching_comments[:5]  # Take first 5 as backup
            }
        
        # Update main_points, add accurate volume and backup comments
        updated_points = []
        for point_data in main_points:
            if isinstance(point_data, dict):
                point_text = point_data.get('point', '')
            else:
                point_text = str(point_data)
            
            stats = point_stats.get(point_text, {'comment_count': 0, 'backup_comments': []})
            
            updated_points.append({
                'point': point_text,
                'description': point_data.get('description', '') if isinstance(point_data, dict) else '',
                'comment_count': stats['comment_count'],
                'backup_comments': stats['backup_comments']
            })
        
        print("✓")
        return {'main_points': updated_points}
        
    except Exception as e:
        print(f"⚠️  Error in Haiku mapping: {e}")
        return None


def analyze_with_ai_old(reviews_text, category, sentiment, api_key=None, use_haiku=False):
    """Use AI to analyze review text, only extract card-related reviews"""
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("BROWSER_API_KEY")
    
    if not api_key:
        print(f"⚠️  No API key found. Skipping AI analysis for {category} - {sentiment}")
        return None
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        
        # Choose model: haiku is cheaper, sonnet is more accurate
        if use_haiku:
            model = "claude-haiku-4-5-20251001"
            max_tokens = 4000
        else:
            model = "claude-sonnet-4-5-20250929"
            max_tokens = 2000
        
        # Adjust sample size based on model
        # Haiku: 200k context, can handle more reviews
        # Sonnet: 200k context, but more expensive, sample fewer
        if use_haiku:
            sample_size = min(100, len(reviews_text))  # Haiku can handle more
        else:
            sample_size = min(50, len(reviews_text))   # Sonnet samples 50
        
        sample_reviews = reviews_text[:sample_size]
        
        reviews_str = "\n\n---\n\n".join([f"Review {i+1}: {review}" for i, review in enumerate(sample_reviews)])
        
        prompt = f"""You are a game review analysis expert. You have a set of {sentiment} reviews from players with playtime {category}.

**Important: These reviews may contain other aspects of the game (graphics, sound, matchmaking, etc.), but you only need to focus on and extract the parts related to the "card system".**

Please complete the following tasks:
1. **Extract card-related reviews**: From each review, only extract content related to cards, decks, card drawing, card abilities, card balance, etc.
2. **Ignore other content**: Ignore content about other game aspects (graphics, sound, matchmaking, UI, etc.)
3. **Analyze card system opinions**: Based on extracted card-related reviews, summarize:
   - Main points players have about the card system (3-5 key points)
   - Main issues or advantages mentioned (card system only)
   - Specific evaluation words players use for the card system
   - Comparisons with card systems from other games (if any)
   - Player suggestions for improving the card system (if any)

Review content:
{reviews_str}

Return analysis results in JSON format as follows:
{{
  "card_related_extracts": ["Card-related content extracted from review 1", "Card-related content extracted from review 2", ...],
  "main_points": ["Main point 1 about card system", "Main point 2", ...],
  "key_issues": ["Card system issue 1", "Issue 2", ...] or ["Card system advantage 1", "Advantage 2", ...],
  "evaluation_words": ["Evaluation word 1", "Evaluation word 2", ...],
  "comparisons": ["Comparison with other game card systems 1", "Comparison 2", ...],
  "suggestions": ["Improvement suggestion 1", "Suggestion 2", ...],
  "summary": "Overall summary about the card system"
}}

**Note**: Only analyze and return content related to the card system, ignore other game content in reviews.

Return only JSON, no other explanations."""
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text.strip()
        
        # Clean JSON
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:]
            text = text.strip()
        
        result = json.loads(text)
        return result
        
    except Exception as e:
        print(f"⚠️  Error in AI analysis: {e}")
        return None


def analyze_by_playtime_with_ai(df, use_ai=True, api_key=None):
    """Analyze by playtime category using Haiku classification + Sonnet theme analysis
    
    Args:
        df: DataFrame
        use_ai: Whether to use AI analysis
        api_key: API key (if None, will try to get from environment or files)
    """
    # Category name mapping (keeping English)
    category_map = {
        'Early (0-2h)': 'Early (0-2h)',
        'Mid (2-10h)': 'Mid (2-10h)',
        'Late (10-50h)': 'Late (10-50h)',
        'Veteran (50h+)': 'Veteran (50h+)'
    }
    
    playtime_categories = ['Early (0-2h)', 'Mid (2-10h)', 'Late (10-50h)', 'Veteran (50h+)']
    
    results = {}
    
    # Get API key if not provided
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("BROWSER_API_KEY")
        
        # Try to get from script file if not in environment
        if not api_key:
            script_file = Path(__file__).parent.parent.parent / 'youtube' / 'run_insight_generator.sh'
            if script_file.exists():
                api_key = get_api_key_from_file(str(script_file), r'export ANTHROPIC_API_KEY="([^"]+)"')
        
        # Try .env file
        if not api_key:
            env_file = Path(__file__).parent / '.env'
            if env_file.exists():
                api_key = get_api_key_from_file(str(env_file), r'ANTHROPIC_API_KEY=(.+)')
    
    if use_ai and not api_key:
        print("⚠️  No API key found. Running without AI analysis.")
        use_ai = False
    
    if use_ai:
        print("\nUsing three-step analysis process (process all reviews, no sampling):")
        print("  1. Haiku: Extract card-related paragraphs from all reviews (reduce cost)")
        print("  2. Haiku: Classify extracted content as positive/negative")
        print("  3. Sonnet: Theme analysis on classified reviews (analyze all content)")
    
    for category in playtime_categories:
        # Get category name
        category_cn = category_map.get(category, category)
        print(f'\n{"="*60}')
        print(f'Analyzing {category_cn}')
        print("="*60)
        
        subset = df[df['playtime_category'] == category].copy()
        
        if len(subset) == 0:
            continue
        
        category_results = {
            'total_reviews': len(subset),
        }
        
        if not use_ai:
            results[category] = category_results
            continue
        
        # Step 1: Use Haiku to extract card-related paragraphs from all reviews
        print(f'  Step 1: Using Haiku to extract card content from {len(subset)} reviews (process all reviews)...')
        review_texts = subset['review_text'].dropna().tolist()
        
        extracts = extract_card_content_with_haiku(review_texts, api_key)
        
        if not extracts:
            print("  ⚠️  Extraction failed, skipping this category")
            results[category] = category_results
            continue
        
        # Count extraction results
        card_contents = [e.get("card_content", "").strip() for e in extracts]
        no_card_related = sum(1 for c in card_contents if not c)
        has_card_content = [c for c in card_contents if c]
        
        print(f'    Extraction results: {len(has_card_content)} contain card content, {no_card_related} do not')
        
        if len(has_card_content) == 0:
            print("  ⚠️  No card-related content extracted, skipping this category")
            category_results['card_positive_count'] = 0
            category_results['card_negative_count'] = 0
            category_results['no_card_related_count'] = no_card_related
            results[category] = category_results
            continue
        
        # Step 2: Use Haiku to classify extracted content as positive or negative
        print(f'  Step 2: Using Haiku to classify {len(has_card_content)} card contents as positive/negative...')
        classifications = classify_extracted_content_with_haiku(has_card_content, api_key)
        
        # Create index mapping
        content_to_sentiment = {}
        for cls in classifications:
            orig_idx = cls.get("original_index", -1)
            if orig_idx >= 0 and orig_idx < len(has_card_content):
                content_to_sentiment[orig_idx] = cls.get("sentiment", "positive")
        
        # Count classification results
        card_positive_texts = []
        card_negative_texts = []
        
        for idx, content in enumerate(has_card_content):
            sentiment = content_to_sentiment.get(idx, "positive")  # default positive
            if sentiment == "positive":
                card_positive_texts.append(content)
            else:
                card_negative_texts.append(content)
        
        category_results['card_positive_count'] = len(card_positive_texts)
        category_results['card_negative_count'] = len(card_negative_texts)
        category_results['no_card_related_count'] = no_card_related
        category_results['card_positive_pct'] = len(card_positive_texts) / len(subset) * 100 if len(subset) > 0 else 0
        category_results['card_negative_pct'] = len(card_negative_texts) / len(subset) * 100 if len(subset) > 0 else 0
        
        # ✅ Save all intermediate results: classified text content
        category_results['card_positive_texts'] = card_positive_texts
        category_results['card_negative_texts'] = card_negative_texts
        
        print(f'    Classification results: {len(card_positive_texts)} positive, {len(card_negative_texts)} negative, {no_card_related} no card content')
        
        # Step 3: Use Sonnet for theme analysis (separately for positive and negative)
        if len(card_positive_texts) > 0 or len(card_negative_texts) > 0:
            print(f'  Step 3: Using Sonnet to generate positive and negative main points separately...')
            theme_analysis = analyze_themes_with_sonnet(
                card_positive_texts, 
                card_negative_texts, 
                category, 
                api_key
            )
            
            if theme_analysis:
                # ✅ Save raw theme analysis results (separated by positive/negative)
                category_results['theme_analysis_raw'] = theme_analysis
                
                # Step 4: Use Haiku to map comments to points and calculate accurate volume/backup
                # Process positive and negative points separately
                positive_points = theme_analysis.get('positive_points', [])
                negative_points = theme_analysis.get('negative_points', [])
                
                mapped_analysis = {
                    'positive_points': [],
                    'negative_points': []
                }
                
                # Map positive comments to positive points
                if len(positive_points) > 0 and len(card_positive_texts) > 0:
                    print(f'  Step 4a: Using Haiku to map positive comments to positive points...')
                    positive_mapped = map_comments_to_points_with_haiku(
                        positive_points,
                        card_positive_texts,
                        api_key
                    )
                    if positive_mapped and 'main_points' in positive_mapped:
                        mapped_analysis['positive_points'] = positive_mapped['main_points']
                
                # Map negative comments to negative points
                if len(negative_points) > 0 and len(card_negative_texts) > 0:
                    print(f'  Step 4b: Using Haiku to map negative comments to negative points...')
                    negative_mapped = map_comments_to_points_with_haiku(
                        negative_points,
                        card_negative_texts,
                        api_key
                    )
                    if negative_mapped and 'main_points' in negative_mapped:
                        mapped_analysis['negative_points'] = negative_mapped['main_points']
                
                # ✅ Save mapped analysis results
                category_results['theme_analysis'] = mapped_analysis
        
        results[category] = category_results
    
    return results


def save_ai_analysis(results, output_file):
    """Save AI analysis results (full version - preserve all intermediate results)"""
    # Convert to serializable format, preserve all intermediate results
    output_data = {}
    for category, data in results.items():
        output_data[category] = {
            'total_reviews': int(data['total_reviews']),
            'card_positive_count': int(data.get('card_positive_count', 0)),
            'card_negative_count': int(data.get('card_negative_count', 0)),
            'no_card_related_count': int(data.get('no_card_related_count', 0)),
            'card_positive_pct': float(data.get('card_positive_pct', 0)),
            'card_negative_pct': float(data.get('card_negative_pct', 0)),
        }
        
        # ✅ Save all intermediate results: classified text content
        if 'card_positive_texts' in data:
            output_data[category]['card_positive_texts'] = data['card_positive_texts']
        if 'card_negative_texts' in data:
            output_data[category]['card_negative_texts'] = data['card_negative_texts']
        
        # ✅ Save raw theme analysis results (separated by positive/negative)
        if 'theme_analysis_raw' in data and data['theme_analysis_raw']:
            output_data[category]['theme_analysis_raw'] = data['theme_analysis_raw']
        
        # ✅ Save mapped theme analysis results (includes backup comments and volume, separated by positive/negative)
        if 'theme_analysis' in data and data['theme_analysis']:
            theme = data['theme_analysis']
            output_data[category]['positive_points'] = theme.get('positive_points', [])
            output_data[category]['negative_points'] = theme.get('negative_points', [])
            
            # For compatibility, also save merged main_points
            all_points = theme.get('positive_points', []) + theme.get('negative_points', [])
            output_data[category]['main_points'] = all_points
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f'\n✅ Saved AI analysis results (including all intermediate results): {output_file}')
    return output_data


def print_ai_summary(results):
    """Print AI analysis summary (simplified version)"""
    # Category name mapping
    category_map = {
        'Early (0-2h)': 'Early (0-2h)',
        'Mid (2-10h)': 'Mid (2-10h)',
        'Late (10-50h)': 'Late (10-50h)',
        'Veteran (50h+)': 'Veteran (50h+)'
    }
    
    print('\n' + '='*60)
    print('Card System Evaluation Summary by Playtime Segment')
    print('='*60)
    
    playtime_categories = ['Early (0-2h)', 'Mid (2-10h)', 'Late (10-50h)', 'Veteran (50h+)']
    
    for category in playtime_categories:
        if category not in results:
            continue
        
        data = results[category]
        category_cn = category_map.get(category, category)
        print(f'\n{category_cn}:')
        print(f'  Total reviews: {data["total_reviews"]:,}')
        print(f'  Card positive: {data.get("card_positive_count", 0):,} ({data.get("card_positive_pct", 0):.1f}%)')
        print(f'  Card negative: {data.get("card_negative_count", 0):,} ({data.get("card_negative_pct", 0):.1f}%)')
        print(f'  No card content: {data.get("no_card_related_count", 0):,}')
        
        if 'theme_analysis' in data and data['theme_analysis']:
            theme = data['theme_analysis']
            if 'main_points' in theme:
                print(f'\n  📊 Main points about card system:')
                for i, point_data in enumerate(theme['main_points'][:5], 1):
                    # Support new format (object) and old format (string)
                    if isinstance(point_data, dict):
                        point_text = point_data.get('point', '')
                        comment_count = point_data.get('comment_count', 0)
                        backup_comments = point_data.get('backup_comments', [])
                        print(f'    {i}. {point_text} (mentioned in {comment_count} comments)')
                        if backup_comments:
                            print(f'       Example comments:')
                            for j, backup in enumerate(backup_comments[:3], 1):  # Show only first 3
                                backup_short = backup[:100] + '...' if len(backup) > 100 else backup
                                print(f'         • {backup_short}')
                    else:
                        # Compatible with old format (pure string)
                        print(f'    {i}. {point_data}')


def generate_dashboard(analysis_data, output_file):
    """Generate Dashboard"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        playtime_categories = ['Early (0-2h)', 'Mid (2-10h)', 'Late (10-50h)', 'Veteran (50h+)']
        
        # Prepare data
        categories = []
        positive_counts = []
        negative_counts = []
        main_points_list = []
        
        for category in playtime_categories:
            if category not in analysis_data:
                continue
            data = analysis_data[category]
            categories.append(category)
            positive_counts.append(data.get('card_positive_count', 0))
            negative_counts.append(data.get('card_negative_count', 0))
            main_points_list.append(data.get('main_points', []))
        
        # Category name mapping
        category_map = {
            'Early (0-2h)': 'Early (0-2h)',
            'Mid (2-10h)': 'Mid (2-10h)',
            'Late (10-50h)': 'Late (10-50h)',
            'Veteran (50h+)': 'Veteran (50h+)'
        }
        
        # Convert to category names
        categories_cn = [category_map.get(cat, cat) for cat in categories]
        
        # Create chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Card System Sentiment Distribution', 'Main Points by Playtime Segment'),
            vertical_spacing=0.15,
            row_heights=[0.4, 0.6],
            specs=[[{"type": "bar"}], [{"type": "table"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=categories_cn,
                y=positive_counts,
                name='Positive',
                marker_color='#90EE90',
                hovertemplate='<b>%{x}</b><br>Positive: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=categories_cn,
                y=negative_counts,
                name='Negative',
                marker_color='#e74c3c',
                hovertemplate='<b>%{x}</b><br>Negative: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Update layout (remove table from subplot, will add as HTML table)
        fig.update_xaxes(title_text="Playtime Segment", row=1, col=1)
        fig.update_yaxes(title_text="Number of Reviews", row=1, col=1)
        
        fig.update_layout(
            title={
                'text': 'Card System Evaluation by Playtime Segment',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=600,
            showlegend=True,
            barmode='group',
            margin=dict(t=80, b=50, l=50, r=50)
        )
        
        # Prepare detailed HTML table with backup comments and volume
        points_html = ""
        for i, category in enumerate(categories):
            category_cn = category_map.get(category, category)
            points = main_points_list[i]
            points_html += f"""
            <div class="category-section">
                <h3>{category_cn}</h3>
            """
            if points and len(points) > 0:
                for j, point_data in enumerate(points, 1):
                    if isinstance(point_data, dict):
                        point_text = point_data.get('point', '')
                        comment_count = point_data.get('comment_count', 0)
                        backup_comments = point_data.get('backup_comments', [])
                        points_html += f"""
                <div class="point-item">
                    <div class="point-header">
                        <span class="point-number">{j}.</span>
                        <span class="point-text">{point_text}</span>
                        <span class="point-volume">Mentioned in: {comment_count} comments</span>
                    </div>
                    {f'''
                    <div class="backup-comments">
                        <strong>Example Comments:</strong>
                        <ul>
                            {''.join([f'<li>{html.escape(backup)}</li>' for backup in backup_comments[:5]])}
                        </ul>
                    </div>
                    ''' if backup_comments else ''}
                </div>
                """
                    else:
                        # 兼容旧格式
                        points_html += f"""
                <div class="point-item">
                    <div class="point-header">
                        <span class="point-number">{j}.</span>
                        <span class="point-text">{point_data}</span>
                    </div>
                </div>
                """
            else:
                points_html += '<p style="color: #999; font-style: italic;">⚠️ Main points analysis for this segment is incomplete or data is missing. Please rerun the full analysis (including AI steps).</p>'
            points_html += "</div>"
        
        # 生成HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Card System Evaluation Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: 700;
            color: #667eea;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }}
        
        .points-section {{
            margin-top: 40px;
        }}
        
        .category-section {{
            margin-bottom: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
        }}
        
        .category-section h3 {{
            color: #667eea;
            font-size: 1.5em;
            margin-bottom: 20px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .point-item {{
            margin-bottom: 25px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .point-header {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            margin-bottom: 10px;
        }}
        
        .point-number {{
            font-weight: bold;
            color: #667eea;
            min-width: 25px;
        }}
        
        .point-text {{
            flex: 1;
            font-size: 1.1em;
            line-height: 1.5;
        }}
        
        .point-volume {{
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.9em;
            white-space: nowrap;
        }}
        
        .backup-comments {{
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        
        .backup-comments strong {{
            color: #495057;
            display: block;
            margin-bottom: 10px;
        }}
        
        .backup-comments ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        
        .backup-comments li {{
            padding: 8px 0;
            border-bottom: 1px solid #dee2e6;
            color: #495057;
            line-height: 1.6;
        }}
        
        .backup-comments li:last-child {{
            border-bottom: none;
        }}
        
        .backup-comments li::before {{
            content: "💬 ";
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎴 Card System Evaluation Dashboard</h1>
            <p>Card System Feedback Analysis by Playtime Segment</p>
        </div>
        
        <div class="content">
            <div class="stats-grid">
"""
        
        # Add statistics cards
        total_positive = sum(data.get('card_positive_count', 0) for data in analysis_data.values())
        total_negative = sum(data.get('card_negative_count', 0) for data in analysis_data.values())
        total_reviews = sum(data.get('total_reviews', 0) for data in analysis_data.values())
        
        # Calculate positive rate safely
        if total_positive + total_negative > 0:
            positive_rate = total_positive / (total_positive + total_negative) * 100
        else:
            positive_rate = 0.0
        
        html_content += f"""
                <div class="stat-card">
                    <div class="stat-value">{total_reviews:,}</div>
                    <div class="stat-label">Total Reviews</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_positive:,}</div>
                    <div class="stat-label">Card Positive</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_negative:,}</div>
                    <div class="stat-label">Card Negative</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{positive_rate:.1f}%</div>
                    <div class="stat-label">Positive Rate</div>
                </div>
            </div>
            
            <div id="chart-container">
                {fig.to_html(include_plotlyjs='cdn', div_id='card_evaluation_chart')}
            </div>
            
            <div class="points-section">
                <h2 style="color: #667eea; margin-bottom: 30px; font-size: 1.8em;">📊 Main Points Details</h2>
                {points_html}
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f'\n✅ Dashboard generated: {output_file}')
        
    except Exception as e:
        print(f'⚠️  Error generating dashboard: {e}')


def main():
    """Main function"""
    card_file = Path(__file__).parent.parent / 'data' / 'processed' / 'card_system_reviews_20251204_203034.csv'
    
    if not card_file.exists():
        print(f"❌ Error: File not found: {card_file}")
        return
    
    df = load_data(str(card_file))
    
    # Check for API key - try multiple sources
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("BROWSER_API_KEY")
    
    # Try to get from script file if not in environment
    if not api_key:
        script_file = Path(__file__).parent.parent.parent / 'youtube' / 'run_insight_generator.sh'
        if script_file.exists():
            api_key = get_api_key_from_file(str(script_file), r'export ANTHROPIC_API_KEY="([^"]+)"')
            if api_key:
                print(f"✓ Found API key from {script_file.name}")
    
    # Try .env file
    if not api_key:
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            api_key = get_api_key_from_file(str(env_file), r'ANTHROPIC_API_KEY=(.+)')
            if api_key:
                print(f"✓ Found API key from .env file")
    
    use_ai = api_key is not None
    
    if not use_ai:
        print("\n⚠️  No API key found. Set ANTHROPIC_API_KEY to use AI analysis.")
        print("Running basic analysis only...")
    
    # Analyze text
    print('\nAnalyzing card system evaluation by playtime segment...')
    results = analyze_by_playtime_with_ai(df, use_ai=use_ai, api_key=api_key)
    
    # Save results
    output_dir = Path('output/insights')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'card_ai_analysis_by_playtime_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    analysis_data = save_ai_analysis(results, str(output_file))
    
    # Generate Dashboard
    dashboard_dir = Path('output/dashboards')
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    dashboard_file = dashboard_dir / f'card_system_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
    generate_dashboard(analysis_data, str(dashboard_file))
    
    # Print summary
    print_ai_summary(results)
    
    print('\n✅ Analysis complete!')
    print(f'\nResults saved to: {output_file}')
    print(f'Dashboard saved to: {dashboard_file}')


if __name__ == '__main__':
    main()

