"""
Twitch comment sentiment analysis and marketing funnel classification tool
Uses a two-step process:
1. Lightweight filtering: Filter out irrelevant comments and emotes
2. OpenAI GPT-4o classification: 5-category classification based on marketing funnel
"""

import os
import re
import csv
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import openai
from openai import OpenAI
import time

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLAY_DIR = "/Users/zizhengwan/Desktop/Twitch/replay/out"
OUTPUT_DIR = "/Users/zizhengwan/Desktop/Twitch/results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Game-related keywords (for initial filtering)
GAME_KEYWORDS = [
    'fragpunk', 'frag punk', 'game', 'spiel', 'play', 'spielen',
    'download', 'herunterladen', 'install', 'installieren',
    'mod', 'mode', 'modus', 'beta', 'update', 'skill', 'skill',
    'rank', 'ranked', 'rang', 'bot', 'bots', 'leaderboard',
    'tower', 'tower defense', 'turm', 'card', 'karte', 'deck',
    'match', 'matchmaking', 'queue', 'pve', 'pvp', 'multiplayer'
]

# Common Twitch emoji and emote patterns
EMOTE_PATTERNS = [
    r'^[A-Za-z0-9]+[A-Z][A-Z0-9]+$',  # e.g., raxHYPE2, raxL2
    r'^[A-Z]{2,}$',  # e.g., LUL, GG, HYPE
    r'^[a-z]+[A-Z][a-z]+$',  # e.g., nameEmote
]

# Funnel category definitions
FUNNEL_CATEGORIES = {
    "Intent to Play": "The viewer expresses initial curiosity about the game without taking any action or seeking details.",
    "Pre-Install": "The viewer actively forms an opinion before downloading, including comparisons, questions, or go/no-go decisions about installing the game.",
    "Post-Play": "The viewer comments on hands-on experience—gameplay mechanics, modes, updates, or early in-game behavior—after having downloaded or tried the game.",
    "Loyalty": "The viewer demonstrates sustained engagement with the game itself, such as long-term play, repeat spending, community contributions, or referrals tied specifically to the game.",
    "Irrelevant": "The chat line is off-topic, streamer-focused, or general banter not related to the game."
}


def is_likely_emote(text: str) -> bool:
    """Detect if text is likely a Twitch emote"""
    text = text.strip()
    
    # Too short might be emote
    if len(text) < 2:
        return True
    
    # Check if matches emote patterns
    for pattern in EMOTE_PATTERNS:
        if re.match(pattern, text):
            return True
    
    # If only contains words and no spaces, might be emote
    if ' ' not in text and text.isalnum() and len(text) < 15:
        # Check if contains numbers (common in emotes)
        if re.search(r'\d', text):
            return True
    
    return False


def is_only_url(text: str) -> bool:
    """Detect if text only contains URL"""
    url_pattern = r'https?://\S+|www\.\S+'
    # After removing all URLs, check if there's still content
    without_urls = re.sub(url_pattern, '', text).strip()
    return len(without_urls) == 0


def contains_game_keywords(text: str) -> bool:
    """Detect if text contains game-related keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in GAME_KEYWORDS)


def filter_relevant_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    First step filtering: Filter out irrelevant comments and emotes
    Keep potentially relevant comments for subsequent classification
    """
    filtered_rows = []
    
    for _, row in df.iterrows():
        message = str(row['message']).strip()
        
        # Skip empty messages
        if not message or len(message) == 0:
            continue
        
        # Skip messages that are too short (might be emote or spam)
        if len(message) < 3:
            continue
        
        # Skip messages that only contain emotes
        if is_likely_emote(message):
            continue
        
        # Skip messages that only contain URLs (usually ads)
        if is_only_url(message):
            continue
        
        # Keep comments containing game keywords, or longer comments (might be relevant discussion)
        if contains_game_keywords(message) or len(message) > 20:
            filtered_rows.append(row)
    
    return pd.DataFrame(filtered_rows)


def classify_with_openai(client: OpenAI, messages: List[str], game_name: str = "fragpunk") -> List[str]:
    """
    Use OpenAI GPT-4o API for batch classification
    Returns list of classification results
    """
    # Build batch prompt
    messages_str = "\n".join([f"{i+1}. {msg}" for i, msg in enumerate(messages)])
    
    prompt = f"""You are analyzing Twitch chat comments about the game "{game_name}". 
Classify each comment into ONE of the following 5 categories based on the marketing funnel:

1. Intent to Play: The viewer expresses initial curiosity about the game without taking any action or seeking details.

2. Pre-Install: The viewer actively forms an opinion before downloading, including comparisons, questions, or go/no-go decisions about installing the game.

3. Post-Play: The viewer comments on hands-on experience—gameplay mechanics, modes, updates, or early in-game behavior—after having downloaded or tried the game.

4. Loyalty: The viewer demonstrates sustained engagement with the game itself, such as long-term play, repeat spending, community contributions, or referrals tied specifically to the game.

5. Irrelevant: The chat line is off-topic, streamer-focused, or general banter not related to the game.

Comments to classify:
{messages_str}

Respond with a JSON object with a "categories" key containing an array of category names, one for each comment in order. Example: {{"categories": ["Intent to Play", "Pre-Install", "Irrelevant"]}}
Only use these exact category names: Intent to Play, Pre-Install, Post-Play, Loyalty, Irrelevant."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a marketing analyst specializing in gaming chat sentiment analysis. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            result_json = json.loads(result_text)
            # Find key containing classification results
            if "categories" in result_json:
                categories = result_json["categories"]
            elif isinstance(result_json, list):
                categories = result_json
            else:
                # Try to find first array value
                categories = list(result_json.values())[0] if result_json else []
        except:
            # If not JSON, try parsing line by line
            lines = result_text.split('\n')
            categories = [line.strip().strip('"').strip("'") for line in lines if line.strip()]
        
        # Validate and correct classification results
        valid_categories = list(FUNNEL_CATEGORIES.keys())
        validated_categories = []
        
        for cat in categories:
            cat = str(cat).strip()
            if cat in valid_categories:
                validated_categories.append(cat)
            else:
                # Try fuzzy matching
                cat_lower = cat.lower()
                matched = False
                for valid_cat in valid_categories:
                    if valid_cat.lower() in cat_lower or cat_lower in valid_cat.lower():
                        validated_categories.append(valid_cat)
                        matched = True
                        break
                if not matched:
                    validated_categories.append("Irrelevant")
        
        # Ensure returned count matches
        while len(validated_categories) < len(messages):
            validated_categories.append("Irrelevant")
        
        return validated_categories[:len(messages)]
        
    except Exception as e:
        print(f"Error classifying messages: {e}")
        # 返回默认分类
        return ["Irrelevant"] * len(messages)


def process_csv_file(csv_path: str, client: OpenAI) -> Dict[str, any]:
    """
    Process a single CSV file, return classification result statistics
    """
    # Extract streamer name from filename
    filename = os.path.basename(csv_path)
    # Filename format: twitch_chat_{ID}_{streamer_name}.csv
    match = re.search(r'twitch_chat_\d+_(.+)\.csv$', filename)
    if match:
        streamer_name = match.group(1)
    else:
        streamer_name = "Unknown"
    
    print(f"\nProcessing streamer: {streamer_name}")
    print(f"File: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    total_comments = len(df)
    print(f"Total comments: {total_comments}")
    
    # Step 1: Filter relevant comments
    filtered_df = filter_relevant_comments(df)
    filtered_count = len(filtered_df)
    print(f"Filtered comments: {filtered_count} ({filtered_count/total_comments*100:.1f}%)")
    
    # Step 2: Use OpenAI for batch classification
    categories = defaultdict(int)
    classifications = []
    
    # Batch processing, 10 comments per batch
    batch_size = 10
    batches = [filtered_df.iloc[i:i+batch_size] for i in range(0, len(filtered_df), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        messages = [str(row['message']) for _, row in batch.iterrows()]
        
        # Batch classification
        batch_categories = classify_with_openai(client, messages)
        
        # Save results
        for (_, row), category in zip(batch.iterrows(), batch_categories):
            categories[category] += 1
            classifications.append({
                'message': str(row['message']),
                'category': category,
                'username': row.get('username_display', ''),
                'timestamp': row.get('utc_time', '')
            })
        
        # Show progress
        processed = min((batch_idx + 1) * batch_size, filtered_count)
        print(f"  Classified: {processed}/{filtered_count} ({processed/filtered_count*100:.1f}%)")
        
        # Avoid API rate limiting (GPT-4o usually has higher rate limits)
        time.sleep(0.5)  # Pause between batches
    
    print(f"Classification complete: {streamer_name}")
    
    return {
        'streamer': streamer_name,
        'total_comments': total_comments,
        'filtered_comments': filtered_count,
        'categories': dict(categories),
        'classifications': classifications
    }


def calculate_percentages(results: Dict[str, any]) -> Dict[str, float]:
    """Calculate percentage for each category"""
    categories = results['categories']
    total = sum(categories.values())
    
    if total == 0:
        return {cat: 0.0 for cat in FUNNEL_CATEGORIES.keys()}
    
    percentages = {}
    for category in FUNNEL_CATEGORIES.keys():
        count = categories.get(category, 0)
        percentages[category] = (count / total) * 100
    
    return percentages


def generate_report(all_results: List[Dict[str, any]]):
    """Generate report"""
    # Create summary report
    summary_data = []
    
    for result in all_results:
        streamer = result['streamer']
        percentages = calculate_percentages(result)
        
        summary_data.append({
            'Streamer': streamer,
            'Total Comments': result['total_comments'],
            'Filtered Comments': result['filtered_comments'],
            'Filter Rate (%)': f"{result['filtered_comments']/result['total_comments']*100:.1f}",
            'Intent to Play (%)': f"{percentages['Intent to Play']:.1f}",
            'Pre-Install (%)': f"{percentages['Pre-Install']:.1f}",
            'Post-Play (%)': f"{percentages['Post-Play']:.1f}",
            'Loyalty (%)': f"{percentages['Loyalty']:.1f}",
            'Irrelevant (%)': f"{percentages['Irrelevant']:.1f}",
        })
    
    # Save summary report
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, "summary_report.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary report saved: {summary_path}")
    
    # Save detailed classification results for each streamer
    for result in all_results:
        streamer = result['streamer']
        classifications_df = pd.DataFrame(result['classifications'])
        detail_path = os.path.join(OUTPUT_DIR, f"{streamer}_classifications.csv")
        classifications_df.to_csv(detail_path, index=False)
        print(f"Detailed results saved: {detail_path}")
    
    # Print summary table
    print("\n" + "="*100)
    print("Summary Report")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)


def main():
    """Main function"""
    # Check OpenAI API key
    if not OPENAI_API_KEY:
        print("Error: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key'")
        return
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Find all CSV files
    csv_files = list(Path(REPLAY_DIR).glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {REPLAY_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Process all files
    all_results = []
    
    for csv_file in csv_files:
        try:
            result = process_csv_file(str(csv_file), client)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
            continue
    
    # Generate report
    if all_results:
        generate_report(all_results)
    else:
        print("No files were successfully processed")


if __name__ == "__main__":
    main()

