#!/usr/bin/env python3
"""
Generate an interactive HTML webpage to visualize insights
Supports filtering by module (theme), sentiment, and priority
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List


def load_latest_insights(results_dir: str = "insights_results") -> Dict:
    """Load the latest insights JSON file"""
    # Find all insights JSON files
    pattern = os.path.join(results_dir, "insights_*.json")
    insight_files = glob.glob(pattern)
    
    if not insight_files:
        raise FileNotFoundError(f"No insights files found in {results_dir}")
    
    # Get the latest file
    latest_file = max(insight_files, key=os.path.getmtime)
    print(f"📂 Loading insights from: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def merge_insights_files(results_dir: str = "insights_results", prefer_chinese: bool = True) -> Dict:
    """Merge all insights JSON files into one, preferring Chinese content"""
    # Find all insights JSON files
    pattern = os.path.join(results_dir, "insights_*.json")
    insight_files = glob.glob(pattern)
    
    if not insight_files:
        raise FileNotFoundError(f"No insights files found in {results_dir}")
    
    print(f"📂 Found {len(insight_files)} insights files")
    
    # Check which files contain Chinese content
    chinese_files = []
    english_files = []
    
    for file_path in insight_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            groups = data.get('groups', {})
            
            has_chinese = False
            for group_key, group_data in groups.items():
                insights = group_data.get('key_insights', [])
                if insights:
                    first_insight = insights[0].get('insight', '')
                    # Check if contains Chinese characters
                    if any('\u4e00' <= char <= '\u9fff' for char in first_insight):
                        has_chinese = True
                        break
            
            if has_chinese:
                chinese_files.append(file_path)
            else:
                english_files.append(file_path)
    
    # Prefer Chinese files if requested
    if prefer_chinese and chinese_files:
        print(f"   ✅ Found {len(chinese_files)} Chinese files, {len(english_files)} English files")
        print(f"   📝 Using Chinese files only (to match insights file format)")
        files_to_use = chinese_files
    else:
        files_to_use = insight_files
    
    # Merge all groups
    merged_groups = {}
    all_timestamps = []
    
    for file_path in files_to_use:
        print(f"   Loading: {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract timestamp
        timestamp = data.get('metadata', {}).get('timestamp', '')
        if timestamp:
            all_timestamps.append(timestamp)
        
        # Merge groups
        for group_key, group_data in data.get('groups', {}).items():
            # If group already exists, prefer Chinese content or more insights
            if group_key not in merged_groups:
                merged_groups[group_key] = group_data
            else:
                # Check if new data is Chinese
                new_insights = group_data.get('key_insights', [])
                existing_insights = merged_groups[group_key].get('key_insights', [])
                
                new_is_chinese = False
                existing_is_chinese = False
                
                if new_insights:
                    new_insight_text = new_insights[0].get('insight', '')
                    new_is_chinese = any('\u4e00' <= char <= '\u9fff' for char in new_insight_text)
                
                if existing_insights:
                    existing_insight_text = existing_insights[0].get('insight', '')
                    existing_is_chinese = any('\u4e00' <= char <= '\u9fff' for char in existing_insight_text)
                
                # Prefer Chinese content, or more insights
                if new_is_chinese and not existing_is_chinese:
                    merged_groups[group_key] = group_data
                elif new_is_chinese == existing_is_chinese:
                    # Both same language, keep the one with more insights
                    if len(new_insights) > len(existing_insights):
                        merged_groups[group_key] = group_data
                # If existing is Chinese and new is not, keep existing
    
    # Create merged data structure
    merged_data = {
        'metadata': {
            'timestamp': max(all_timestamps) if all_timestamps else '',
            'total_groups': len(merged_groups),
            'source_files': len(files_to_use),
            'chinese_files': len(chinese_files),
            'english_files': len(english_files)
        },
        'groups': merged_groups
    }
    
    print(f"✅ Merged {len(merged_groups)} unique groups from {len(files_to_use)} files")
    return merged_data


def generate_html(insights_data: Dict, output_file: str = "insights_dashboard.html", 
                  source_csv_file: str = None):
    """Generate interactive HTML dashboard"""
    
    # Extract all groups and their insights
    groups = insights_data.get('groups', {})
    
    # Load comment counts from CSV file if provided
    comment_counts_by_group = {}
    if source_csv_file and os.path.exists(source_csv_file):
        try:
            import pandas as pd
            df = pd.read_csv(source_csv_file, lineterminator='\n')
            
            # Group by module and sentiment
            grouping_cols = ['_module_name', '_sentiment_name']
            available_cols = [col for col in grouping_cols if col in df.columns]
            
            if available_cols:
                grouped = df.groupby(available_cols, dropna=False)
                for group_key, group_df in grouped:
                    if isinstance(group_key, tuple) and len(group_key) >= 2:
                        module_name = str(group_key[0]) if pd.notna(group_key[0]) else 'Unknown'
                        sentiment_name = str(group_key[1]) if pd.notna(group_key[1]) else 'Unknown'
                        group_key_str = f"{module_name}|{sentiment_name}"
                        comment_counts_by_group[group_key_str] = len(group_df)
                print(f"   📊 Loaded comment counts for {len(comment_counts_by_group)} groups from CSV file")
        except Exception as e:
            print(f"   ⚠️  Unable to load comment counts from CSV file: {e}")
    
    # Collect all insights for filtering
    all_insights = []
    all_priority_themes = []
    
    # Also collect comment counts from insights data metadata if available
    for group_key, group_data in groups.items():
        # Split group key (format: "Module|Sentiment" or "Module|Submodule|Sentiment")
        parts = group_key.split('|')
        
        # Handle different formats
        if len(parts) == 2:
            # Format: "Module|Sentiment"
            module_name = parts[0]
            sentiment_name = parts[1]
        elif len(parts) >= 3:
            # Format: "Module|Submodule|Sentiment" - use first and last part
            module_name = parts[0]
            sentiment_name = parts[-1]  # Last part is sentiment
        else:
            # Fallback if format is different
            module_name = parts[0] if parts else 'Unknown'
            sentiment_name = 'Unknown'
        
        # Clean up module and sentiment names (remove common issues)
        module_name = module_name.strip()
        sentiment_name = sentiment_name.strip()
        
        # Skip if module or sentiment is a priority value (data quality issue)
        priority_values = ['High', 'Medium', 'Low']
        if module_name in priority_values or sentiment_name in priority_values:
            # Try to extract from group_data if available
            if 'sentiment_analysis' in group_data:
                sentiment_analysis = group_data.get('sentiment_analysis', {})
                # Try to get sentiment from sentiment_analysis
                if 'overall_sentiment' in sentiment_analysis:
                    possible_sentiment = sentiment_analysis['overall_sentiment']
                    if possible_sentiment not in priority_values:
                        sentiment_name = possible_sentiment
            continue
        
        # Extract key insights
        for insight in group_data.get('key_insights', []):
            insight_item = {
                'module': module_name,
                'sentiment': sentiment_name,
                'importance': insight.get('importance', 'Unknown'),
                'insight': insight.get('insight', ''),
                'supporting_comments': insight.get('supporting_comments', []),
                'group_key': group_key
            }
            all_insights.append(insight_item)
        
        # Extract priority themes
        for theme in group_data.get('priority_themes', []):
            theme_item = {
                'module': module_name,
                'sentiment': sentiment_name,
                'theme_name': theme.get('theme_name', ''),
                'why_important': theme.get('why_important', ''),
                'supporting_comments': theme.get('supporting_comments', []),
                'group_key': group_key
            }
            all_priority_themes.append(theme_item)
    
    # Get unique values for filters
    # Include modules from both insights and priority_themes
    unique_modules_raw = set()
    for item in all_insights:
        if item['module'] and item['module'] != 'Unknown':
            unique_modules_raw.add(item['module'])
    for item in all_priority_themes:
        if item['module'] and item['module'] != 'Unknown':
            unique_modules_raw.add(item['module'])
    
    # Normalize module names
    module_mapping = {
        'Monetization': 'Monetization'
    }
    unique_modules = []
    for m in unique_modules_raw:
        if m in module_mapping:
            if module_mapping[m] not in unique_modules:
                unique_modules.append(module_mapping[m])
        else:
            if m not in unique_modules:
                unique_modules.append(m)
    unique_modules = sorted(unique_modules)
    
    # Normalize sentiment values
    sentiment_mapping = {
        'Positive': 'Positive',
        'Negative': 'Negative',
        'Neutral': 'Neutral'
    }
    unique_sentiments_raw = set(item['sentiment'] for item in all_insights if item['sentiment'] and item['sentiment'] != 'Unknown')
    normalized_sentiments = []
    for s in unique_sentiments_raw:
        if s in sentiment_mapping:
            if sentiment_mapping[s] not in normalized_sentiments:
                normalized_sentiments.append(sentiment_mapping[s])
        else:
            # Already in Chinese or other format
            if s not in normalized_sentiments:
                normalized_sentiments.append(s)
    unique_sentiments = sorted(normalized_sentiments)
    
    # Normalize priority values
    priority_mapping = {
        'High': 'High',
        'Medium': 'Medium',
        'Low': 'Low'
    }
    unique_priorities_raw = set(item['importance'] for item in all_insights if item['importance'] and item['importance'] != 'Unknown')
    normalized_priorities = []
    for p in unique_priorities_raw:
        if p in priority_mapping:
            if priority_mapping[p] not in normalized_priorities:
                normalized_priorities.append(priority_mapping[p])
        else:
            if p not in normalized_priorities:
                normalized_priorities.append(p)
    unique_priorities = sorted(normalized_priorities)
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S3 PVE Halloween Version YouTube Video Comment Insights Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
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
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .filters {{
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .filter-group label {{
            font-weight: 600;
            color: #495057;
            font-size: 0.9em;
        }}
        
        .filter-group select {{
            padding: 10px 15px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            font-size: 1em;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
            min-width: 180px;
        }}
        
        .filter-group select:hover {{
            border-color: #667eea;
        }}
        
        .filter-group select:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        
        .stats {{
            padding: 20px 30px;
            background: white;
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .chart-container {{
            padding: 30px;
            background: white;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .chart-title {{
            font-size: 1.3em;
            font-weight: 700;
            color: #212529;
            margin-bottom: 20px;
        }}
        
        .chart-wrapper {{
            position: relative;
            height: 320px;
            margin-top: 20px;
            padding-left: 50px;
        }}
        
        .bar-chart {{
            display: flex;
            align-items: flex-end;
            gap: 20px;
            height: 220px;
            padding-right: 20px;
            position: relative;
        }}
        
        .bar-group {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: stretch;
            gap: 0;
            min-width: 60px;
            max-width: 120px;
        }}
        
        .bar-segments-container {{
            display: flex;
            flex-direction: column;
            align-items: stretch;
            justify-content: flex-end;
            height: 100%;
            min-height: 200px;
        }}
        
        .bar-segment {{
            width: 100%;
            border-radius: 2px;
            transition: opacity 0.3s;
            cursor: pointer;
            position: relative;
            min-height: 2px;
        }}
        
        .bar-segment:hover {{
            opacity: 0.8;
        }}
        
        .bar-segment-positive {{
            background: #4caf50;
        }}
        
        .bar-segment-negative {{
            background: #f44336;
        }}
        
        .bar-segment-neutral {{
            background: #ff9800;
        }}
        
        .bar-label {{
            margin-top: 10px;
            font-size: 0.85em;
            color: #495057;
            text-align: center;
            font-weight: 600;
            word-break: break-word;
            line-height: 1.2;
        }}
        
        .bar-count {{
            margin-top: 5px;
            font-size: 0.75em;
            color: #6c757d;
            text-align: center;
        }}
        
        .chart-legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
        
        .legend-label {{
            font-size: 0.9em;
            color: #495057;
        }}
        
        .y-axis {{
            position: absolute;
            left: 0;
            top: 0;
            bottom: 100px;
            width: 45px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding-right: 10px;
            align-items: flex-end;
            border-right: 1px solid #dee2e6;
        }}
        
        .y-axis-label {{
            font-size: 0.75em;
            color: #6c757d;
            line-height: 1;
        }}
        
        .stat-item {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: 700;
            color: #667eea;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            color: #6c757d;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .tab {{
            padding: 15px 30px;
            background: none;
            border: none;
            font-size: 1.1em;
            font-weight: 600;
            color: #6c757d;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }}
        
        .tab:hover {{
            color: #667eea;
        }}
        
        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .insights-grid {{
            display: grid;
            gap: 25px;
        }}
        
        .insight-card {{
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 25px;
            transition: all 0.3s;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .insight-card:hover {{
            border-color: #667eea;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
            transform: translateY(-2px);
        }}
        
        .insight-header {{
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .insight-badges {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .badge {{
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            white-space: nowrap;
        }}
        
        .badge-module {{
            background: #e3f2fd;
            color: #1976d2;
        }}
        
        .badge-sentiment-positive {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        
        .badge-sentiment-negative {{
            background: #ffebee;
            color: #c62828;
        }}
        
        .badge-sentiment-neutral {{
            background: #fff3e0;
            color: #f57c00;
        }}
        
        .badge-importance-high {{
            background: #ffebee;
            color: #c62828;
        }}
        
        .badge-importance-medium {{
            background: #fff9c4;
            color: #f57f17;
        }}
        
        .badge-importance-low {{
            background: #e8f5e9;
            color: #388e3c;
        }}
        
        .insight-text {{
            font-size: 1.1em;
            line-height: 1.8;
            color: #212529;
            margin-bottom: 20px;
        }}
        
        .supporting-comments {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #f1f3f5;
        }}
        
        .supporting-comments h4 {{
            font-size: 0.95em;
            color: #6c757d;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .comment-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
            font-size: 0.95em;
            line-height: 1.6;
            color: #495057;
        }}
        
        .priority-theme-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px solid #dee2e6;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
        }}
        
        .priority-theme-title {{
            font-size: 1.3em;
            font-weight: 700;
            color: #212529;
            margin-bottom: 15px;
        }}
        
        .priority-theme-why {{
            font-size: 1em;
            line-height: 1.8;
            color: #495057;
            margin-bottom: 20px;
        }}
        
        .no-results {{
            text-align: center;
            padding: 60px 20px;
            color: #6c757d;
        }}
        
        .no-results-icon {{
            font-size: 4em;
            margin-bottom: 20px;
        }}
        
        .reset-filters {{
            padding: 10px 20px;
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .reset-filters:hover {{
            background: #5a6268;
        }}
        
        @media (max-width: 768px) {{
            .filters {{
                flex-direction: column;
                align-items: stretch;
            }}
            
            .filter-group select {{
                width: 100%;
            }}
            
            .stats {{
                flex-direction: column;
                gap: 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>S3 PVE Halloween Version YouTube Video Comment Insights Analysis</h1>
            <p>October YouTube KOC comments (NyteFalli, PW Games, Belle Teke Harald, BnFire R6)</p>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📊 Sentiment Distribution by Theme (Comment Count)</div>
            <div class="chart-wrapper">
                <div class="y-axis" id="y-axis">
                    <!-- Y-axis labels will be inserted here -->
                </div>
                <div class="bar-chart" id="bar-chart">
                    <!-- Chart bars will be inserted here -->
                </div>
            </div>
            <div class="chart-legend">
                <div class="legend-item">
                    <div class="legend-color bar-segment-positive"></div>
                    <span class="legend-label">Positive</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color bar-segment-neutral"></div>
                    <span class="legend-label">Neutral</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color bar-segment-negative"></div>
                    <span class="legend-label">Negative</span>
                </div>
            </div>
        </div>
        
        <div class="filters">
            <div class="filter-group">
                <label>📊 Theme (Module)</label>
                <select id="filter-module">
                    <option value="all">All Themes</option>
                    {chr(10).join([f'<option value="{module}">{module}</option>' for module in unique_modules])}
                </select>
            </div>
            
            <div class="filter-group">
                <label>😊 Sentiment</label>
                <select id="filter-sentiment">
                    <option value="all">All Sentiments</option>
                    {chr(10).join([f'<option value="{sentiment}">{sentiment}</option>' for sentiment in unique_sentiments])}
                </select>
            </div>
            
            <div class="filter-group">
                <label>⭐ Priority (Importance)</label>
                <select id="filter-priority">
                    <option value="all">All Priorities</option>
                    {chr(10).join([f'<option value="{priority}">{priority}</option>' for priority in unique_priorities])}
                </select>
            </div>
            
            <button class="reset-filters" onclick="resetFilters()">Reset Filters</button>
        </div>
        
            <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="stat-total-insights">0</div>
                <div class="stat-label">Total Insights</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="stat-total-themes">0</div>
                <div class="stat-label">Priority Themes</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="stat-high-priority">0</div>
                <div class="stat-label">High Priority</div>
            </div>
        </div>
        
        <div class="content">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('insights')">Key Insights</button>
                <button class="tab" onclick="switchTab('themes')">Priority Themes</button>
            </div>
            
            <div id="tab-insights" class="tab-content active">
                <div class="insights-grid" id="insights-container">
                    <!-- Insights will be inserted here -->
                </div>
            </div>
            
            <div id="tab-themes" class="tab-content">
                <div class="insights-grid" id="themes-container">
                    <!-- Priority themes will be inserted here -->
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Data
        const insightsData = {json.dumps(all_insights, ensure_ascii=False, indent=2)};
        const themesData = {json.dumps(all_priority_themes, ensure_ascii=False, indent=2)};
        const commentCountsByGroup = {json.dumps(comment_counts_by_group, ensure_ascii=False, indent=2)};
        
        // Current filters
        let currentFilters = {{
            module: 'all',
            moduleEnglish: 'all',
            sentiment: 'all',
            sentimentEnglish: 'all',
            priority: 'all',
            priorityEnglish: 'all'
        }};
        
        // Current tab
        let currentTab = 'insights';
        
        // Filter functions
        function filterData() {{
            let moduleFilter = document.getElementById('filter-module').value;
            let sentimentFilter = document.getElementById('filter-sentiment').value;
            let priorityFilter = document.getElementById('filter-priority').value;
            
            // Convert Chinese filter values back to English for comparison with data
            const moduleReverseMap = {{
                '商业化': 'Monetization'
            }};
            const sentimentReverseMap = {{
                '正面': 'Positive',
                '负面': 'Negative',
                '中性': 'Neutral'
            }};
            const priorityReverseMap = {{
                '高': 'High',
                '中': 'Medium',
                '低': 'Low'
            }};
            
            // Keep original for display, but also store English equivalent for filtering
            currentFilters = {{
                module: moduleFilter,
                moduleEnglish: moduleReverseMap[moduleFilter] || moduleFilter,
                sentiment: sentimentFilter,
                sentimentEnglish: sentimentReverseMap[sentimentFilter] || sentimentFilter,
                priority: priorityFilter,
                priorityEnglish: priorityReverseMap[priorityFilter] || priorityFilter
            }};
            
            renderContent();
            updateStats();
        }}
        
        function resetFilters() {{
            document.getElementById('filter-module').value = 'all';
            document.getElementById('filter-sentiment').value = 'all';
            document.getElementById('filter-priority').value = 'all';
            filterData();
        }}
        
        function switchTab(tabName) {{
            currentTab = tabName;
            
            // Update tab buttons
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            event.target.classList.add('active');
            
            // Update tab content
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            document.getElementById(`tab-${{tabName}}`).classList.add('active');
            
            renderContent();
        }}
        
        function renderContent() {{
            if (currentTab === 'insights') {{
                renderInsights();
            }} else {{
                renderThemes();
            }}
        }}
        
        function renderInsights() {{
            const container = document.getElementById('insights-container');
            const filtered = insightsData.filter(item => {{
                if (currentFilters.module !== 'all') {{
                    // Check both Chinese and English module
                    const itemModule = item.module || '';
                    const filterModule = currentFilters.module;
                    const filterModuleEnglish = currentFilters.moduleEnglish || filterModule;
                    if (itemModule !== filterModule && itemModule !== filterModuleEnglish) return false;
                }}
                if (currentFilters.sentiment !== 'all') {{
                    // Check both Chinese and English sentiment
                    const itemSentiment = item.sentiment || '';
                    const filterSentiment = currentFilters.sentiment;
                    const filterSentimentEnglish = currentFilters.sentimentEnglish || filterSentiment;
                    // Also check reverse mapping (if data has English but filter is Chinese)
                    const sentimentMap = {{
                        'Positive': '正面',
                        'Negative': '负面',
                        'Neutral': '中性'
                    }};
                    const itemSentimentChinese = sentimentMap[itemSentiment] || itemSentiment;
                    if (itemSentiment !== filterSentiment && 
                        itemSentiment !== filterSentimentEnglish &&
                        itemSentimentChinese !== filterSentiment &&
                        itemSentimentChinese !== filterSentimentEnglish) return false;
                }}
                if (currentFilters.priority !== 'all') {{
                    // Normalize importance for comparison (handle both Chinese and English)
                    const itemImportance = item.importance || '';
                    const filterPriority = currentFilters.priority;
                    const filterPriorityEnglish = currentFilters.priorityEnglish || filterPriority;
                    if (itemImportance !== filterPriority && itemImportance !== filterPriorityEnglish) return false;
                }}
                return true;
            }});
            
            if (filtered.length === 0) {{
                container.innerHTML = `
                    <div class="no-results">
                        <div class="no-results-icon">🔍</div>
                        <h3>没有找到匹配的洞察</h3>
                        <p>请尝试调整筛选条件</p>
                    </div>
                `;
                return;
            }}
            
            container.innerHTML = filtered.map(item => {{
                // Normalize importance for CSS class (handle both Chinese and English)
                let importanceClass = 'badge-importance-medium';
                const importance = item.importance || '';
                if (importance === '高' || importance.toLowerCase() === 'high') {{
                    importanceClass = 'badge-importance-high';
                }} else if (importance === '低' || importance.toLowerCase() === 'low') {{
                    importanceClass = 'badge-importance-low';
                }} else if (importance === '中' || importance.toLowerCase() === 'medium') {{
                    importanceClass = 'badge-importance-medium';
                }}
                
                // Display importance (prefer Chinese)
                let importanceDisplay = importance;
                if (importance === 'High') importanceDisplay = '高';
                else if (importance === 'Medium') importanceDisplay = '中';
                else if (importance === 'Low') importanceDisplay = '低';
                
                // Display sentiment (prefer Chinese)
                let sentimentDisplay = item.sentiment || '';
                let sentimentClass = 'badge-sentiment-neutral';
                if (sentimentDisplay === 'Positive' || sentimentDisplay === '正面') {{
                    sentimentDisplay = '正面';
                    sentimentClass = 'badge-sentiment-positive';
                }} else if (sentimentDisplay === 'Negative' || sentimentDisplay === '负面') {{
                    sentimentDisplay = '负面';
                    sentimentClass = 'badge-sentiment-negative';
                }} else if (sentimentDisplay === 'Neutral' || sentimentDisplay === '中性') {{
                    sentimentDisplay = '中性';
                    sentimentClass = 'badge-sentiment-neutral';
                }}
                
                // Display module (prefer Chinese)
                let moduleDisplay = item.module || '';
                if (moduleDisplay === 'Monetization') moduleDisplay = '商业化';
                
                const commentsHtml = item.supporting_comments.map(comment => 
                    `<div class="comment-item">${{escapeHtml(comment)}}</div>`
                ).join('');
                
                return `
                    <div class="insight-card">
                        <div class="insight-header">
                            <div class="insight-badges">
                                <span class="badge badge-module">${{moduleDisplay}}</span>
                                <span class="badge ${{sentimentClass}}">${{sentimentDisplay}}</span>
                                <span class="badge ${{importanceClass}}">${{importanceDisplay}}</span>
                            </div>
                        </div>
                        <div class="insight-text">${{escapeHtml(item.insight)}}</div>
                        ${{item.supporting_comments && item.supporting_comments.length > 0 ? `
                        <div class="supporting-comments">
                            <h4>📝 支撑评论 (${{item.supporting_comments.length}}条)</h4>
                            ${{commentsHtml}}
                        </div>
                        ` : ''}}
                    </div>
                `;
            }}).join('');
        }}
        
        function renderThemes() {{
            const container = document.getElementById('themes-container');
            const filtered = themesData.filter(item => {{
                if (currentFilters.module !== 'all') {{
                    // Check both Chinese and English module
                    const itemModule = item.module || '';
                    const filterModule = currentFilters.module;
                    const filterModuleEnglish = currentFilters.moduleEnglish || filterModule;
                    if (itemModule !== filterModule && itemModule !== filterModuleEnglish) return false;
                }}
                if (currentFilters.sentiment !== 'all') {{
                    // Check both Chinese and English sentiment
                    const itemSentiment = item.sentiment || '';
                    const filterSentiment = currentFilters.sentiment;
                    const filterSentimentEnglish = currentFilters.sentimentEnglish || filterSentiment;
                    // Also check reverse mapping (if data has English but filter is Chinese)
                    const sentimentMap = {{
                        'Positive': '正面',
                        'Negative': '负面',
                        'Neutral': '中性'
                    }};
                    const itemSentimentChinese = sentimentMap[itemSentiment] || itemSentiment;
                    if (itemSentiment !== filterSentiment && 
                        itemSentiment !== filterSentimentEnglish &&
                        itemSentimentChinese !== filterSentiment &&
                        itemSentimentChinese !== filterSentimentEnglish) return false;
                }}
                return true;
            }});
            
            if (filtered.length === 0) {{
                container.innerHTML = `
                    <div class="no-results">
                        <div class="no-results-icon">🔍</div>
                        <h3>没有找到匹配的主题</h3>
                        <p>请尝试调整筛选条件</p>
                    </div>
                `;
                return;
            }}
            
            container.innerHTML = filtered.map(item => {{
                // Display sentiment (prefer Chinese)
                let sentimentDisplay = item.sentiment || '';
                let sentimentClass = 'badge-sentiment-neutral';
                if (sentimentDisplay === 'Positive' || sentimentDisplay === '正面') {{
                    sentimentDisplay = '正面';
                    sentimentClass = 'badge-sentiment-positive';
                }} else if (sentimentDisplay === 'Negative' || sentimentDisplay === '负面') {{
                    sentimentDisplay = '负面';
                    sentimentClass = 'badge-sentiment-negative';
                }} else if (sentimentDisplay === 'Neutral' || sentimentDisplay === '中性') {{
                    sentimentDisplay = '中性';
                    sentimentClass = 'badge-sentiment-neutral';
                }}
                
                // Display module (prefer Chinese)
                let moduleDisplay = item.module || '';
                if (moduleDisplay === 'Monetization') moduleDisplay = '商业化';
                
                const commentsHtml = item.supporting_comments.map(comment => 
                    `<div class="comment-item">${{escapeHtml(comment)}}</div>`
                ).join('');
                
                return `
                    <div class="priority-theme-card">
                        <div class="insight-header">
                            <div class="insight-badges">
                                <span class="badge badge-module">${{moduleDisplay}}</span>
                                <span class="badge ${{sentimentClass}}">${{sentimentDisplay}}</span>
                            </div>
                        </div>
                        <div class="priority-theme-title">${{escapeHtml(item.theme_name)}}</div>
                        <div class="priority-theme-why">${{escapeHtml(item.why_important)}}</div>
                        ${{item.supporting_comments && item.supporting_comments.length > 0 ? `
                        <div class="supporting-comments">
                            <h4>📝 支撑评论 (${{item.supporting_comments.length}}条)</h4>
                            ${{commentsHtml}}
                        </div>
                        ` : ''}}
                    </div>
                `;
            }}).join('');
        }}
        
        function updateStats() {{
            const filtered = insightsData.filter(item => {{
                if (currentFilters.module !== 'all') {{
                    const itemModule = item.module || '';
                    const filterModule = currentFilters.module;
                    const filterModuleEnglish = currentFilters.moduleEnglish || filterModule;
                    if (itemModule !== filterModule && itemModule !== filterModuleEnglish) return false;
                }}
                if (currentFilters.sentiment !== 'all') {{
                    const itemSentiment = item.sentiment || '';
                    const filterSentiment = currentFilters.sentiment;
                    const filterSentimentEnglish = currentFilters.sentimentEnglish || filterSentiment;
                    if (itemSentiment !== filterSentiment && itemSentiment !== filterSentimentEnglish) return false;
                }}
                if (currentFilters.priority !== 'all') {{
                    const itemImportance = item.importance || '';
                    const filterPriority = currentFilters.priority;
                    const filterPriorityEnglish = currentFilters.priorityEnglish || filterPriority;
                    if (itemImportance !== filterPriority && itemImportance !== filterPriorityEnglish) return false;
                }}
                return true;
            }});
            
            const filteredThemes = themesData.filter(item => {{
                if (currentFilters.module !== 'all') {{
                    const itemModule = item.module || '';
                    const filterModule = currentFilters.module;
                    const filterModuleEnglish = currentFilters.moduleEnglish || filterModule;
                    if (itemModule !== filterModule && itemModule !== filterModuleEnglish) return false;
                }}
                if (currentFilters.sentiment !== 'all') {{
                    const itemSentiment = item.sentiment || '';
                    const filterSentiment = currentFilters.sentiment;
                    const filterSentimentEnglish = currentFilters.sentimentEnglish || filterSentiment;
                    if (itemSentiment !== filterSentiment && itemSentiment !== filterSentimentEnglish) return false;
                }}
                return true;
            }});
            
            document.getElementById('stat-total-insights').textContent = filtered.length;
            document.getElementById('stat-total-themes').textContent = filteredThemes.length;
            // Count high priority (both Chinese and English)
            const highPriorityCount = filtered.filter(item => {{
                const importance = item.importance || '';
                return importance === '高' || importance === 'High';
            }}).length;
            document.getElementById('stat-high-priority').textContent = highPriorityCount;
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        // Generate stacked bar chart using comment counts
        function renderChart() {{
            // Count comments by module and sentiment from commentCountsByGroup
            const moduleSentimentCounts = {{}};
            
            // Process comment counts from CSV data
            Object.keys(commentCountsByGroup).forEach(groupKey => {{
                const parts = groupKey.split('|');
                if (parts.length >= 2) {{
                    let module = parts[0].trim();
                    let sentiment = parts[1].trim();
                    
                    // Normalize module name
                    if (module === 'Monetization') module = '商业化';
                    
                    // Normalize sentiment
                    if (sentiment === 'Positive') sentiment = '正面';
                    else if (sentiment === 'Negative') sentiment = '负面';
                    else if (sentiment === 'Neutral') sentiment = '中性';
                    
                    if (!moduleSentimentCounts[module]) {{
                        moduleSentimentCounts[module] = {{
                            '正面': 0,
                            '负面': 0,
                            '中性': 0
                        }};
                    }}
                    
                    const count = commentCountsByGroup[groupKey] || 0;
                    if (moduleSentimentCounts[module][sentiment] !== undefined) {{
                        moduleSentimentCounts[module][sentiment] += count;
                    }}
                }}
            }});
            
            // Get all modules
            const modules = Object.keys(moduleSentimentCounts).sort();
            
            if (modules.length === 0) {{
                document.getElementById('bar-chart').innerHTML = '<p style="text-align: center; color: #6c757d; padding: 40px;">暂无数据</p>';
                document.getElementById('y-axis').innerHTML = '';
                return;
            }}
            
            // Calculate max count for scaling
            let maxCount = 0;
            modules.forEach(module => {{
                const counts = moduleSentimentCounts[module];
                const total = counts['正面'] + counts['负面'] + counts['中性'];
                if (total > maxCount) maxCount = total;
            }});
            
            // Generate Y-axis labels
            const yAxisLabels = [];
            const numLabels = 5;
            for (let i = 0; i <= numLabels; i++) {{
                const value = Math.round((maxCount / numLabels) * (numLabels - i));
                yAxisLabels.push(`<div class="y-axis-label">${{value}}</div>`);
            }}
            document.getElementById('y-axis').innerHTML = yAxisLabels.join('');
            
            // Generate bars
            const barHtml = modules.map(module => {{
                const counts = moduleSentimentCounts[module];
                const positive = counts['正面'] || 0;
                const negative = counts['负面'] || 0;
                const neutral = counts['中性'] || 0;
                const total = positive + negative + neutral;
                
                if (total === 0) return '';
                
                // Calculate heights as percentages of maxCount
                const maxHeight = 200; // Fixed max height in pixels
                const positiveHeight = maxCount > 0 ? (positive / maxCount) * maxHeight : 0;
                const negativeHeight = maxCount > 0 ? (negative / maxCount) * maxHeight : 0;
                const neutralHeight = maxCount > 0 ? (neutral / maxCount) * maxHeight : 0;
                
                return `
                    <div class="bar-group">
                        <div class="bar-segments-container">
                            ${{negative > 0 ? `<div class="bar-segment bar-segment-negative" style="height: ${{negativeHeight}}px;" title="负面: ${{negative}} 条评论"></div>` : ''}}
                            ${{neutral > 0 ? `<div class="bar-segment bar-segment-neutral" style="height: ${{neutralHeight}}px;" title="中性: ${{neutral}} 条评论"></div>` : ''}}
                            ${{positive > 0 ? `<div class="bar-segment bar-segment-positive" style="height: ${{positiveHeight}}px;" title="正面: ${{positive}} 条评论"></div>` : ''}}
                        </div>
                        <div class="bar-label">${{escapeHtml(module)}}</div>
                        <div class="bar-count">${{total}} 条</div>
                    </div>
                `;
            }}).join('');
            
            document.getElementById('bar-chart').innerHTML = barHtml;
        }}
        
        // Initialize
        document.getElementById('filter-module').addEventListener('change', filterData);
        document.getElementById('filter-sentiment').addEventListener('change', filterData);
        document.getElementById('filter-priority').addEventListener('change', filterData);
        
        // Initial render
        renderContent();
        updateStats();
        renderChart();
    </script>
</body>
</html>"""
    
    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Generated HTML dashboard: {output_file}")


def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Generate interactive HTML dashboard for insights")
    ap.add_argument("--results-dir", type=str, default="insights_results",
                   help="Directory containing insights JSON files")
    ap.add_argument("--output", type=str, default="insights_dashboard.html",
                   help="Output HTML file name")
    ap.add_argument("--insights-file", type=str, default=None,
                   help="Specific insights JSON file to use (optional)")
    ap.add_argument("--merge", action="store_true",
                   help="Merge all insights files instead of using latest")
    ap.add_argument("--prefer-chinese", action="store_true", default=True,
                   help="Prefer Chinese content when merging (default: True)")
    
    args = ap.parse_args()
    
    # Load insights data
    if args.insights_file:
        print(f"📂 Loading insights from: {args.insights_file}")
        with open(args.insights_file, 'r', encoding='utf-8') as f:
            insights_data = json.load(f)
    elif args.merge:
        insights_data = merge_insights_files(args.results_dir, prefer_chinese=args.prefer_chinese)
    else:
        insights_data = load_latest_insights(args.results_dir)
    
    # Try to find source CSV file from metadata
    source_csv_file = None
    if 'metadata' in insights_data:
        source_file = insights_data['metadata'].get('source_file', '')
        if source_file and os.path.exists(source_file):
            source_csv_file = source_file
        else:
            # Try to find the CSV file in current directory
            import glob
            csv_files = glob.glob('comments_processed_*.csv')
            if csv_files:
                # Use the latest one
                source_csv_file = max(csv_files, key=os.path.getmtime)
                print(f"   📂 找到CSV文件: {source_csv_file}")
    
    # Generate HTML
    generate_html(insights_data, args.output, source_csv_file=source_csv_file)
    
    print(f"\n🎉 Dashboard generated successfully!")
    print(f"📁 Open {args.output} in your browser to view the insights")


if __name__ == "__main__":
    main()

