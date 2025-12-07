#!/usr/bin/env python3
"""
Theme Insight Generator
Aggregates comments by category (module), module name (submodule), and sentiment.
Uses Claude Haiku 4.5 to map comments to candidate themes.
Uses Claude Sonnet 4.5 to reduce and generate insights.
"""

import os
import pandas as pd
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import yaml
import numpy as np

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️ anthropic not installed. Install with: pip install anthropic")


class ThemeInsightGenerator:
    def __init__(self, processed_csv_file: str, config_file: str = "themes_config_new.yaml", 
                 api_key: str = None, output_dir: str = None):
        """
        Initialize the Theme Insight Generator
        
        Args:
            processed_csv_file: Path to the processed comments CSV file
            config_file: Path to the theme configuration YAML file
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            output_dir: Output directory for results
        """
        self.processed_csv_file = processed_csv_file
        self.config_file = config_file
        self.output_dir = output_dir or ""
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Anthropic setup
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            # Try to get from Browser API key format if provided
            browser_key = os.environ.get("BROWSER_API_KEY", "")
            if browser_key:
                self.api_key = browser_key
            else:
                raise ValueError("Anthropic API key is required. Provide via api_key parameter, ANTHROPIC_API_KEY, or BROWSER_API_KEY env var.")
        
        self.anthropic_client = Anthropic(api_key=self.api_key)
        self.haiku_model = "claude-haiku-4-5-20251001"  # Claude Haiku 4.5
        self.sonnet_model = "claude-sonnet-4-5-20250929"  # Claude Sonnet 4.5
        
        # Data storage
        self.df = None
        self.config = None
        self.grouped_comments = {}
        self.candidate_themes = {}
        self.insights = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.limit_groups = None  # Limit number of groups to process (int) or filter by pattern (str) (for testing)
    
    def _fix_json_text(self, text: str) -> str:
        """Try to fix common JSON formatting issues with more aggressive fixes"""
        import re
        import json
        
        original_text = text
        
        # Step 1: Remove markdown code blocks if still present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        # Step 2: Remove trailing commas before closing brackets/braces (multiple passes)
        for _ in range(5):  # More passes for nested cases
            text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Step 3: Try to extract JSON object/array if text contains other content
        first_brace = text.find('{')
        first_bracket = text.find('[')
        
        if first_brace >= 0 and (first_bracket < 0 or first_brace < first_bracket):
            start = first_brace
            # Find matching closing brace, handling strings properly
            depth = 0
            in_string = False
            escape_next = False
            for i in range(start, len(text)):
                if escape_next:
                    escape_next = False
                    continue
                if text[i] == '\\':
                    escape_next = True
                    continue
                if text[i] == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            text = text[start:i+1]
                            break
        elif first_bracket >= 0:
            start = first_bracket
            # Find matching closing bracket
            depth = 0
            in_string = False
            escape_next = False
            for i in range(start, len(text)):
                if escape_next:
                    escape_next = False
                    continue
                if text[i] == '\\':
                    escape_next = True
                    continue
                if text[i] == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if text[i] == '[':
                        depth += 1
                    elif text[i] == ']':
                        depth -= 1
                        if depth == 0:
                            text = text[start:i+1]
                            break
        
        # Step 4: Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)
        
        # Step 5: Try to fix common JSON issues
        try:
            json.loads(text)
            return text.strip()
        except json.JSONDecodeError as e:
            # More aggressive fixes
            error_pos = getattr(e, 'pos', None)
            
            # Fix unescaped newlines in strings (replace actual newlines with \n)
            # This is a common issue when comments contain newlines
            lines = text.split('\n')
            fixed_lines = []
            in_string = False
            escape_next = False
            
            for line in lines:
                fixed_line = ""
                i = 0
                while i < len(line):
                    char = line[i]
                    
                    if escape_next:
                        fixed_line += char
                        escape_next = False
                        i += 1
                        continue
                    
                    if char == '\\':
                        fixed_line += char
                        escape_next = True
                        i += 1
                        continue
                    
                    if char == '"':
                        in_string = not in_string
                        fixed_line += char
                        i += 1
                        continue
                    
                    # If we're in a string and hit a newline (shouldn't happen in valid JSON)
                    # This means the string wasn't properly closed or escaped
                    if in_string and char == '\n':
                        # Replace with \n escape sequence
                        fixed_line += '\\n'
                        i += 1
                        continue
                    
                    fixed_line += char
                    i += 1
                
                fixed_lines.append(fixed_line)
            
            text = '\n'.join(fixed_lines)
            
            # Step 6: Try to fix unterminated strings - more aggressive approach
            # If error mentions "Unterminated string", try to close it
            error_msg = str(e)
            if 'Unterminated string' in error_msg or 'Expecting' in error_msg:
                if error_pos and error_pos < len(text):
                    # Strategy 1: Find the unterminated string and close it
                    # Look backwards from error_pos to find the opening quote
                    start_pos = error_pos
                    quote_count = 0
                    while start_pos > 0:
                        if text[start_pos] == '"':
                            # Check if it's escaped
                            if start_pos == 0 or text[start_pos-1] != '\\':
                                quote_count += 1
                                if quote_count == 1:  # Found the opening quote
                                    break
                        start_pos -= 1
                    
                    if start_pos >= 0 and text[start_pos] == '"':
                        # Found opening quote, now find where to close
                        # Look forward for natural closing points
                        insert_pos = None
                        
                        # Strategy A: Look for next comma, brace, or bracket (likely end of value)
                        for i in range(error_pos, min(error_pos + 300, len(text))):
                            if i >= len(text):
                                break
                            char = text[i]
                            # If we hit a comma or closing brace/bracket, likely end of value
                            if char in [',', '}', ']']:
                                # Check if previous char is not escaped quote
                                if i > 0 and text[i-1] != '"':
                                    insert_pos = i
                                    break
                            # If we hit a newline followed by spaces and then quote/brace/comma
                            elif char == '\n':
                                # Look ahead to see if we're at a structure boundary
                                ahead = text[i+1:min(i+50, len(text))].strip()
                                if ahead.startswith('"') or ahead.startswith(',') or ahead.startswith('}') or ahead.startswith(']'):
                                    insert_pos = i
                                    break
                        
                        # Strategy B: If no natural end found, close at error_pos + reasonable distance
                        if insert_pos is None:
                            # Look for end of line or reasonable stopping point
                            for i in range(error_pos, min(error_pos + 200, len(text))):
                                if i >= len(text):
                                    insert_pos = len(text)
                                    break
                                if text[i] == '\n':
                                    # Check if next line starts a new JSON structure
                                    next_line = text[i+1:min(i+30, len(text))].strip()
                                    if next_line.startswith('"') or next_line.startswith('}') or next_line.startswith(']'):
                                        insert_pos = i
                                        break
                            
                            if insert_pos is None:
                                # Last resort: close at error_pos + 100 chars
                                insert_pos = min(error_pos + 100, len(text))
                        
                        # Insert closing quote
                        if insert_pos is not None:
                            text = text[:insert_pos] + '"' + text[insert_pos:]
                    
                    # Strategy 2: If still failing, try to extract valid JSON up to error
                    try:
                        json.loads(text)
                        return text.strip()
                    except json.JSONDecodeError as e2:
                        # Try to extract partial valid JSON
                        if error_pos and error_pos < len(text):
                            # Try to find the last complete JSON structure before error
                            # Look for last complete object/array
                            last_brace = text.rfind('}', 0, error_pos)
                            last_bracket = text.rfind(']', 0, error_pos)
                            
                            if last_brace > 0 or last_bracket > 0:
                                # Try to extract up to the last complete structure
                                extract_pos = max(last_brace, last_bracket)
                                if extract_pos > 0:
                                    # Find matching opening
                                    if text[extract_pos] == '}':
                                        # Find matching {
                                        depth = 1
                                        for i in range(extract_pos - 1, -1, -1):
                                            if text[i] == '}':
                                                depth += 1
                                            elif text[i] == '{':
                                                depth -= 1
                                                if depth == 0:
                                                    try:
                                                        partial_json = text[i:extract_pos+1]
                                                        json.loads(partial_json)
                                                        # If partial JSON is valid, try to reconstruct
                                                        # For now, just try the fixed version
                                                        pass
                                                    except:
                                                        pass
                                                    break
                            
                            # Final attempt: try the fixed text
                            try:
                                json.loads(text)
                                return text.strip()
                            except:
                                pass
            
            # Final attempt: return the text and let caller handle
            return text.strip()
    
    def load_config(self):
        """Load theme configuration from YAML"""
        print(f"📋 Loading theme configuration from {self.config_file}...")
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        print(f"   ✅ Configuration loaded")
    
    def load_data(self):
        """Load processed comments from CSV"""
        print(f"\n📂 Loading processed comments from {self.processed_csv_file}...")
        self.df = pd.read_csv(self.processed_csv_file, lineterminator="\n")
        
        # Filter out duplicates if _is_duplicate column exists
        if '_is_duplicate' in self.df.columns:
            before_count = len(self.df)
            self.df = self.df[~self.df['_is_duplicate']].reset_index(drop=True)
            after_count = len(self.df)
            print(f"   ✅ Loaded {after_count} comments (removed {before_count - after_count} duplicates)")
        else:
            print(f"   ✅ Loaded {len(self.df)} comments")
        
        # Find text column
        text_cols = ["comment_text", "_text_original", "text", "content", "comment", "body"]
        self.text_col = None
        for col in text_cols:
            if col in self.df.columns:
                self.text_col = col
                break
        
        if not self.text_col:
            raise ValueError("Could not find text column in CSV file")
        
        print(f"   📝 Using text column: {self.text_col}")
    
    def aggregate_comments(self):
        """Aggregate comments by module (category) and sentiment only (no submodule)"""
        print(f"\n📊 Aggregating comments by category and sentiment (no submodule)...")
        
        # Fill NaN values with defaults
        if '_sentiment' in self.df.columns:
            self.df['_sentiment'] = self.df['_sentiment'].fillna('none')
        if '_sentiment_name' in self.df.columns:
            self.df['_sentiment_name'] = self.df['_sentiment_name'].fillna('N/A')
        
        # Group by module and sentiment only (no submodule)
        grouping_cols = ['_module', '_module_name', '_sentiment', '_sentiment_name']
        
        # Check which columns exist
        available_cols = [col for col in grouping_cols if col in self.df.columns]
        
        if not available_cols:
            raise ValueError("Required classification columns not found in CSV file")
        
        print(f"   Grouping by: {', '.join(available_cols)}")
        
        # Group the data
        grouped = self.df.groupby(available_cols, dropna=False)
        
        self.grouped_comments = {}
        
        for group_key, group_df in grouped:
            if isinstance(group_key, tuple):
                # Extract values, handling NaN
                values = []
                for val in group_key:
                    if pd.isna(val):
                        values.append('N/A')
                    else:
                        values.append(str(val))
                
                # Create a key from the group (module + sentiment only)
                if len(values) >= 4:
                    module_id, module_name, sentiment, sentiment_name = values[:4]
                elif len(values) >= 2:
                    module_id, module_name = values[:2]
                    sentiment = values[2] if len(values) > 2 else 'none'
                    sentiment_name = values[3] if len(values) > 3 else 'N/A'
                else:
                    continue
            else:
                continue
            
            # Skip if module_name is missing
            if module_name == 'N/A' or not module_name:
                continue
            
            # Create a unique key for this group (module + sentiment only)
            group_key_str = f"{module_name}|{sentiment_name}"
            
            # Get comments for this group (filter out empty comments)
            comments = [str(c) for c in group_df[self.text_col].tolist() if pd.notna(c) and str(c).strip()]
            
            if not comments:
                continue
            
            # Store group information
            self.grouped_comments[group_key_str] = {
                'module_id': module_id,
                'module_name': module_name,
                'sentiment': sentiment,
                'sentiment_name': sentiment_name,
                'comments': comments,
                'comment_count': len(comments),
                'like_count': int(group_df['like_count'].sum()) if 'like_count' in group_df.columns else 0,
                'avg_likes': float(group_df['like_count'].mean()) if 'like_count' in group_df.columns else 0.0
            }
        
        print(f"   ✅ Aggregated into {len(self.grouped_comments)} groups")
        
        # Print summary
        print(f"\n   All groups by comment count:")
        sorted_groups = sorted(self.grouped_comments.items(), 
                             key=lambda x: x[1]['comment_count'], 
                             reverse=True)
        for group_key, group_data in sorted_groups:
            print(f"      {group_key}: {group_data['comment_count']} comments")
    
    def map_candidate_themes_haiku(self, comments: List[str], module_name: str, 
                                   sentiment_name: str, 
                                   max_retries: int = 5) -> Dict:
        """
        Use Claude Haiku 4.5 to map comments to candidate themes
        
        Args:
            comments: List of comments for this group
            module_name: Module name (category)
            sentiment_name: Sentiment name
            max_retries: Maximum number of retries for API calls
        
        Returns:
            Dictionary with candidate themes
        """
        # Limit number of comments to process (to avoid token limits)
        # Take a sample if too many comments
        # Increased from 100 to 150 to allow more key_examples
        max_comments_per_batch = 150
        if len(comments) > max_comments_per_batch:
            # Sample comments, prioritizing longer ones
            comments_sorted = sorted(comments, key=len, reverse=True)
            comments_to_process = comments_sorted[:max_comments_per_batch]
            comments_sample_info = f" (sampled {max_comments_per_batch} from {len(comments)})"
        else:
            comments_to_process = comments
            comments_sample_info = ""
        
        # Prepare comments text - 清洗评论以避免JSON解析问题
        cleaned_comments = []
        for c in comments_to_process:
            if not c or not isinstance(c, str):
                continue
            # 基本清洗：移除控制字符，限制长度
            import re
            cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', str(c))
            if len(cleaned) > 500:
                last_space = cleaned.rfind(' ', 0, 500)
                if last_space > 400:
                    cleaned = cleaned[:last_space] + "..."
                else:
                    cleaned = cleaned[:500] + "..."
            cleaned_comments.append(cleaned.strip())
        
        comments_text = "\n\n".join([f"{i+1}. {comment}" for i, comment in enumerate(cleaned_comments)])
        
        for attempt in range(max_retries):
            try:
                prompt = f"""分析以下关于"{module_name}"类别、情感为"{sentiment_name}"的评论。

你的任务是识别这些评论中出现的主要主题（话题、关注点或模式）。

评论列表：
{comments_text}

要求：
1. 识别3-10个不同的候选主题，代表评论中的主要话题、关注点或模式
2. 对于每个主题，提供：
   - 主题名称（简洁、描述性）
   - 描述（该主题是关于什么的）
   - 关键示例（5-10条代表性的评论原文，保持原样。如果该主题相关评论很多，可以包含更多，最多15条）
   - 频率指标（该主题出现的频率："非常常见"、"常见"、"偶尔"、"罕见"）
   - 评论数量（提到该主题的评论数量）

3. 关注可操作的洞察 - 能够为产品决策提供信息的主题
4. 将相似的评论归类到同一主题下
5. 要具体和具体 - 避免模糊的主题
6. 关键示例必须是评论的原文，不要改写或总结

输出格式（仅JSON）：
{{
  "candidate_themes": [
    {{
      "theme_name": "主题名称",
      "description": "详细描述",
      "key_examples": ["评论原文1", "评论原文2", "评论原文3", "评论原文4", "评论原文5", "...更多评论"],
      "frequency": "非常常见|常见|偶尔|罕见",
      "comment_count": 提到该主题的评论数量
    }}
  ],
  "summary": "评论的整体摘要（2-3句话）"
}}

重要要求：
1. 仅返回有效的JSON，不要markdown代码块，不要解释文字
2. 确保所有字符串中的引号都正确转义（使用\\"）
3. 确保所有JSON语法正确（没有尾随逗号，括号匹配等）
4. 如果评论中包含引号、换行符等特殊字符，必须正确转义
5. 确保JSON可以立即被解析，无需任何修改"""

                response = self.anthropic_client.messages.create(
                    model=self.haiku_model,
                    max_tokens=8192,  # Large limit to allow full responses
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                
                result_text = response.content[0].text.strip()
                
                # Remove markdown code blocks if present
                if result_text.startswith("```"):
                    result_text = result_text.split("```")[1]
                    if result_text.startswith("json"):
                        result_text = result_text[4:]
                result_text = result_text.strip()
                
                # Try to fix common JSON issues
                result_text = self._fix_json_text(result_text)
                
                # Parse JSON
                try:
                    result = json.loads(result_text)
                    # Add metadata
                    result['total_comments'] = len(comments)
                    result['comments_analyzed'] = len(comments_to_process)
                    result['sample_info'] = comments_sample_info
                    # Ensure all themes have key_examples with actual comment text
                    for theme in result.get('candidate_themes', []):
                        if 'key_examples' not in theme or not theme['key_examples']:
                            theme['key_examples'] = []
                    return result
                except json.JSONDecodeError as e:
                    error_msg = str(e)
                    error_pos = getattr(e, 'pos', None)
                    print(f"   ⚠️ JSON解析错误 (尝试 {attempt + 1}/{max_retries}): {error_msg[:150]}")
                    if error_pos:
                        print(f"      错误位置: {error_pos}, 上下文: {result_text[max(0, error_pos-50):error_pos+50]}")
                    
                    if attempt < max_retries - 1:
                        # Try more aggressive JSON fixing
                        print(f"   🔄 尝试更激进的JSON修复...")
                        result_text = self._fix_json_text(result_text)
                        
                        # Try parsing again with fixed text
                        try:
                            result = json.loads(result_text)
                            # If successful, continue with the fixed result
                            print(f"   ✅ JSON修复成功！")
                            # Ensure all insights have supporting_comments
                            for insight in result.get('key_insights', []):
                                if 'supporting_comments' not in insight:
                                    insight['supporting_comments'] = []
                            for theme in result.get('priority_themes', []):
                                if 'supporting_comments' not in theme:
                                    theme['supporting_comments'] = []
                            if 'sentiment_analysis' in result and 'supporting_comments' not in result['sentiment_analysis']:
                                result['sentiment_analysis']['supporting_comments'] = []
                            return result
                        except json.JSONDecodeError:
                            # Still failing, retry with new API call
                            print(f"   🔄 JSON修复失败，重试API调用 (等待 {2 ** attempt} 秒)...")
                            # Save problematic response for debugging
                            if attempt == 1:
                                debug_file = f"debug_json_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                try:
                                    with open(debug_file, 'w', encoding='utf-8') as f:
                                        f.write(f"Error: {error_msg}\n")
                                        f.write(f"Error position: {error_pos}\n")
                                        f.write(f"Response length: {len(result_text)}\n")
                                        f.write(f"Context around error: {result_text[max(0, error_pos-100):error_pos+100]}\n")
                                        f.write(f"\nFull response:\n{result_text}\n")
                                    print(f"   💾 已保存调试信息到: {debug_file}")
                                except:
                                    pass
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                    else:
                        # Final attempt failed
                        print(f"   ⚠️ 所有重试失败，使用fallback结果")
                        # Save the problematic response
                        debug_file = f"debug_json_final_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        try:
                            with open(debug_file, 'w', encoding='utf-8') as f:
                                f.write(f"Final Error: {error_msg}\n")
                                f.write(f"Error position: {error_pos}\n")
                                f.write(f"Response length: {len(result_text)}\n")
                                f.write(f"Context: {result_text[max(0, error_pos-200):error_pos+200] if error_pos else 'N/A'}\n")
                                f.write(f"\nFull response:\n{result_text}\n")
                            print(f"   💾 已保存最终错误信息到: {debug_file}")
                        except:
                            pass
                        raise
                    
            except Exception as e:
                print(f"   ⚠️ 错误 (尝试 {attempt + 1}/{max_retries}): {str(e)[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        
        # Fallback if all retries failed
        return {
            "candidate_themes": [],
            "summary": "错误: 重试后仍无法生成主题",
            "total_comments": len(comments),
            "comments_analyzed": len(comments_to_process),
            "sample_info": comments_sample_info
        }
    
    def _clean_comment_for_json(self, comment: str, max_length: int = 500) -> str:
        """
        清洗评论内容，确保可以安全地放入JSON字符串中
        
        Args:
            comment: 原始评论
            max_length: 最大长度，超过则截断
        
        Returns:
            清洗后的评论
        """
        if not comment or not isinstance(comment, str):
            return ""
        
        # 1. 移除控制字符（保留换行符，稍后处理）
        import re
        comment = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', comment)
        
        # 2. 转义反斜杠（必须在转义引号之前）
        comment = comment.replace('\\', '\\\\')
        
        # 3. 转义引号
        comment = comment.replace('"', '\\"')
        
        # 4. 将换行符转换为 \n（JSON格式）
        comment = comment.replace('\n', '\\n').replace('\r', '')
        
        # 5. 移除制表符或转换为空格
        comment = comment.replace('\t', ' ')
        
        # 6. 限制长度（在句子或单词边界截断）
        if len(comment) > max_length:
            # 尝试在句号、感叹号、问号后截断
            truncate_pos = max_length
            for punct in ['. ', '! ', '? ', '。', '！', '？']:
                last_pos = comment.rfind(punct, 0, max_length)
                if last_pos > 0:
                    truncate_pos = last_pos + len(punct)
                    break
            
            # 如果没找到标点，尝试在空格处截断
            if truncate_pos == max_length:
                last_space = comment.rfind(' ', 0, max_length)
                if last_space > max_length * 0.8:  # 至少保留80%的内容
                    truncate_pos = last_space
            
            comment = comment[:truncate_pos] + "..."
        
        # 7. 移除首尾空白
        comment = comment.strip()
        
        return comment
    
    def generate_insights_sonnet(self, candidate_themes: Dict, module_name: str, 
                                 sentiment_name: str,
                                 comment_count: int, comments: List[str] = None, max_retries: int = 5) -> Dict:
        """
        Use Claude Sonnet 4.5 to generate insights from candidate themes
        
        Args:
            candidate_themes: Dictionary with candidate themes from Haiku
            module_name: Module name (category)
            sentiment_name: Sentiment name
            comment_count: Total number of comments
            comments: Original comments list for reference
            max_retries: Maximum number of retries for API calls
        
        Returns:
            Dictionary with insights
        """
        themes_text = json.dumps(candidate_themes, indent=2, ensure_ascii=False)
        
        # Include sample comments for reference (to ensure supporting_comments are accurate)
        # 清洗评论内容，避免JSON解析错误
        comments_ref = ""
        if comments:
            # Include up to 200 comments for reference (increased to allow more supporting_comments)
            sample_comments = comments[:200] if len(comments) > 200 else comments
            # 清洗每条评论，确保可以安全地放入JSON（但不在prompt中转义，让Claude自己处理）
            # 只做基本清理：移除控制字符、限制长度
            cleaned_comments = []
            for c in sample_comments:
                if not c or not isinstance(c, str):
                    continue
                # 移除控制字符
                import re
                cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', str(c))
                # 限制长度（在单词边界截断）
                if len(cleaned) > 400:
                    last_space = cleaned.rfind(' ', 0, 400)
                    if last_space > 300:
                        cleaned = cleaned[:last_space] + "..."
                    else:
                        cleaned = cleaned[:400] + "..."
                cleaned_comments.append(cleaned.strip())
            
            comments_ref = f"\n\n原始评论样本（供参考，确保支撑评论的准确性，共{len(cleaned_comments)}条）：\n" + "\n".join([f"{i+1}. {c}" for i, c in enumerate(cleaned_comments)])
        
        for attempt in range(max_retries):
            try:
                prompt = f"""基于从"{module_name}"类别、情感为"{sentiment_name}"的{comment_count}条评论中识别出的候选主题，生成深入的洞察分析。

候选主题：
{themes_text}
{comments_ref}

要求：
1. 将候选主题综合成具体的洞察（可以包括表面观察和深层分析）
2. 识别最重要的主题及其具体表现
3. 突出任何紧急问题或机会
4. 考虑业务背景：这是游戏产品的用户反馈
5. 洞察必须基于实际的评论内容，要具体、详细
6. 每个洞察必须包含支撑该洞察的具体评论原文作为证据
7. 关键洞察数量：根据候选主题数量和评论内容的重要性，生成5-10个关键洞察。如果候选主题数量较多（>8个），可以生成更多洞察；如果候选主题较少（<5个），可以适当合并或聚焦最重要的话题
8. 优先主题数量：从关键洞察中识别3-5个最重要的优先主题

输出格式（仅JSON）：
{{
  "key_insights": [
    {{
      "insight": "具体的洞察陈述（要详细、具体，基于实际评论内容）",
      "importance": "高|中|低",
      "supporting_comments": ["支撑该洞察的评论原文1", "支撑该洞察的评论原文2", "支撑该洞察的评论原文3", "支撑该洞察的评论原文4", "支撑该洞察的评论原文5", "..."]
    }}
  ],
  "priority_themes": [
    {{
      "theme_name": "主题名称",
      "why_important": "为什么这个主题重要（具体原因）",
      "supporting_comments": ["支撑该主题的评论原文1", "支撑该主题的评论原文2", "支撑该主题的评论原文3", "支撑该主题的评论原文4", "支撑该主题的评论原文5", "..."]
    }}
  ],
  "sentiment_analysis": {{
    "overall_sentiment": "正面|负面|中性|混合",
    "sentiment_explanation": "情感模式的解释（要具体）",
    "emotional_tone": "情感基调的描述（要具体）",
    "supporting_comments": ["支撑情感分析的评论原文1", "支撑情感分析的评论原文2", "支撑情感分析的评论原文3", "支撑情感分析的评论原文4", "支撑情感分析的评论原文5", "..."]
  }},
  "summary": "执行摘要（3-5句话，要具体）"
}}

重要提示：
- 洞察必须具体、详细，基于实际评论中的具体内容和表述
- 每个洞察和主题都必须包含支撑评论的原文作为证据。supporting_comments中的评论必须是从原始评论样本中准确引用的原文，不能改写或总结
- supporting_comments数量要求：
  * 关键洞察（key_insights）：每个洞察应该包含5-10条支撑评论原文，如果该洞察有很多相关评论，可以包含更多（最多15条）
  * 优先主题（priority_themes）：每个主题应该包含5-8条支撑评论原文，如果该主题有很多相关评论，可以包含更多（最多12条）
  * 情感分析（sentiment_analysis）：应该包含5-8条支撑评论原文，展示不同情感的评论示例
- 选择支撑评论时，应该选择最能代表该洞察/主题的评论，优先选择点赞数较高、表述清晰的评论
- 如果某个洞察或主题的相关评论很多，不要只选择3-4条，应该包含更多支撑评论以充分证明该洞察
- 不要提供recommendation或implication字段
- 洞察可以包括表面观察和深层分析，只要是基于实际评论内容的真实洞察即可
- 仅返回有效的JSON，不要markdown代码块，不要解释
- 重要：所有字符串值中的引号、换行符、反斜杠等特殊字符必须正确转义（使用\\\"表示引号，\\n表示换行，\\\\表示反斜杠）
- 重要：确保所有JSON字符串都正确关闭，不要截断任何字符串值。如果某个字段的值很长，可以适当缩短，但必须确保JSON结构完整有效
- 重要：supporting_comments中的评论原文应该从上面提供的"原始评论样本"中准确引用，引用时保持原文的转义格式（引号已转义为\\\"，换行已转义为\\n）"""

                response = self.anthropic_client.messages.create(
                    model=self.sonnet_model,
                    max_tokens=8192,  # Large limit to allow full responses
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                
                result_text = response.content[0].text.strip()
                
                # Remove markdown code blocks if present
                if result_text.startswith("```"):
                    result_text = result_text.split("```")[1]
                    if result_text.startswith("json"):
                        result_text = result_text[4:]
                result_text = result_text.strip()
                
                # Try to fix common JSON issues
                result_text = self._fix_json_text(result_text)
                
                # Parse JSON
                try:
                    result = json.loads(result_text)
                    # Ensure all insights have supporting_comments
                    for insight in result.get('key_insights', []):
                        if 'supporting_comments' not in insight:
                            insight['supporting_comments'] = []
                    for theme in result.get('priority_themes', []):
                        if 'supporting_comments' not in theme:
                            theme['supporting_comments'] = []
                    if 'sentiment_analysis' in result and 'supporting_comments' not in result['sentiment_analysis']:
                        result['sentiment_analysis']['supporting_comments'] = []
                    return result
                except json.JSONDecodeError as e:
                    error_msg = str(e)
                    error_pos = getattr(e, 'pos', None)
                    print(f"   ⚠️ JSON解析错误 (尝试 {attempt + 1}/{max_retries}): {error_msg[:150]}")
                    if error_pos:
                        print(f"      错误位置: {error_pos}, 上下文: {result_text[max(0, error_pos-50):error_pos+50]}")
                    
                    if attempt < max_retries - 1:
                        # Try more aggressive JSON fixing
                        print(f"   🔄 尝试更激进的JSON修复...")
                        result_text = self._fix_json_text(result_text)
                        
                        # Try parsing again with fixed text
                        try:
                            result = json.loads(result_text)
                            # If successful, continue with the fixed result
                            print(f"   ✅ JSON修复成功！")
                            # Ensure all insights have supporting_comments
                            for insight in result.get('key_insights', []):
                                if 'supporting_comments' not in insight:
                                    insight['supporting_comments'] = []
                            for theme in result.get('priority_themes', []):
                                if 'supporting_comments' not in theme:
                                    theme['supporting_comments'] = []
                            if 'sentiment_analysis' in result and 'supporting_comments' not in result['sentiment_analysis']:
                                result['sentiment_analysis']['supporting_comments'] = []
                            return result
                        except json.JSONDecodeError:
                            # Still failing, retry with new API call
                            print(f"   🔄 JSON修复失败，重试API调用 (等待 {2 ** attempt} 秒)...")
                            # Save problematic response for debugging
                            if attempt == 1:
                                debug_file = f"debug_json_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                try:
                                    with open(debug_file, 'w', encoding='utf-8') as f:
                                        f.write(f"Error: {error_msg}\n")
                                        f.write(f"Error position: {error_pos}\n")
                                        f.write(f"Response length: {len(result_text)}\n")
                                        f.write(f"Context: {result_text[max(0, error_pos-100):error_pos+100] if error_pos else 'N/A'}\n")
                                        f.write(f"\nFull response:\n{result_text}\n")
                                    print(f"   💾 已保存调试信息到: {debug_file}")
                                except:
                                    pass
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                    else:
                        # Final attempt failed
                        print(f"   ⚠️ 所有重试失败，使用fallback结果")
                        # Save the problematic response
                        debug_file = f"debug_json_final_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        try:
                            with open(debug_file, 'w', encoding='utf-8') as f:
                                f.write(f"Final Error: {error_msg}\n")
                                f.write(f"Error position: {error_pos}\n")
                                f.write(f"Response length: {len(result_text)}\n")
                                f.write(f"Context: {result_text[max(0, error_pos-200):error_pos+200] if error_pos else 'N/A'}\n")
                                f.write(f"\nFull response:\n{result_text}\n")
                            print(f"   💾 已保存最终错误信息到: {debug_file}")
                        except:
                            pass
                        raise
                    
            except Exception as e:
                print(f"   ⚠️ 错误 (尝试 {attempt + 1}/{max_retries}): {str(e)[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        
        # Fallback if all retries failed
        return {
            "key_insights": [],
            "priority_themes": [],
            "sentiment_analysis": {
                "overall_sentiment": sentiment_name,
                "sentiment_explanation": "错误: 无法生成洞察",
                "emotional_tone": "未知",
                "supporting_comments": []
            },
            "summary": "错误: 重试后仍无法生成洞察"
        }
    
    def process_all_groups(self):
        """Process all comment groups: map themes with Haiku, then generate insights with Sonnet"""
        print(f"\n🤖 使用 Claude AI 处理 {len(self.grouped_comments)} 个组...")
        print(f"   步骤 1: 使用 Claude Haiku 4.5 生成候选主题")
        print(f"   步骤 2: 使用 Claude Sonnet 4.5 生成洞察")
        
        # Sort groups by comment count (process larger groups first)
        sorted_groups = sorted(self.grouped_comments.items(), 
                             key=lambda x: x[1]['comment_count'], 
                             reverse=True)
        
        # Apply limit if specified (for testing)
        # Also support filtering by group key pattern
        if self.limit_groups:
            if isinstance(self.limit_groups, str):
                # Filter by group key pattern (e.g., "Monetization|Positive")
                sorted_groups = [(k, v) for k, v in sorted_groups if self.limit_groups in k]
                print(f"   ⚠️  筛选组: 包含 '{self.limit_groups}'")
            elif isinstance(self.limit_groups, int) and self.limit_groups > 0:
                sorted_groups = sorted_groups[:self.limit_groups]
                print(f"   ⚠️  限制为 {self.limit_groups} 个组进行测试")
        
        self.candidate_themes = {}
        self.insights = {}
        
        # Process each group
        for idx, (group_key, group_data) in enumerate(tqdm(sorted_groups, desc="   处理组")):
            module_name = group_data['module_name']
            sentiment_name = group_data['sentiment_name']
            comments = group_data['comments']
            comment_count = group_data['comment_count']
            
            print(f"\n   [{idx+1}/{len(sorted_groups)}] 处理中: {group_key} ({comment_count} 条评论)")
            
            # Step 1: Map candidate themes with Haiku
            try:
                candidate_themes = self.map_candidate_themes_haiku(
                    comments, module_name, sentiment_name
                )
                self.candidate_themes[group_key] = candidate_themes
                print(f"      ✅ 生成了 {len(candidate_themes.get('candidate_themes', []))} 个候选主题")
            except Exception as e:
                print(f"      ❌ 生成主题时出错: {str(e)[:100]}")
                self.candidate_themes[group_key] = {
                    "candidate_themes": [],
                    "summary": f"Error: {str(e)}",
                    "total_comments": comment_count
                }
                continue
            
            # Step 2: Generate insights with Sonnet
            try:
                insights = self.generate_insights_sonnet(
                    candidate_themes, module_name, sentiment_name, comment_count, comments
                )
                self.insights[group_key] = insights
                print(f"      ✅ 生成了洞察")
            except Exception as e:
                print(f"      ❌ Error generating insights: {str(e)[:100]}")
                self.insights[group_key] = {
                    "key_insights": [],
                    "priority_themes": [],
                    "sentiment_analysis": {
                        "overall_sentiment": sentiment_name,
                        "sentiment_explanation": f"错误: {str(e)}",
                        "emotional_tone": "未知",
                        "supporting_comments": []
                    },
                    "summary": f"错误: {str(e)}"
                }
                continue
            
            # Rate limiting: small delay between groups
            time.sleep(1)
        
        print(f"\n   ✅ 已处理 {len(self.candidate_themes)} 个组")
    
    def save_results(self):
        """Save results to files"""
        print(f"\n💾 Saving results...")
        
        # Create output directory if needed
        output_dir = self.output_dir or ""
        if output_dir and not output_dir.endswith('/'):
            output_dir += '/'
        
        # Save candidate themes
        themes_file = f"{output_dir}candidate_themes_{self.timestamp}.json"
        with open(themes_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': self.timestamp,
                    'total_groups': len(self.candidate_themes),
                    'source_file': self.processed_csv_file
                },
                'groups': self.candidate_themes
            }, f, indent=2, ensure_ascii=False)
        print(f"   • {themes_file}")
        
        # Save insights
        insights_file = f"{output_dir}insights_{self.timestamp}.json"
        with open(insights_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': self.timestamp,
                    'total_groups': len(self.insights),
                    'source_file': self.processed_csv_file
                },
                'groups': self.insights
            }, f, indent=2, ensure_ascii=False)
        print(f"   • {insights_file}")
        
        # Save combined results (themes + insights + group data)
        combined_file = f"{output_dir}theme_insights_combined_{self.timestamp}.json"
        combined_data = {
            'metadata': {
                'timestamp': self.timestamp,
                'total_groups': len(self.grouped_comments),
                'source_file': self.processed_csv_file
            },
            'groups': {}
        }
        
        for group_key in self.grouped_comments.keys():
            combined_data['groups'][group_key] = {
                'group_info': self.grouped_comments[group_key],
                'candidate_themes': self.candidate_themes.get(group_key, {}),
                'insights': self.insights.get(group_key, {})
            }
        
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        print(f"   • {combined_file}")
        
        # Save summary CSV
        summary_file = f"{output_dir}insights_summary_{self.timestamp}.csv"
        summary_rows = []
        
        for group_key, group_data in self.grouped_comments.items():
            insights_data = self.insights.get(group_key, {})
            themes_data = self.candidate_themes.get(group_key, {})
            
            row = {
                'module_name': group_data['module_name'],
                'sentiment_name': group_data['sentiment_name'],
                'comment_count': group_data['comment_count'],
                'like_count': group_data['like_count'],
                'avg_likes': group_data['avg_likes'],
                'num_candidate_themes': len(themes_data.get('candidate_themes', [])),
                'num_key_insights': len(insights_data.get('key_insights', [])),
                'num_priority_themes': len(insights_data.get('priority_themes', [])),
                'overall_sentiment': insights_data.get('sentiment_analysis', {}).get('overall_sentiment', ''),
                'summary': insights_data.get('summary', '')[:500] if insights_data.get('summary') else ''  # Limit summary length for CSV
            }
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values('comment_count', ascending=False)
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"   • {summary_file}")
        
        return themes_file, insights_file, combined_file, summary_file
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("🎬 Theme Insight Generator")
        print("=" * 50)
        print(f"⚙️ Configuration:")
        print(f"   Model (Map): Claude Haiku 4.5")
        print(f"   Model (Reduce): Claude Sonnet 4.5")
        print(f"   Source file: {self.processed_csv_file}")
        
        # Load data
        self.load_config()
        self.load_data()
        
        # Aggregate comments
        self.aggregate_comments()
        
        # Process all groups
        self.process_all_groups()
        
        # Save results
        output_files = self.save_results()
        
        # Print summary
        print("\n" + "=" * 50)
        print("🎉 ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"\n📊 Summary:")
        print(f"   Total groups processed: {len(self.grouped_comments)}")
        print(f"   Groups with themes: {len(self.candidate_themes)}")
        print(f"   Groups with insights: {len(self.insights)}")
        
        print(f"\n📁 Output files:")
        for output_file in output_files:
            print(f"   • {output_file}")


def main():
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Generate theme insights from processed comments using Claude AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process comments with API key from environment
  python theme_insight_generator.py comments_processed_20251110_105015.csv
  
  # Process with explicit API key
  python theme_insight_generator.py comments_processed_20251110_105015.csv --api-key YOUR_API_KEY
  
  # Specify output directory
  python theme_insight_generator.py comments_processed_20251110_105015.csv --output-dir results/
        """
    )
    
    ap.add_argument("processed_csv", help="Path to processed comments CSV file")
    ap.add_argument("--config", type=str, default="themes_config_new.yaml", 
                   help="Theme configuration YAML file")
    ap.add_argument("--api-key", type=str, default=None, 
                   help="Anthropic API key (or use ANTHROPIC_API_KEY env var)")
    ap.add_argument("--output-dir", type=str, default=None, 
                   help="Output directory for results")
    ap.add_argument("--limit-groups", type=str, default=None,
                   help="Limit number of groups to process (integer) or filter by group pattern (string, e.g., 'Monetization|Positive')")
    
    args = ap.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.processed_csv):
        print(f"❌ Error: File not found: {args.processed_csv}")
        return
    
    # Initialize and run
    generator = ThemeInsightGenerator(
        processed_csv_file=args.processed_csv,
        config_file=args.config,
        api_key=args.api_key,
        output_dir=args.output_dir
    )
    
    # Add limit_groups attribute if specified
    if args.limit_groups:
        # Try to parse as integer first
        try:
            generator.limit_groups = int(args.limit_groups)
        except ValueError:
            # If not an integer, treat as string pattern
            generator.limit_groups = args.limit_groups
    else:
        generator.limit_groups = None
    
    generator.run_full_analysis()


if __name__ == "__main__":
    main()

