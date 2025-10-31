import os
import json
import time
import pandas as pd
from typing import Dict, Any, List
from openai import OpenAI

# Optional .env support (matches llm.py behavior)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Missing API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY in your environment or .env"
    )

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

REASON_FIELDS: List[str] = [
    "reason_for_theme_and_logic",
    "reason_for_creativity",
    "reason_for_layout_and_composition",
    "reason_for_space_and_perspective",
    "reason_for_sense_of_order",
    "reason_for_light_and_shadow",
    "reason_for_color",
    "reason_for_details_and_texture",
    "reason_for_overall",
    "reason_for_mood",
]


def create_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_model(client: OpenAI, prompt: str) -> Dict[str, Any]:
    """Call the LLM with a provided prompt that returns a JSON string."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON substring if the model wrapped it with text
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def enrich_row_with_reasons(row: pd.Series, prompt_template: str, client: OpenAI) -> Dict[str, Any]:
    """Use the comment from the row to fetch reasons and return a mapping for columns."""
    comment_text = str(row.get("comment", ""))
    if not comment_text.strip():
        return {field: None for field in REASON_FIELDS}

    # Build format context safely
    format_context = {}
    for col in row.index:
        val = row.get(col, "")
        if isinstance(val, float) and pd.isna(val):
            val = ""
        # Use simple column names without special characters
        safe_col = str(col).replace('"', '').replace("'", "").strip()
        format_context[safe_col] = val
    
    # Ensure comment is available
    format_context["comment"] = comment_text
    
    try:
        prompt = prompt_template.format(**format_context)
    except KeyError as e:
        print(f"Warning: Key error in formatting: {e}. Using raw template.")
        prompt = prompt_template.replace("{comment}", comment_text)
    
    result = call_model(client, prompt)

    updates: Dict[str, Any] = {}
    for field in REASON_FIELDS:
        val = result.get(field)
        if isinstance(val, str) and val.strip():
            updates[field] = val.strip()
        else:
            updates[field] = None
    return updates


def enrich_csv(
    input_csv_path: str,
    output_csv_path: str,
    prompt_template: str,
    batch_size: int = 1,
    rate_limit_sleep_s: float = 0.0,
) -> None:
    """Read CSV, call model for each row (optionally batched), and write enriched CSV."""
    if "{comment}" not in prompt_template:
        raise ValueError("prompt_template must contain '{comment}' placeholder")

    df = pd.read_csv(input_csv_path)
    # Ensure all reason columns exist so we can assign by name later
    for col in REASON_FIELDS:
        if col not in df.columns:
            df[col] = None

    client = create_client()
    total_rows = len(df)
    
    print(f"Starting to process {total_rows} rows...")

    for idx in range(total_rows):
        row = df.iloc[idx]
        # Skip rows with empty/NaN comment to save tokens
        comment_val = row.get("comment", "")
        if (isinstance(comment_val, float) and pd.isna(comment_val)) or (isinstance(comment_val, str) and not comment_val.strip()):
            continue

        print(f"Processing row {idx + 1}/{total_rows}...")
        updates = enrich_row_with_reasons(row, prompt_template, client)
        
        for k, v in updates.items():
            df.at[row.name, k] = v
            
        if rate_limit_sleep_s > 0:
            time.sleep(rate_limit_sleep_s)
            
        # Save progress every 10 rows
        if (idx + 1) % 10 == 0:
            df.to_csv(output_csv_path, index=False, encoding="utf-8")
            print(f"Progress saved at row {idx + 1}")

    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Completed processing {total_rows} rows.")


if __name__ == "__main__":
    # 修复后的提示词模板 - 使用三引号避免转义问题
    PROMPT_TEMPLATE = '''
Comment: {comment}

Please extract short sentences and phrases from the comments that reflect the reasons for the score of a specific aesthetic attribute.

Return the result in the following JSON format:
{{
    "reason_for_theme_and_logic": "",
    "reason_for_creativity": "",
    "reason_for_layout_and_composition": "",
    "reason_for_space_and_perspective": "",
    "reason_for_sense_of_order": "",
    "reason_for_light_and_shadow": "",
    "reason_for_color": "",
    "reason_for_details_and_texture": "",
    "reason_for_overall": "",
    "reason_for_mood": ""
}}

Note: 
- If no reasons for a certain dimension can be found in the comment, fill in an empty string
- Do not fabricate information
- Only use phrases directly from the comment

Here are 10 aesthetic attributes and their interpretations:
- Theme and Logic: The central idea aligns with the artistic expression, ensuring consistency and appropriateness in composition, layout, and color.
- Creativity: Innovative qualities that break conventions, including satire, self-deprecation, and allegorical warnings.
- Layout and Composition: The visual structure and organization of an image, reflecting the underlying logic and essence of its form.
- Space and Perspective: Layered spatial arrangements and perspective techniques create three-dimensionality and spatial effects.
- Sense of Order: Visual unity and consistency in morphological, spatial, orientational, and dynamic elements.
- Light and Shadow: Enhance visual rhythm and realism, decorate space, suggest themes, and segment the image.
- Color: Evokes emotional atmospheres with a harmonious palette, using contrasts in temperature, brightness, and purity.
- Details and Texture: Vivid details and delicate textures enhance realism, imbuing life into the image.
- The Overall: Emphasizes coherence and a clear theme, combining form and spirit in the presentation.
- Mood: Creates a poetic space blending scenes, reality, and illusion, emphasizing tranquility, emptiness, and spirituality.

example:
the comment is "Each side of the picture is good, making it a great landscape painting. The brushstrokes are skilled, and the visual effect is realistic"
the answer should be:
{{
    "reason_for_theme_and_logic": "",
    "reason_for_creativity": "",
    "reason_for_layout_and_composition": "Each side of the picture is good",
    "reason_for_space_and_perspective": "",
    "reason_for_sense_of_order": "",
    "reason_for_light_and_shadow": "",
    "reason_for_color": "",
    "reason_for_details_and_texture": "The brushstrokes are skilled",
    "reason_for_overall": "making it a great landscape painting",
    "reason_for_mood": ""
}}
'''.strip()

    input_path = "APDD.csv"
    output_path = "APDD_enriched.csv"

    enrich_csv(
        input_csv_path=input_path,
        output_csv_path=output_path,
        prompt_template=PROMPT_TEMPLATE,
        batch_size=1,
        rate_limit_sleep_s=0.5,  # 添加0.5秒延迟避免速率限制
    )
    print(f"Enriched CSV written to: {output_path}")