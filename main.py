from mcp.server import FastMCP
import google.generativeai as genai
import json
import re
from typing import Dict, Any

# Initialize FastMCP server
app = FastMCP('entity-comparison')

def normalize_text(text: str) -> str:
    """
    标准化文本：转换为小写、去除标点符号、归一化空白字符。
    """
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compare_values(val1: Any, val2: Any) -> tuple[bool, bool]:
    """
    对比两个 value：
    - exact_equal：直接比较原始数据
    - semantic_equal：比较标准化后的文本
    """
    if not isinstance(val1, list):
        val1 = [val1]
    if not isinstance(val2, list):
        val2 = [val2]
    
    exact_equal = (val1 == val2)
    
    norm1 = [normalize_text(item) for item in val1]
    norm2 = [normalize_text(item) for item in val2]
    
    semantic_equal = (sorted(norm1) == sorted(norm2))
    
    return exact_equal, semantic_equal

@app.tool()
async def compare_entities(json1: Dict[str, Any], json2: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    比较两个实体的JSON数据，使用Gemini进行语义相似度分析
    
    Args:
        json1: 第一个实体的JSON数据
        json2: 第二个实体的JSON数据 
        api_key: Google API Key

    Returns:
        包含比较结果的字典
    """
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    # Set safety settings
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    # Initialize models
    comparison_model = genai.GenerativeModel("gemini-2.0-flash-exp")
    final_model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp")

    # Compare individual fields
    comparison_results = {}
    system_instruction = '请判断这个两个要素在语义层面是否一致，直接返回true或者false'

    for key in json1:
        if key not in json2:
            comparison_results[key] = {"error": "在第二份数据中未找到该 key"}
            continue
            
        exact_equal, semantic_equal = compare_values(json1[key], json2[key])
        
        # Get LLM comparison
        llm_result = comparison_model.generate_content(
            system_instruction + ":" + str(json1[key]) + "\n" + str(json2[key])
        )

        comparison_results[key] = {
            "exact_equal": exact_equal,
            "semantic_equal": semantic_equal,
            "LLM_equal": llm_result.text,
            "value1": json1[key],
            "value2": json2[key],
        }

    # Get final analysis
    final_result = final_model.generate_content(
        "综合这些信息，你认为可以判断两个数据来自同一主体吗？" + 
        json.dumps(comparison_results, ensure_ascii=False, indent=4)
    )

    return {
        "field_comparisons": comparison_results,
        "final_analysis": final_result.text
    }

if __name__ == "__main__":
    app.run()
