import os
import logging
from typing import Dict, Any, Optional
import openai
from cachetools import cached
from ...utils import timeit, safe_json_loads, PROMPT_CACHE, get_config

from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# OpenAI API 키 설정

# ----------------------------------------------------------------------------------------------------------
CONTEXT_PROMPT_TEMPLATE = """
You are a medical expert and natural language processing specialist. Your task is to create a structured context using a Chain-of-Thought (CoT) method to support a system designed to select relevant clinical cases by automatically filtering unstructured medical records, such as radiology reports, based on specific medical criteria.
Given a user's medical question: "{question}", your structured thought process must focus explicitly on case identification and selection through the following detailed steps:
1. Analyze the clinical relevance of the user's question for case selection
- Clearly identify the specific clinical condition, symptom, or scenario implied by the user's question.
- Clarify why identifying this condition is important for case selection.
2. Define relevant medical terms for accurate selection
- Provide concise, accurate definitions of all medical conditions, symptoms, or clinical findings explicitly or implicitly referenced by the question.
- Highlight how each defined term relates to clinical case identification.
3. Detail the clinical characteristics essential for case selection
- Clearly describe typical clinical signs, diagnostic features, or radiologic findings that are critical for recognizing the targeted cases.
4. Anticipate language used in medical records that indicates case relevance
- Specify typical words, phrases, or sentence patterns clinicians commonly use in radiology or medical reports to describe the clinical condition relevant for case selection.
5. Step-by-step reasoning for present criteria (case-positive identification)
- Explicitly list phrases or report content that strongly indicate the medical record meets the diagnostic criteria for case selection.
- Provide a logical explanation for each identified absent criterion.
6. Step-by-step reasoning for exclusion criteria (case-negative identification)
- Explicitly list phrases or report content that clearly indicate the medical record does not meet the criteria and should be excluded from selection.
- Provide a logical explanation for each identified exclusion criterion.
7. Synthesize into a clear, actionable context for clinical case selection
- Integrate definitions, present/absent criteria, and instructions in a clear and concise manner specifically tailored to facilitate accurate and efficient clinical case filtering and selection by an LLM.
The resulting context will directly guide the LLM in accurately identifying and selecting relevant cases from each medical report.
"""
# ----------------------------------------------------------------------------------------------------------
client = openai.OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
@cached(cache=PROMPT_CACHE)
@timeit
def generate_context(question: str, temperature: float) -> str:
    """
    Generate medical context based on user question.
    
    Args:
        question (str): Natural language question from user
        
    Returns:
        str: Generated medical context
    """
    try:
        logger.info(f"Context generation started: {question}")
        config = get_config()
        
        # Call GPT API
        response = client.chat.completions.create(
            model=config.llm_config.openai_model_name,
            messages=[
                {"role": "system", "content": "You are a medical expert and natural language processing specialist."},
                {"role": "user", "content": CONTEXT_PROMPT_TEMPLATE.format(question=question)}
            ],
            temperature=temperature,
            max_tokens=config.llm_config.max_tokens,
        )
        
        # Extract context from response
        context = response.choices[0].message.content.strip()
        print(f"Context : {context}")
        logger.info(f"Context : {context}")
        logger.info(f"Context generation completed: {len(context)} characters")
        return context
    
    except Exception as e:
        logger.error(f"Error during context generation: {e}")
        # Return default context on error
        return f"""
        User question: {question}
        
        Analyze the reports according to the following steps:
        1. Look for keywords or sentences related to the question in the text.
        2. If related content is clearly mentioned, classify as "Present".
        3. If related content is clearly negated or absent, classify as "Absent".
        4. If judgment is difficult, classify as "Uncertain".
        """