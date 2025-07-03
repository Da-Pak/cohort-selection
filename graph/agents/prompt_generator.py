import os
import logging
from typing import Dict, Any, Optional
import torch
from cachetools import cached
from ...utils import timeit, safe_json_loads, PROMPT_CACHE, get_config, get_loaded_model

from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 2000
TEMPERATURE = 0.1
TOP_P = 0.95
DO_SAMPLE = True

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
5. Step-by-step reasoning for inclusion criteria (case-positive identification)
- Explicitly list phrases or report content that strongly indicate the medical record meets the diagnostic criteria for case selection.
- Provide a logical explanation for each identified inclusion criterion.
6. Step-by-step reasoning for exclusion criteria (case-negative identification)
- Explicitly list phrases or report content that clearly indicate the medical record does not meet the criteria and should be excluded from selection.
- Provide a logical explanation for each identified exclusion criterion.
7. Synthesize into a clear, actionable context for clinical case selection
- Integrate definitions, inclusion/exclusion criteria, and instructions in a clear and concise manner specifically tailored to facilitate accurate and efficient clinical case filtering and selection by an LLM.
The resulting context will directly guide the LLM in accurately identifying and selecting relevant cases from each medical report.
"""
# ----------------------------------------------------------------------------------------------------------

@cached(cache=PROMPT_CACHE)
@timeit
def generate_context(question: str, temperature: float) -> str:
    """
    Generate medical context based on user question using local Gemma3 27b model.
    
    Args:
        question (str): Natural language question from user
        temperature (float): Temperature setting for generation
        
    Returns:
        str: Generated medical context
    """
    try:
        logger.info(f"Context generation started with local model: {question}")
        
        # 로컬 모델 로드
        model, tokenizer = get_loaded_model()
        
        # 프롬프트 생성
        prompt = CONTEXT_PROMPT_TEMPLATE.format(question=question)
        
        # 토크나이저 적용
        input_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_bos=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        
        inputs = tokenizer(input_prompt, return_tensors="pt").to(DEVICE)
        logger.info(f"Context generation input length: {len(inputs.input_ids[0])}")
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temperature,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
            )
        
        # 출력 디코딩
        context = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        logger.info(f"Context generation completed: {len(context)} characters")
        logger.info(f"Generated context length: {len(outputs[0][inputs.input_ids.shape[1]:])}")
        
        return context.strip()
    
    except Exception as e:
        logger.error(f"Error during context generation with local model: {e}")
        # Return default context on error
        return f"""
        User question: {question}
        
        Analyze the reports according to the following steps:
        1. Look for keywords or sentences related to the question in the text.
        2. If related content is clearly mentioned, classify as "INCLUDE".
        3. If related content is clearly negated or absent, classify as "EXCLUDE".
        4. If judgment is difficult, classify as "UNCERTAIN".
        """