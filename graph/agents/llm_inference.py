import os
import logging
import openai
import torch
from typing import Dict, Any, Optional, List, Union, Tuple, Literal
from cachetools import cached
from ...utils import timeit, safe_json_loads, get_config, get_loaded_model, is_model_loaded
import time

# ----------------------------------------------------------------------------------------------------------
ANALYSIS_PROMPT_TEMPLATE = """ <Medical Report Analysis Task>

The following is a medical report text:
{text}

The following is the user question:
{question}

Below is the context for analyzing the report:
{context}

{verifier_feedback_section}

You are given a clinical report and a clinical condition as context. Perform the following two tasks:

1. From the report, extract the single most relevant sentence that directly addresses the **current** presence or absence (including uncertainty) of the condition described in the context. Only consider statements describing the patient's present status; do not consider historical, past, or resolved findings as present unless they are explicitly stated as ongoing or currently active. Consider the overall context of the report to avoid relying on protocol descriptions or general instructions unless they directly refer to the patient's current clinical findings.
    - If there is no sentence that directly addresses the presence or absence of the condition, set the value of "sentence" to "No relevant sentence found."
2. Based on that sentence and the full context of the report, categorize the report into one of the following categories:
    - "Present": The condition is explicitly reported as being present in the current clinical context.
    - "Absent": The condition is clearly reported as not being present currently, or is mentioned only as past history/resolved condition, or there is no relevant information about the condition in the report.
    - "Uncertain": The condition's presence is unclear, ambiguous, or cannot be determined from the available information. Use this when the report contains conflicting information, uses tentative language (e.g., "possible," "rule out," "suspected"), or when the extracted sentence does not provide sufficient clarity for a definitive Present/Absent determination.

IMPORTANT: If verifier feedback is provided above indicating that your previous answer was incorrect, you MUST carefully review the feedback and revise your answer accordingly. Pay special attention to the correction hint and ensure your revised answer directly addresses the identified issue.

Your response must strictly follow the JSON format below:

{{
"sentence": "The sentence that serves as the basis for your judgment, or 'No relevant sentence found'",
"opinion": "Present or Absent or Uncertain"
}}
"""
# ----------------------------------------------------------------------------------------------------------

def format_verifier_feedback(verifier_feedback: Optional[Dict[str, Any]]) -> str:
    """
    verifier 피드백을 프롬프트에 포함할 수 있는 형태로 포매팅합니다.
    
    Args:
        verifier_feedback (Optional[Dict[str, Any]]): verifier의 피드백 객체
        
    Returns:
        str: 포매팅된 피드백 텍스트 (공란이거나 피드백 내용)
    """
    if not verifier_feedback:
        return ""
    
    # verifier가 올바르다고 판단했거나 피드백이 없는 경우
    if verifier_feedback.get("is_correct", True):
        return ""
    
    # verifier가 틀렸다고 판단한 경우 피드백 포함
    reason = verifier_feedback.get("reason", "").strip()
    correction_hint = verifier_feedback.get("correction_hint", "").strip()
    
    feedback_parts = []
    feedback_parts.append("**VERIFIER FEEDBACK - Previous Answer Correction Required:**")
    
    if reason:
        feedback_parts.append(f"**Reason for correction:** {reason}")
    
    if correction_hint:
        feedback_parts.append(f"**Correction hint:** {correction_hint}")
    
    feedback_parts.append("**Please revise your answer based on this feedback. Remember you can choose Present, Absent, or Uncertain.**")
    feedback_parts.append("")  # 빈 줄 추가
    
    return "\n".join(feedback_parts)

logger = logging.getLogger(__name__)
# OpenAI API key setup
client = openai.OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

# Local LLM model and tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 5000
TEMPERATURE = 0.1
TOP_P = 0.95
DO_SAMPLE = True

# Variables for lazy loading
# model = None
# tokenizer = None


@timeit
def infer_with_openai(text: str, context: str, question: str, temperature:float, verifier_feedback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform LLM inference using OpenAI.
    
    Args:
        text (str): Report text to analyze
        context (str): Context for analysis
        question (str): User question
        temperature (float): Temperature setting
        verifier_feedback (Optional[Dict[str, Any]]): Verifier feedback object
        
    Returns:
        Dict[str, Any]: Inference result (sentence, opinion)
    """
    try:
        config = get_config()
        
        # verifier 피드백 포매팅
        verifier_feedback_section = format_verifier_feedback(verifier_feedback)
        
        response = client.chat.completions.create(
            model=config.llm_config.openai_model_name,
            messages=[
                {"role": "system", "content": "You are a medical expert and natural language processing specialist."},
                {"role": "user", "content": ANALYSIS_PROMPT_TEMPLATE.format(
                    text=text,
                    question=question,
                    context=context,
                    verifier_feedback_section=verifier_feedback_section
                )}
            ],
            temperature=temperature,
            max_tokens=150
        )
        
        # Extract result from response
        output_text = response.choices[0].message.content.strip()
        
        # Extract JSON
        try:
            # Find JSON part
            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                json_str = output_text[json_start:json_end]
                result = safe_json_loads(json_str)
                
                # Validate result
                if "sentence" in result and "opinion" in result:
                    # Normalize opinion value
                    opinion_lower = result["opinion"].lower().strip()
                    if opinion_lower in ["present", "포함", "yes", "true"]:
                        result["opinion"] = "Present"
                    elif opinion_lower in ["absent", "제외", "no", "false"]:
                        result["opinion"] = "Absent"
                    elif opinion_lower in ["uncertain", "unclear", "unknown", "ambiguous", "불확실", "모호", "애매"]:
                        result["opinion"] = "Uncertain"
                    else:
                        # 기본값을 Uncertain으로 설정
                        result["opinion"] = "Uncertain"
                    
                    return result
            
            # Return default value if JSON parsing fails
            logger.warning(f"JSON parsing failed, output: {output_text}")
            return {
                "sentence": "",
                "opinion": "Uncertain"
            }
            
        except Exception as e:
            logger.error(f"JSON processing error: {e}")
            return {
                "sentence": "",
                "opinion": "Uncertain"
            }
            
    except Exception as e:
        logger.error(f"Error during OpenAI inference: {e}")
        return {
            "sentence": "",
            "opinion": "Uncertain"
        }

@timeit
def infer_with_local_model(text: str, context: str, question: str, temperature:float, verifier_feedback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform LLM inference using local model.
    
    Args:
        text (str): Report text to analyze
        context (str): Context for analysis
        question (str): User question
        temperature (float): Temperature setting
        verifier_feedback (Optional[Dict[str, Any]]): Verifier feedback object
        
    Returns:
        Dict[str, Any]: Inference result (sentence, opinion)
    """
    # 수정된 부분: utils.py에서 제공하는 함수 사용
    model, tokenizer = get_loaded_model()
    
    try:
        # verifier 피드백 포매팅
        verifier_feedback_section = format_verifier_feedback(verifier_feedback)
        
        # Generate prompt
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            text=text, 
            context=context, 
            question=question, 
            verifier_feedback_section=verifier_feedback_section
        )
        # print("ANALYSIS_PROMPT_TEMPLATE",prompt)
        # Tokenize and infer
        
        input_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_bos=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        # logger.info(f"Inference input context length : {len(context)}")
        # context_input = tokenizer(context, return_tensors="pt").to(DEVICE)
        # logger.info(f"Inference input context tokenized length : {len(context_input.input_ids[0])}")
        
        inputs = tokenizer(input_prompt, return_tensors="pt").to(DEVICE)
        
        # inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temperature,
                do_sample=DO_SAMPLE,
            )
        
        # Decode output
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        logger.info(f"Inference generated text length : {len(outputs[0][inputs.input_ids.shape[1]:])}")
        # Extract JSON (find only the JSON part in the output)
        try:
            # Find JSON part
            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                json_str = output_text[json_start:json_end]
                result = safe_json_loads(json_str)
                
                # Validate result
                if "sentence" in result and "opinion" in result:
                    # Normalize opinion value
                    opinion_lower = result["opinion"].lower().strip()
                    if opinion_lower in ["present", "포함", "yes", "true"]:
                        result["opinion"] = "Present"
                    elif opinion_lower in ["absent", "제외", "no", "false"]:
                        result["opinion"] = "Absent"
                    elif opinion_lower in ["uncertain", "unclear", "unknown", "ambiguous", "불확실", "모호", "애매"]:
                        result["opinion"] = "Uncertain"
                    else:
                        # 기본값을 Uncertain으로 설정
                        result["opinion"] = "Uncertain"
                    
                    logger.info(f"opinion generation completed: Judgment - {result['opinion']}")
                    return result
            
            # Return default value if JSON parsing fails
            logger.warning(f"JSON parsing failed, output: {output_text}")
            return {
                "sentence": "",
                "opinion": "Uncertain"
            }
            
        except Exception as e:
            logger.error(f"JSON processing error: {e}")
            return {
                "sentence": "",
                "opinion": "Uncertain"
            }
            
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return {
            "sentence": "",
            "opinion": "Uncertain"
        }
        
@timeit
def inference_llm(text: str, context: str, question: str, llm_type: Literal["local", "openai"], temperature: float, verifier_feedback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform LLM inference on a single text with selected LLM type.
    
    Args:
        text (str): Report text to analyze
        context (str): Context for analysis
        question (str): User question  
        llm_type (str): LLM type to use ("local" or "openai")
        temperature (float): Temperature setting
        verifier_feedback (Optional[Dict[str, Any]]): Verifier feedback object
        
    Returns:
        Dict[str, Any]: Inference result (sentence, opinion)
    """
    logger.warning(f"inference setting- llm_type : {llm_type} temperature : {temperature}")
    if llm_type == "local":
        return infer_with_local_model(text, context, question, temperature, verifier_feedback)
    else:
        return infer_with_openai(text, context, question, temperature, verifier_feedback)