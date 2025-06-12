import os
import logging
import openai
import torch
from typing import Dict, Any, Optional, List, Union, Tuple, Literal
from cachetools import cached
from ...utils import timeit, safe_json_loads, VERIFIER_CACHE, get_config, get_loaded_model, is_model_loaded
import time
# ----------------------------------------------------------------------------------------------------------
VERIFIER_PROMPT_TEMPLATE = """ 
You are a medical expert acting as a "verifier" in a multi-stage system designed to analyze medical imaging reports.

In a previous step, a "LLM" acting as an "case identifier" analyzed a clinical report and attempted to answer a specific clinical question. 

The LLM extracted a sentence from the report and assigned a label (Present / Absent / Uncertain) to indicate whether the condition in question was present, absent, or uncertain in that sentence.

The label was assigned according to the following guideline:
If the sentence clearly supports the presence of the condition, then the case identifier return: **Present**  
If the sentence clearly supports the absence of the condition, then the case identifier return: **Absent**  
If the sentence is ambiguous, hypothetical, or uses uncertain language (e.g., "possible,"r/o"), then the case identifier return: **Uncertain**

You will receive the following input:
- A clinical report: {text}
- A clinical question: {question}
- Context for interpretation: {context}
- A LLM (case identifier)'s decision, including:
  - Extracted sentence: "{sentence}"
  - Label assigned: "{opinion}" (Present / Absent / Uncertain)

Your task is to independently verify the correctness of the LLM's label based on the extracted sentence and the context.
Then, compare it with the label assigned by the case identifier (LLM).

If the LLM's label matches your judgment, return `true`. If not, return `false`.

Your response must strictly follow the JSON format below:
{{
  "is_correct": True or False 
}}
"""
# ----------------------------------------------------------------------------------------------------------
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

# @cached(cache=VERIFIER_CACHE, key=lambda text, question, context, inference_result, **kwargs: 
#           (text, question, context, tuple(sorted(inference_result.items()))))
@timeit
def verify_with_openai(text: str, question: str, context: str, inference_result: Dict[str, Any], temperature: float) -> bool:
    """
    Verify the LLM's judgment using OpenAI.
    
    Args:
        text (str): Original report text
        question (str): User question
        inference_result (Dict[str, Any]): LLM's inference result (sentence, opinion)
        
    Returns:
        bool: Verification result (True: valid, False: invalid)
    """
    try:
        # Extract inference results
        sentence = inference_result.get("sentence", "")
        opinion = inference_result.get("opinion", "Uncertain")

        logger.info(f"Verification started: Judgment - {opinion}")
        config = get_config()
        
        # API 요청에 타임아웃 설정
        response = client.chat.completions.create(
            model=config.llm_config.openai_model_name,
            messages=[
                {"role": "system", "content": "You are a medical expert and natural language processing specialist."},
                {"role": "user", "content": VERIFIER_PROMPT_TEMPLATE.format(
                    text=text,
                    question=question,
                    context=context,
                    sentence=sentence,
                    opinion=opinion
                )}
            ],
            temperature=temperature,  # Use low temperature for consistency
            max_tokens=MAX_NEW_TOKENS,  # 10에서 500으로 증가
            timeout=30.0,    # 30초 타임아웃 설정
        )
        
        # Extract verification result from response
        verification_result = response.choices[0].message.content.strip().lower()
        try:
            # Find JSON part
            json_start = verification_result.find("{")
            json_end = verification_result.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                json_str = verification_result[json_start:json_end]
                result = safe_json_loads(json_str)
                # Validate result
                if "is_correct" in result:
                    # Normalize opinion value
                    is_valid = result["is_correct"].lower() == "present"
                    logger.info(f"Verification completed: opinion - {opinion}, Verification result - {is_valid}")
                    return is_valid
               
        except Exception as e:
            logger.error(f"JSON processing error: {e}")
            return {
                "sentence": "",
                "opinion": "Uncertain"
            }
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        # Return default value (True) on error
        return True

# @cached(cache=VERIFIER_CACHE, key=lambda text, question, context, inference_result, **kwargs: 
#           (text, question, context, tuple(sorted(inference_result.items()))))
@timeit
def verify_with_local_model(text: str, question: str, context: str, inference_result: Dict[str, Any], temperature: float) -> bool:
    """
    Verify the LLM's judgment using local model.
    
    Args:
        text (str): Original report text
        question (str): User question
        inference_result (Dict[str, Any]): LLM's inference result (sentence, opinion)
        
    Returns:
        bool: Verification result (True: valid, False: invalid)
    """
    # 수정된 부분: utils.py에서 제공하는 함수 사용
    model, tokenizer = get_loaded_model()
    
    try:
        # Extract inference results
        sentence = inference_result.get("sentence", "")
        opinion = inference_result.get("opinion", "Uncertain")
        
        # Generate prompt
        prompt = VERIFIER_PROMPT_TEMPLATE.format(
            text=text,
            question=question,
            context=context,
            sentence=sentence,
            opinion=opinion
        )
        # logger.info(f"Verification input context length : {len(context)}")
        # context_input = tokenizer(context, return_tensors="pt").to(DEVICE)
        # logger.info(f"Inference input context tokenized length : {len(context_input.input_ids[0])}")
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
        logger.info(f"Verification input length : {len(inputs.input_ids[0])}")
        # print("VERIFIER_PROMPT_TEMPLATE",prompt)
        # Tokenize and infer
        # inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,  # 전역 변수인 MAX_NEW_TOKENS를 사용
                temperature=temperature,
                do_sample=DO_SAMPLE,
            )
        
        # Decode output
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # print("verification_result",output_text)
        logger.info(f"Verification generated text length : {len(outputs[0][inputs.input_ids.shape[1]:])}")
        try:
            # Find JSON part
            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                json_str = output_text[json_start:json_end]
                result = safe_json_loads(json_str)
                # print('verify result2',result)
                # Validate result
                if "is_correct" in result:
                    # Normalize opinion value
                    is_valid = result["is_correct"].lower().strip()
                    logger.info(f"Verification completed: opinion - {opinion}, Verification result - {is_valid}")
                    return is_valid
               
        except Exception as e:
            logger.error(f"JSON processing error: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        # Return default value (True) on error
        return None

# @cached(cache=VERIFIER_CACHE, key=lambda text, question, context, inference_result, **kwargs: 
#           (text, question, context, tuple(sorted(inference_result.items()))))
@timeit
def verify_opinion(text: str, question: str, context: str, inference_result: Dict[str, Any], llm_type: Literal["local", "openai"], temperature: float) -> bool:
    """
    Verify the LLM's judgment with selected LLM type.
    
    Args:
        text (str): Original report text
        question (str): User question
        inference_result (Dict[str, Any]): LLM's inference result (sentence, opinion)
        llm_type (str): LLM type to use for verification ("local" or "openai")
        
    Returns:
        bool: Verification result (True: valid, False: invalid)
    """
    try:
        # 기본 타임아웃 설정
        timeout_seconds = 30.0
        
        logger.warning(f"Verification setting- model : {llm_type} temperature : {temperature}")
        
        # 시간 제한을 두고 검증 실행
        start_time = time.time()
        
        if llm_type == "local":
            result = verify_with_local_model(text, question, context, inference_result, temperature)
        else:
            result = verify_with_openai(text, question, context, inference_result, temperature)
        # print('verify result', result)
        # 타임아웃 체크
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            logger.warning(f"Verification took too long: {elapsed_time:.2f} seconds")
        
        return result
    except Exception as e:
        logger.error(f"Error in verify_opinion: {str(e)}")
        # 오류 발생 시 기본값 반환 (검증은 기본적으로 성공으로 간주)
        return True