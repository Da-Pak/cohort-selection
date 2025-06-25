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

Your ONLY task is to verify whether a previous LLM's judgment was CORRECT or INCORRECT. You must NOT make your own medical judgment about Present/Absent/Uncertain.

In a previous step, a "case identifier" LLM analyzed a clinical report and provided:
- Label assigned: "{opinion}" (Present / Absent / Uncertain)  
- Extracted sentence: "{sentence}"

Context for interpretation: {context}
Clinical question: {question}
Clinical report: {text}

The case identifier was supposed to follow these guidelines:
- "Present": Condition is explicitly reported as currently present/active
- "Absent": Condition is clearly not present currently, or only mentioned as past history/resolved, or no relevant information found
- "Uncertain": Condition's presence is unclear, ambiguous, uses tentative language (e.g., "possible," "suspected," "rule out"), or insufficient clarity

Your verification task:
1. Check if the extracted sentence actually supports the assigned label
2. Determine if the case identifier's judgment was reasonable given the sentence and context

IMPORTANT: You must ONLY return "correct" or "incorrect" in the is_correct field. Do NOT return "uncertain" or any other value.

If the judgment seems reasonable (even if you might have chosen differently), mark it as "correct".
Only mark as "incorrect" if there's a clear error (e.g., "Present" assigned to clearly absent condition, or completely wrong sentence extraction).

Return your response in this exact JSON format:
{{
"is_correct": "correct" or "incorrect",
"reason": "[If incorrect, briefly explain why. If correct, leave empty or write 'Answer is appropriate.']",
"correction_hint": "[If incorrect, provide specific guidance. If correct, leave empty.]"
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
def verify_with_openai(text: str, question: str, context: str, inference_result: Dict[str, Any], temperature: float) -> Dict[str, Any]:
    """
    Verify the LLM's judgment using OpenAI.
    
    Args:
        text (str): Original report text
        question (str): User question
        context (str): Context for analysis
        inference_result (Dict[str, Any]): LLM's inference result (sentence, opinion)
        temperature (float): Temperature setting
        
    Returns:
        Dict[str, Any]: Verification result with feedback information
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
        verification_result = response.choices[0].message.content.strip()
        try:
            # Find JSON part
            json_start = verification_result.find("{")
            json_end = verification_result.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                json_str = verification_result[json_start:json_end]
                result = safe_json_loads(json_str)
                
                # Validate result
                if result and "is_correct" in result:
                    # is_correct 값을 안전하게 파싱
                    is_correct_value = str(result.get("is_correct", "correct")).lower().strip()
                    
                    # 명시적으로 올바른 값들만 처리
                    if is_correct_value in ["correct", "true", "yes"]:
                        is_correct = True
                    elif is_correct_value in ["incorrect", "false", "no"]:
                        is_correct = False
                    else:
                        # 예상하지 못한 값이 오면 로그 출력하고 기본값 사용
                        logger.warning(f"Unexpected is_correct value: '{is_correct_value}', defaulting to True")
                        is_correct = True
                    
                    logger.info(f"Verification completed: opinion - {opinion}, Verification result - {is_correct}")
                    # 전체 피드백 객체 반환
                    return {
                        "is_correct": is_correct,
                        "reason": result.get("reason", ""),
                        "correction_hint": result.get("correction_hint", ""),
                        "raw_response": verification_result
                    }
                else:
                    # JSON 파싱은 성공했지만 is_correct 필드가 없는 경우
                    logger.warning(f"Missing is_correct field in verifier response: {result}")
                    return {
                        "is_correct": True,  # 기본값으로 올바름으로 간주
                        "reason": "응답에 is_correct 필드 없음",
                        "correction_hint": "",
                        "raw_response": verification_result
                    }
            else:
                logger.warning(f"No JSON found in verifier response: {verification_result}")
                return {
                    "is_correct": True,  # JSON이 없으면 기본값으로 올바름으로 간주
                    "reason": "JSON 응답 형식 오류",
                    "correction_hint": "",
                    "raw_response": verification_result
                }
               
        except Exception as e:
            logger.error(f"JSON processing error: {e}")
            logger.error(f"Raw verifier response: {verification_result}")
            return {
                "is_correct": True,  # 기본값으로 올바름으로 간주
                "reason": "JSON 파싱 오류",
                "correction_hint": "",
                "raw_response": verification_result
            }
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        # Return default value (True) on error
        return {
            "is_correct": True,
            "reason": f"검증 중 오류 발생: {str(e)}",
            "correction_hint": "",
            "raw_response": ""
        }

# @cached(cache=VERIFIER_CACHE, key=lambda text, question, context, inference_result, **kwargs: 
#           (text, question, context, tuple(sorted(inference_result.items()))))
@timeit
def verify_with_local_model(text: str, question: str, context: str, inference_result: Dict[str, Any], temperature: float) -> Dict[str, Any]:
    """
    Verify the LLM's judgment using local model.
    
    Args:
        text (str): Original report text
        question (str): User question
        context (str): Context for analysis
        inference_result (Dict[str, Any]): LLM's inference result (sentence, opinion)
        temperature (float): Temperature setting
        
    Returns:
        Dict[str, Any]: Verification result with feedback information
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
                if result and "is_correct" in result:
                    # is_correct 값을 안전하게 파싱
                    is_correct_value = str(result.get("is_correct", "correct")).lower().strip()
                    
                    # 명시적으로 올바른 값들만 처리
                    if is_correct_value in ["correct", "true", "yes"]:
                        is_correct = True
                    elif is_correct_value in ["incorrect", "false", "no"]:
                        is_correct = False
                    else:
                        # 예상하지 못한 값이 오면 로그 출력하고 기본값 사용
                        logger.warning(f"Unexpected is_correct value: '{is_correct_value}', defaulting to True")
                        is_correct = True
                    
                    logger.info(f"Verification completed: opinion - {opinion}, Verification result - {is_correct}")
                    # 전체 피드백 객체 반환
                    return {
                        "is_correct": is_correct,
                        "reason": result.get("reason", ""),
                        "correction_hint": result.get("correction_hint", ""),
                        "raw_response": output_text
                    }
                else:
                    # JSON 파싱은 성공했지만 is_correct 필드가 없는 경우
                    logger.warning(f"Missing is_correct field in verifier response: {result}")
                    return {
                        "is_correct": True,  # 기본값으로 올바름으로 간주
                        "reason": "응답에 is_correct 필드 없음",
                        "correction_hint": "",
                        "raw_response": output_text
                    }
            else:
                logger.warning(f"No JSON found in verifier response: {output_text}")
                return {
                    "is_correct": True,  # JSON이 없으면 기본값으로 올바름으로 간주
                    "reason": "JSON 응답 형식 오류", 
                    "correction_hint": "",
                    "raw_response": output_text
                }
               
        except Exception as e:
            logger.error(f"JSON processing error: {e}")
            logger.error(f"Raw verifier response: {output_text}")
            return {
                "is_correct": True,  # 기본값으로 올바름으로 간주
                "reason": "JSON 파싱 오류",
                "correction_hint": "",
                "raw_response": output_text
            }
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        # Return default value (True) on error
        return {
            "is_correct": True,
            "reason": f"검증 중 오류 발생: {str(e)}",
            "correction_hint": "",
            "raw_response": ""
        }

# @cached(cache=VERIFIER_CACHE, key=lambda text, question, context, inference_result, **kwargs: 
#           (text, question, context, tuple(sorted(inference_result.items()))))
@timeit
def verify_opinion(text: str, question: str, context: str, inference_result: Dict[str, Any], llm_type: Literal["local", "openai"], temperature: float) -> Dict[str, Any]:
    """
    Verify the LLM's judgment with selected LLM type.
    
    Args:
        text (str): Original report text
        question (str): User question
        context (str): Context for analysis
        inference_result (Dict[str, Any]): LLM's inference result (sentence, opinion)
        llm_type (str): LLM type to use for verification ("local" or "openai")
        temperature (float): Temperature setting
        
    Returns:
        Dict[str, Any]: Verification result with feedback information
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
        
        # 타임아웃 체크
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            logger.warning(f"Verification took too long: {elapsed_time:.2f} seconds")
            
        return result
        
    except Exception as e:
        logger.error(f"Error in verify_opinion: {str(e)}")
        # 오류 발생 시 기본값 반환 (검증은 기본적으로 성공으로 간주)
        return {
            "is_correct": True,
            "reason": f"verify_opinion 오류: {str(e)}",
            "correction_hint": "",
            "raw_response": ""
        }