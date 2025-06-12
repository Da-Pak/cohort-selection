import os
import logging
import openai
import torch
from typing import Dict, Any, Optional, List, Union, Tuple, Literal
from cachetools import cached
from ...utils import timeit, safe_json_loads, get_config, get_loaded_model, is_model_loaded
import time

# ----------------------------------------------------------------------------------------------------------
ANALYSIS_PROMPT_TEMPLATE = """ <Medical Report Analysis Task> The following is a medical report text: 
{text}
The following is the user question: 
{question}
Below is the context for analyzing the report: 
{context}
You are given a clinical report and a clinical condition as context. Perform the following two tasks:

1. From the report, extract the single most relevant sentence that directly addresses the presence, absence, or uncertainty of the condition described in the context.
2. Based on that sentence, categorize the report into one of the following categories:
   - "Present": The condition is clearly reported as being present.
   - "Absent": The condition is clearly reported as not being present.
   - "Uncertain": The condition is mentioned with ambiguous, tentative, or hypothetical language (e.g., "rule out," "possible", etc.).

Your response must strictly follow the JSON format below:

{{ 
"sentence": "The sentence that serves as the basis for your judgment", 
"opinion": "Present or Absent or Uncertain"
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


@timeit
def infer_with_openai(text: str, context: str, question: str, temperature:float) -> Dict[str, Any]:
    """
    Perform LLM inference using OpenAI.
    
    Args:
        text (str): Report text to analyze
        context (str): Context for analysis
        
    Returns:
        Dict[str, Any]: Inference result (sentence, opinion)
    """
    try:
        config = get_config()
        
        response = client.chat.completions.create(
            model=config.llm_config.openai_model_name,
            messages=[
                {"role": "system", "content": "You are a medical expert and natural language processing specialist."},
                {"role": "user", "content": ANALYSIS_PROMPT_TEMPLATE.format(
                    text=text,
                    question=question,
                    context=context
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
                    if result["opinion"].lower() in ["present", "포함", "yes", "true"]:
                        result["opinion"] = "Present"
                    elif result["opinion"].lower() in ["absent", "제외", "no", "false"]:
                        result["opinion"] = "Absent"
                    else:
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
            "opinion": "UNCERTAIN"
        }

@timeit
def infer_with_local_model(text: str, context: str, question: str, temperature:float) -> Dict[str, Any]:
    """
    Perform LLM inference using local model.
    
    Args:
        text (str): Report text to analyze
        context (str): Context for analysis
        
    Returns:
        Dict[str, Any]: Inference result (sentence, opinion)
    """
    # 수정된 부분: utils.py에서 제공하는 함수 사용
    model, tokenizer = get_loaded_model()
    
    try:
        # Generate prompt
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(text=text, context=context, question=question)
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
                    if result["opinion"].lower() in ["present", "포함", "yes", "true"]:
                        result["opinion"] = "Present"
                    elif result["opinion"].lower() in ["absent", "제외", "no", "false"]:
                        result["opinion"] = "Absent"
                    else:
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
            "opinion": "UNCERTAIN"
        }
        
@timeit
def inference_llm(text: str, context: str, question: str, llm_type: Literal["local", "openai"], temperature: float) -> Dict[str, Any]:
    """
    Perform LLM inference on a single text with selected LLM type.
    
    Args:
        text (str): Report text to analyze
        context (str): Context for analysis
        llm_type (str): LLM type to use ("local" or "openai")
        
    Returns:
        Dict[str, Any]: Inference result (sentence, opinion)
    """
    logger.warning(f"inference setting- llm_type : {llm_type} temperature : {temperature}")
    if llm_type == "local":
        return infer_with_local_model(text, context, question, temperature)
    else:
        return infer_with_openai(text, context, question, temperature)