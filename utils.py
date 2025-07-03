import logging
import uuid
import json
import os
from typing import Dict, Any, Optional, Literal
from cachetools import TTLCache, cached
import pandas as pd
import time
from functools import wraps
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log'),
    ]
)
logger = logging.getLogger(__name__)

# In-memory data cache
DATA_CACHE = {}
# Task status cache
TASK_STATUS = {}
# API call cache (TTL: 1 hour)
API_CACHE = TTLCache(maxsize=100, ttl=3600)
# Prompt cache (TTL: 1 hour)
PROMPT_CACHE = TTLCache(maxsize=100, ttl=3600)
# Verifier cache (TTL: 1 hour)
VERIFIER_CACHE = TTLCache(maxsize=100, ttl=3600)

# Configuration models
class LLMConfig(BaseModel):
    """Configuration for LLM models"""
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model_name: str = Field(default="gpt-4o")
    llm_type: Literal["local", "openai"] = Field(default="local", description="Type of LLM to use (local or openai)")
    local_model_path: str = Field(default="google/gemma-3-27b-it")
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=2000)

class PipelineConfig(BaseModel):
    """Configuration for the filtering pipeline"""
    batch_size: int = Field(default=10, description="Number of texts to process in a single batch")
    max_workers: int = Field(default=4, description="Maximum number of parallel workers")
    retry_count: int = Field(default=1, description="Number of retries for failed operations")
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    
    def __str__(self):
        return (
            f"PipelineConfig(batch_size={self.batch_size}, "
            f"max_workers={self.max_workers}, "
            f"retry_count={self.retry_count})"
        )

# Default pipeline configuration
DEFAULT_PIPELINE_CONFIG = PipelineConfig()

# 전역 변수 선언
model = None
tokenizer = None

def is_model_loaded():
    """모델이 이미 로드되었는지 확인"""
    global model, tokenizer
    return model is not None and tokenizer is not None

def get_loaded_model():
    """로드된 모델과 토크나이저 반환, 로드되지 않았으면 로드"""
    global model, tokenizer
    if not is_model_loaded():
        load_model()
    return model, tokenizer

def get_config() -> PipelineConfig:
    """Get the current pipeline configuration"""
    return DEFAULT_PIPELINE_CONFIG

def update_config(config: PipelineConfig) -> None:
    """Update the pipeline configuration"""
    global DEFAULT_PIPELINE_CONFIG
    DEFAULT_PIPELINE_CONFIG = config
    logger.info(f"Configuration updated: {config}")

def generate_id() -> str:
    """Generate a unique ID"""
    return str(uuid.uuid4())


def save_dataframe(df: pd.DataFrame, metadata: Dict[str, Any] = None) -> str:
    """
    데이터프레임을 캐시에 저장하고 ID를 반환합니다.
    
    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        metadata (Dict[str, Any], optional): 함께 저장할 메타데이터
        
    Returns:
        str: 생성된 데이터 ID
    """
    data_id = generate_id()
    
    # 메타데이터가 없으면 빈 딕셔너리 생성
    if metadata is None:
        metadata = {}
    
    # 데이터와 메타데이터 저장
    DATA_CACHE[data_id] = {
        "data": df,
        "metadata": metadata
    }
    
    logger.info(f"데이터프레임 저장: {data_id}, 행 수: {len(df)}, 메타데이터: {metadata}")
    return data_id


def get_dataframe(data_id: str) -> Optional[pd.DataFrame]:
    """
    ID로 데이터프레임을 가져옵니다.
    
    Args:
        data_id (str): 데이터 ID
        
    Returns:
        Optional[pd.DataFrame]: 데이터프레임
    """
    if data_id not in DATA_CACHE:
        logger.warning(f"데이터프레임을 찾을 수 없음: {data_id}")
        return None
    
    # 새로운 형식이면 데이터 필드에서 데이터프레임 반환
    if isinstance(DATA_CACHE[data_id], dict) and "data" in DATA_CACHE[data_id]:
        return DATA_CACHE[data_id]["data"]
    
    # 이전 형식 지원 (호환성 유지)
    return DATA_CACHE[data_id]


def get_metadata(data_id: str) -> Optional[Dict[str, Any]]:
    """
    ID로 메타데이터를 가져옵니다.
    
    Args:
        data_id (str): 데이터 ID
        
    Returns:
        Optional[Dict[str, Any]]: 메타데이터
    """
    if data_id not in DATA_CACHE:
        logger.warning(f"데이터프레임을 찾을 수 없음: {data_id}")
        return None
    
    # 새로운 형식이면 메타데이터 필드에서 메타데이터 반환
    if isinstance(DATA_CACHE[data_id], dict) and "metadata" in DATA_CACHE[data_id]:
        return DATA_CACHE[data_id]["metadata"]
    
    # 이전 형식 지원 (호환성 유지)
    return {}


def create_task() -> str:
    """Create a new task and return its ID"""
    task_id = generate_id()
    TASK_STATUS[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Task pending",
        "created_at": time.time()
    }
    return task_id


def update_task_status(task_id: str, status: str, progress: float, message: str = None) -> None:
    """Update task status"""
    if task_id in TASK_STATUS:
        TASK_STATUS[task_id]["status"] = status
        TASK_STATUS[task_id]["progress"] = progress
        if message:
            TASK_STATUS[task_id]["message"] = message
        TASK_STATUS[task_id]["updated_at"] = time.time()
        logger.info(f"Task status updated: {task_id}, status: {status}, progress: {progress}")


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status by ID"""
    if task_id not in TASK_STATUS:
        logger.warning(f"Task not found: {task_id}")
        return None
    print("TASK_STATUS",TASK_STATUS[task_id])
    return TASK_STATUS[task_id]

def timeit(func):
    """Function execution time measurement decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} execution time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def safe_json_loads(text: str) -> Dict[str, Any]:
    """Safe JSON parsing"""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}\nText: {text}")
        return {}    

def load_model():
    """Load LLama model and tokenizer"""
    global model, tokenizer
    config = get_config()
    model_path = config.llm_config.local_model_path or "meta-llama/Llama-2-7b-chat-hf"
    
    if not is_model_loaded():
        logger.info(f"Loading local LLM model: {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = Gemma3ForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map='auto')
            
            logger.info(f"모델 로드 완료: {model_path}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    return model, tokenizer
