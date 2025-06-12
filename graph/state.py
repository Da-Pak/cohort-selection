from typing import Dict, List, Any, Optional, TypedDict
from pydantic import BaseModel, Field
import pandas as pd


class FilterState(TypedDict):
    """필터 그래프 상태"""
    # 입력 데이터
    question: str
    data_id: str
    task_id: str
    target_column: str
    temperature: float
    
    # 처리 상태
    status: str
    progress: float
    message: str
    
    # 생성된 데이터
    context: Optional[str]
    dataframe: Optional[pd.DataFrame]
    filtered_dataframe: Optional[pd.DataFrame]
    results: List[Dict[str, Any]]
    reasoning: List[str]
    
    # 오류 처리
    error: Optional[str]
    retries: int


def get_initial_state(question: str, data_id: str, task_id: str, target_column: str, temperature: float) -> FilterState:
    """필터 그래프의 초기 상태를 생성합니다."""
    return FilterState(
        # 입력 데이터
        question=question,
        data_id=data_id,
        task_id=task_id,
        target_column=target_column,
        temperature = temperature,
        # 처리 상태
        status="pending",
        progress=0.0,
        message="작업 대기 중",
        
        # 생성된 데이터
        context=None,
        dataframe=None,
        filtered_dataframe=None,
        results=[],
        reasoning=[],
        
        # 오류 처리
        error=None,
        retries=0
    ) 