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


class SingleTextState(TypedDict):
    """단일 텍스트 처리용 서브그래프 상태"""
    # 입력 데이터
    text: str
    context: str
    question: str
    temperature: float
    llm_type: str
    
    # 처리 결과
    sentence: Optional[str]
    opinion: Optional[str]
    verified_sentence: Optional[bool]
    
    # 첫 번째 추론 결과 (retry와 상관없이 보존)
    first_sentence: Optional[str]
    first_opinion: Optional[str]
    
    # CoT 분석 결과 저장
    inference_cot: Optional[Dict[str, Any]]  # inference의 CoT 분석 결과
    
    # 루프 제어
    retry_count: int
    max_retries: int
    
    # 상태 관리
    current_step: str  # "inference", "validate_sentence", "completed"
    error: Optional[str]


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


def get_initial_single_text_state(text: str, context: str, question: str, temperature: float, llm_type: str) -> SingleTextState:
    """단일 텍스트 처리용 초기 상태를 생성합니다."""
    return SingleTextState(
        # 입력 데이터
        text=text,
        context=context,
        question=question,
        temperature=temperature,
        llm_type=llm_type,
        
        # 처리 결과
        sentence=None,
        opinion=None,
        verified_sentence=None,
        
        # 첫 번째 추론 결과 (retry와 상관없이 보존)
        first_sentence=None,
        first_opinion=None,
        
        # CoT 분석 결과 저장
        inference_cot=None,
        
        # 루프 제어
        retry_count=0,
        max_retries=5,
        
        # 상태 관리
        current_step="inference",
        error=None
    ) 