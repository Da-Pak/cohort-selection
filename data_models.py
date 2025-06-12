from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import pandas as pd


class UploadRequest(BaseModel):
    """업로드 요청을 위한 모델"""
    target_column: str = Field(..., description="처리할 대상 열 이름")


class FilterRequest(BaseModel):
    """필터 요청을 위한 모델"""
    question: str = Field(..., description="사용자의 자연어 질문")
    data_id: str = Field(..., description="필터링할 데이터의 고유 ID")
    target_column: Optional[str] = Field(None, description="처리할 대상 열 이름 (없으면 업로드 시 지정된 열 사용)")


class UploadResponse(BaseModel):
    """업로드 응답을 위한 모델"""
    data_id: str = Field(..., description="업로드된 데이터의 고유 ID")
    rows: int = Field(..., description="업로드된 데이터의 행 수")
    columns: List[str] = Field(..., description="업로드된 데이터의 열 이름 목록")
    preview: List[Dict[str, Any]] = Field(..., description="업로드된 데이터의 미리보기 (처음 5행)")
    target_column: str = Field(..., description="처리할 대상 열 이름")


class FilterResponse(BaseModel):
    """Model for filter response"""
    data_id: str = Field(..., description="Unique ID of the filtered data")
    task_id: str = Field(..., description="Unique ID of the task")
    original_rows: int = Field(..., description="Number of rows in the original data")
    filtered_rows: int = Field(..., description="Number of rows in the filtered data")
    results: List[Dict[str, Any]] = Field([], description="Filtered data results")
    reasoning: List[str] = Field([], description="Reasoning for each filtered row")


class StatusResponse(BaseModel):
    """Model for status response"""
    task_id: str = Field(..., description="Unique ID of the task")
    status: str = Field(..., description="Task status (pending, processing, completed, failed)")
    progress: float = Field(..., description="Task progress (0.0 to 1.0)")
    message: Optional[str] = Field(None, description="Status message")


class InferenceResult(BaseModel):
    """Model for Local LLM inference result"""
    sentence: str = Field(..., description="Sentence that serves as the basis for judgment")
    opinion: str = Field(..., description="Judgment result (INCLUDE, EXCLUDE, UNCERTAIN)")
    verified_sentence: bool = Field(False, description="Sentence verification result")
    verified_opinion: Optional[bool] = Field(None, description="GPT-4 verification result")


class PipelineConfig(BaseModel):
    """Model for pipeline configuration"""
    use_gpt_verification: bool = Field(True, description="Use GPT-4 verification")
    batch_size: int = Field(10, description="Batch size")
    max_workers: int = Field(4, description="Maximum number of concurrent workers")
    timeout: int = Field(60, description="Timeout (seconds)")
    retry_count: int = Field(3, description="Retry count") 