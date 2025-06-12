from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional, List
import logging
import sys
import os
import json
import asyncio
from pydantic import BaseModel
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
import os
import logging
import uuid
import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Form, Depends, Header, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from data_models import FilterRequest, UploadResponse, FilterResponse, StatusResponse, UploadRequest
from utils import save_dataframe, get_dataframe, create_task, update_task_status, get_task_status, update_config, get_config, get_metadata
from graph import filter_runner
# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 로깅 설정
logger = logging.getLogger(__name__)

# API 키 설정
API_KEY = os.environ.get("X_API_KEY", "your-default-api-key-change-in-production")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# API 키 검증 함수
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="유효하지 않은 API 키입니다."
        )
    return api_key


# API 라우터 생성
routes = APIRouter()

@routes.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medical Imaging Report Filtering API",
        "documentation": "/docs",
        "version": "1.0.0"
    }


@routes.post("/upload", response_model=UploadResponse, dependencies=[Depends(get_api_key)])
async def upload_data(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    """
    CSV 또는 Excel 파일을 업로드하고 데이터프레임으로 변환합니다.
    
    Args:
        file (UploadFile): 업로드할 파일
        target_column (str): 처리할 대상 열 이름
    
    Returns:
        UploadResponse: 업로드 응답
    """
    try:
        # 파일 확장자 확인
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        # 파일 내용 읽기
        contents = await file.read()
        
        # 파일 타입에 따라 데이터프레임 생성
        if file_ext == '.csv':
            df = pd.read_csv(pd.io.common.BytesIO(contents))
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(pd.io.common.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다. CSV 또는 Excel 파일만 허용됩니다.")
        
        # 지정된 열이 있는지 확인
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"데이터에 '{target_column}' 열이 없습니다. 파일에 존재하는 열: {list(df.columns)}")
        
        # 메타데이터 추가 - 대상 열 정보 저장
        metadata = {"target_column": target_column}
        
        # 데이터프레임 저장 및 ID 반환
        data_id = save_dataframe(df, metadata=metadata)
        
        # 응답 생성
        response = UploadResponse(
            data_id=data_id,
            rows=len(df),
            columns=list(df.columns),
            preview=df.head(5).to_dict('records'),
            target_column=target_column  # 응답에 target_column 포함
        )
        
        logger.info(f"데이터 업로드 완료: {file.filename}, 데이터 ID: {data_id}, 행 수: {len(df)}, 대상 열: {target_column}")
        
        return response
        
    except Exception as e:
        logger.error(f"데이터 업로드 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"데이터 처리 중 오류 발생: {str(e)}")


@routes.post("/filter", response_model=FilterResponse, dependencies=[Depends(get_api_key)])
async def filter_data(filter_request: FilterRequest, background_tasks: BackgroundTasks):
    """
    데이터프레임에 필터를 적용하여 조건에 맞는 행을 필터링합니다.
    
    Args:
        filter_request (FilterRequest): 필터 요청
        background_tasks (BackgroundTasks): 백그라운드 작업
        
    Returns:
        FilterResponse: 필터 응답
    """
    try:
        # 데이터프레임 가져오기
        df = get_dataframe(filter_request.data_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"데이터를 찾을 수 없음: ({filter_request.data_id}).")
        
        # 메타데이터 가져오기
        metadata = get_metadata(filter_request.data_id)
        
        # 대상 열 결정 (요청에 있으면 그것을 사용, 없으면 메타데이터에서 가져옴)
        target_column = filter_request.target_column
        if target_column is None and metadata and "target_column" in metadata:
            target_column = metadata["target_column"]
        
        # 대상 열이 없으면 오류 발생
        if not target_column:
            raise HTTPException(status_code=400, detail="대상 열을 지정해주세요.")
        
        # 대상 열이 데이터프레임에 있는지 확인
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"데이터에 '{target_column}' 열이 없습니다. 사용 가능한 열: {list(df.columns)}")
        
        # 작업 생성
        task_id = create_task()
        
        # 백그라운드에서 필터링 실행
        background_tasks.add_task(
            filter_data_background,
            task_id,
            filter_request.data_id,
            filter_request.question,
            target_column
        )
        
        # 초기 응답 반환
        return FilterResponse(
            data_id=filter_request.data_id,
            task_id=task_id,
            original_rows=len(df),
            filtered_rows=0,
            results=[],
            reasoning=[]
        )
        
    except Exception as e:
        logger.error(f"필터링 요청 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"필터링 중 오류 발생: {str(e)}")


async def filter_data_background(task_id: str, data_id: str, question: str, target_column: str):
    """
    백그라운드에서 데이터 필터링을 수행합니다.
    
    Args:
        task_id (str): 작업 ID
        data_id (str): 데이터 ID
        question (str): 사용자 질문
        target_column (str): 처리할 대상 열
    """
    try:
        # 데이터프레임 가져오기
        df = get_dataframe(data_id)
        if df is None:
            update_task_status(task_id, "failed", 0.0, f"데이터를 찾을 수 없음: ({data_id}).")
            return
        
        # 최대 실행 시간 설정 (5분)
        max_execution_time = 300  # 초 단위
        start_time = time.time()
        
        # 필터 그래프 실행 (target_column 정보 전달)
        filtered_df, result_data_id = filter_runner.run_filter(question, data_id, task_id, target_column)
        filtered_df_len = len(filtered_df[filtered_df["final_opinion"] == "INCLUDE"])
        # 실행 시간이 너무 길었는지 확인
        elapsed_time = time.time() - start_time
        if elapsed_time > max_execution_time:
            logger.warning(f"필터링 시간이 너무 오래 걸림: {elapsed_time:.2f}초")
        
        # 여기서 작업 상태를 업데이트할 필요가 없음 - 그래프에서 이미 업데이트함
        if result_data_id:
            # 결과 메시지에 데이터 ID를 명확하게 포함
            completion_message = f"필터링 완료: 원본 {len(df)}행 → 결과 {filtered_df_len}행, 결과 데이터 ID: {result_data_id}"
            update_task_status(task_id, "completed", 1.0, completion_message)
            logger.info(f"필터링 완료: 원본 {len(df)}행 → 결과 {filtered_df_len}행, 결과 데이터 ID: {result_data_id}")
        else:
            logger.error(f"작업 ID에 대한 필터링 실패: {task_id}")
            update_task_status(task_id, "failed", 0.0, "필터링 실패: 결과를 생성할 수 없습니다.")
        
    except Exception as e:
        logger.error(f"백그라운드 필터링 중 오류 발생: {e}")
        update_task_status(task_id, "failed", 0.0, f"필터링 중 오류 발생: {str(e)}")


@routes.get("/status/{task_id}", response_model=StatusResponse, dependencies=[Depends(get_api_key)])
async def get_status(task_id: str):
    """
    Get task status.
    
    Args:
        task_id (str): Task ID
        
    Returns:
        StatusResponse: Task status response
    """
    try:
        # Get task status
        status = get_task_status(task_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Task not found for ID ({task_id}).")
        
        # Create response
        response = StatusResponse(
            task_id=task_id,
            status=status["status"],
            progress=status["progress"],
            message=status.get("message")
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error during status check: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking task status: {str(e)}")


@routes.get("/result/{data_id}", dependencies=[Depends(get_api_key)])
async def get_result(data_id: str, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    """
    Get filtering results.
    
    Args:
        data_id (str): Data ID
        limit (int): Maximum number of rows to return
        offset (int): Starting offset
        
    Returns:
        Dict: Filtering results
    """
    try:
        # Get dataframe
        df = get_dataframe(data_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Data not found for ID ({data_id}).")
        
        # Create result
        total_rows = len(df)
        rows = df.iloc[offset:offset+limit].to_dict('records')
        
        return {
            "data_id": data_id,
            "total_rows": total_rows,
            "limit": limit,
            "offset": offset,
            "rows": rows
        }
        
    except Exception as e:
        logger.error(f"Error during result retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")


@routes.get("/download/{data_id}", dependencies=[Depends(get_api_key)])
async def download_result(data_id: str):
    """
    Download results as CSV.
    
    Args:
        data_id (str): Data ID
        
    Returns:
        Response: CSV file
    """
    try:
        # Get dataframe
        df = get_dataframe(data_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Data not found for ID ({data_id}).")
        
        # Convert to CSV
        csv_content = df.to_csv(index=False)
        
        # Create response
        return JSONResponse(
            content={
                "csv": csv_content,
                "filename": f"filtered_data_{data_id}.csv"
            }
        )
        
    except Exception as e:
        logger.error(f"Error during download: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating download: {str(e)}")


@routes.post("/config", dependencies=[Depends(get_api_key)])
async def update_pipeline_config(config: dict):
    """
    Update pipeline configuration.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        Dict: Updated configuration
    """
    try:
        # Get current config
        current_config = get_config()
        
        # Update config with provided values
        for key, value in config.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
        
        # Save updated config
        update_config(current_config)
        
        return {"message": "Configuration updated successfully", "config": current_config.dict()}
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")