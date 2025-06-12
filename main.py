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
from router import routes



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

# Create FastAPI app
app = FastAPI(
    title="Medical Imaging Report Filtering API",
    description="AI-based system API for automatically identifying specific cases",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트엔드 도메인 지정
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# API 라우터 포함
app.include_router(routes, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 