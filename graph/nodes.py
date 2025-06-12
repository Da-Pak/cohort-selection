import logging
import time
import numpy as np

from utils import get_dataframe, update_task_status, get_config
from .agents.prompt_generator import generate_context
from .agents.llm_inference import inference_llm
from verifier import verify_sentence
from .agents.llm_verifier import verify_opinion
from .state import FilterState

logger = logging.getLogger(__name__)


def load_data(state: FilterState) -> FilterState:
    """데이터를 캐시에서 로드하고 상태를 업데이트합니다."""
    data_id = state["data_id"]
    task_id = state["task_id"]
    target_column = state["target_column"]
    
    try:
        # 상태 업데이트
        update_task_status(task_id, "processing", 0.1, "데이터 로드 중")
        
        # 데이터프레임 가져오기
        df = get_dataframe(data_id)
        if df is None:
            error_msg = f"데이터를 찾을 수 없음: {data_id}"
            logger.error(error_msg)
            update_task_status(task_id, "failed", 0.0, error_msg)
            return {**state, "status": "failed", "error": error_msg}
        
        # 대상 열이 있는지 확인
        if target_column not in df.columns:
            error_msg = f"데이터에 '{target_column}' 열이 없습니다. 사용 가능한 열: {list(df.columns)}"
            logger.error(error_msg)
            update_task_status(task_id, "failed", 0.0, error_msg)
            return {**state, "status": "failed", "error": error_msg}
        
        logger.info(f"데이터 로드 완료: {data_id}, 행 수: {len(df)}, 대상 열: {target_column}")
        update_task_status(task_id, "processing", 0.1, f"데이터 로드 완료: {len(df)}행, 대상 열: {target_column}")
        
        return {**state, "dataframe": df, "progress": 0.1, "message": f"데이터 로드 완료: {len(df)}행, 대상 열: {target_column}"}
        
    except Exception as e:
        error_msg = f"데이터 로드 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        update_task_status(task_id, "failed", 0.0, error_msg)
        return {**state, "status": "failed", "error": error_msg}


def generate_medical_context(state: FilterState) -> FilterState:
    """Generate medical context from user question"""
    question = state["question"]
    task_id = state["task_id"]
    temperature = state['temperature']
    
    try:
        # Update status
        update_task_status(task_id, "processing", 0.2, "Generating medical context")
        
        # Generate context
        context = generate_context(question, temperature)
        logger.info(f"Context generated: {len(context)} cha, racters")
        update_task_status(task_id, "processing", 0.3, "Context generated")
        
        return {**state, "context": context, "progress": 0.3, "message": "Context generated"}
        
    except Exception as e:
        error_msg = f"Error generating context: {str(e)}"
        logger.error(error_msg)
        update_task_status(task_id, "failed", 0.2, error_msg)
        return {**state, "status": "failed", "error": error_msg}


def inference_case(state: FilterState) -> FilterState:
    """모든 텍스트를 순차적으로 처리합니다."""
    task_id = state["task_id"]
    df = state["dataframe"]
    context = state["context"]
    target_column = state["target_column"]
    temperature = state["temperature"]
    config = get_config()
    
    try:
        # 모든 텍스트 가져오기
        texts = df[target_column].tolist()
        total_texts = len(texts)
        
        # 상태 업데이트
        update_task_status(
            task_id, 
            "processing", 
            0.3, 
            "데이터 처리 중"
        )
        
        # 결과 리스트 초기화
        inference_results = []
        
        # 각 텍스트 순차적으로 처리
        for idx, text in enumerate(texts):
            # 진행률 계산
            progress = np.round((idx / total_texts) * 100, 2)

            logger.info(f"{idx+1}번째 시작 : 진행률 {progress} -----------------------------------")
            
            # 상태 업데이트
            if idx % 10 == 0 or idx == len(texts) - 1:  # 10개 항목마다 또는 마지막 항목에서 상태 업데이트
                update_task_status(
                    task_id, 
                    "processing", 
                    progress, 
                    f"항목 처리 중 {idx+1}/{total_texts}"
                )
            
            # 단일 텍스트에 대한 추론 실행
            try:
                result = inference_llm(
                    text, 
                    context, 
                    state["question"],
                    llm_type=config.llm_config.llm_type,
                    temperature = temperature
                )
            except Exception as inference_error:
                logger.error(f"추중 오류 발생: {inference_error}")
                result['sentence'] = None
                result['opinion'] = None
                
            # 문장 검증
            sentence = result.get("sentence", "")
            result["verified_sentence"] = verify_sentence(text, sentence)
            
            try:
                result["verified_opinion"] = verify_opinion(
                    text, 
                    state["question"], 
                    context,
                    result,
                    llm_type=config.llm_config.llm_type,
                    temperature = temperature
                )
            except Exception as verify_error:
                logger.error(f"의견 검증 중 오류 발생: {verify_error}")
                result["verified_opinion"] = None  # 검증 실패 시 None으로 처리
            
            # 결과에 추가
            inference_results.append(result)
        
        # 상태 업데이트
        update_task_status(
            task_id, 
            "processing", 
            0.8, 
            "데이터 처리 완료"
        )
        
        return {
            **state, 
            "results": inference_results, 
            "progress": 0.8,
            "status": "completed",  # 명시적으로 completed 상태 설정
            "message": "데이터 처리 완료"
        }
        
    except Exception as e:
        error_msg = f"데이터 처리 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        
        # 재시도 카운터 증가
        retries = state["retries"] + 1
        
        if retries <= config.retry_count:
            logger.info(f"데이터 처리 재시도 중, 시도 {retries}/{config.retry_count}")
            time.sleep(1)  # 재시도 전 대기
            return {**state, "retries": retries}
        else:
            update_task_status(task_id, "failed", 0.3, error_msg)
            return {**state, "status": "failed", "error": error_msg}


def validate_sentence(state: FilterState) -> FilterState:
    """모든 텍스트를 순차적으로 처리합니다."""
    task_id = state["task_id"]
    df = state["dataframe"]
    context = state["context"]
    target_column = state["target_column"]
    temperature = state["temperature"]
    config = get_config()
    
    try:
        # 모든 텍스트 가져오기
        texts = df[target_column].tolist()
        total_texts = len(texts)
        
        # 상태 업데이트
        update_task_status(
            task_id, 
            "processing", 
            0.3, 
            "데이터 처리 중"
        )
        
        # 결과 리스트 초기화
        inference_results = []
        
        # 각 텍스트 순차적으로 처리
        for idx, text in enumerate(texts):
            # 진행률 계산
            progress = np.round((idx / total_texts) * 100, 2)

            logger.info(f"{idx+1}번째 시작 : 진행률 {progress} -----------------------------------")
            
            # 상태 업데이트
            if idx % 10 == 0 or idx == len(texts) - 1:  # 10개 항목마다 또는 마지막 항목에서 상태 업데이트
                update_task_status(
                    task_id, 
                    "processing", 
                    progress, 
                    f"항목 처리 중 {idx+1}/{total_texts}"
                )
            
            # 단일 텍스트에 대한 추론 실행
            try:
                result = inference_llm(
                    text, 
                    context, 
                    state["question"],
                    llm_type=config.llm_config.llm_type,
                    temperature = temperature
                )
            except Exception as inference_error:
                logger.error(f"추중 오류 발생: {inference_error}")
                result['sentence'] = None
                result['opinion'] = None
                
            # 문장 검증
            sentence = result.get("sentence", "")
            result["verified_sentence"] = verify_sentence(text, sentence)
            
            try:
                result["verified_opinion"] = verify_opinion(
                    text, 
                    state["question"], 
                    context,
                    result,
                    llm_type=config.llm_config.llm_type,
                    temperature = temperature
                )
            except Exception as verify_error:
                logger.error(f"의견 검증 중 오류 발생: {verify_error}")
                result["verified_opinion"] = None  # 검증 실패 시 None으로 처리
            
            # 결과에 추가
            inference_results.append(result)
        
        # 상태 업데이트
        update_task_status(
            task_id, 
            "processing", 
            0.8, 
            "데이터 처리 완료"
        )
        
        return {
            **state, 
            "results": inference_results, 
            "progress": 0.8,
            "status": "completed",  # 명시적으로 completed 상태 설정
            "message": "데이터 처리 완료"
        }
        
    except Exception as e:
        error_msg = f"데이터 처리 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        
        # 재시도 카운터 증가
        retries = state["retries"] + 1
        
        if retries <= config.retry_count:
            logger.info(f"데이터 처리 재시도 중, 시도 {retries}/{config.retry_count}")
            time.sleep(1)  # 재시도 전 대기
            return {**state, "retries": retries}
        else:
            update_task_status(task_id, "failed", 0.3, error_msg)
            return {**state, "status": "failed", "error": error_msg}

def validate_case(state: FilterState) -> FilterState:
    """모든 텍스트를 순차적으로 처리합니다."""
    task_id = state["task_id"]
    df = state["dataframe"]
    context = state["context"]
    target_column = state["target_column"]
    temperature = state["temperature"]
    config = get_config()
    
    try:
        # 모든 텍스트 가져오기
        texts = df[target_column].tolist()
        total_texts = len(texts)
        
        # 상태 업데이트
        update_task_status(
            task_id, 
            "processing", 
            0.3, 
            "데이터 처리 중"
        )
        
        # 결과 리스트 초기화
        inference_results = []
        
        # 각 텍스트 순차적으로 처리
        for idx, text in enumerate(texts):
            # 진행률 계산
            progress = np.round((idx / total_texts) * 100, 2)

            logger.info(f"{idx+1}번째 시작 : 진행률 {progress} -----------------------------------")
            
            # 상태 업데이트
            if idx % 10 == 0 or idx == len(texts) - 1:  # 10개 항목마다 또는 마지막 항목에서 상태 업데이트
                update_task_status(
                    task_id, 
                    "processing", 
                    progress, 
                    f"항목 처리 중 {idx+1}/{total_texts}"
                )
            
            # 단일 텍스트에 대한 추론 실행
            try:
                result = inference_llm(
                    text, 
                    context, 
                    state["question"],
                    llm_type=config.llm_config.llm_type,
                    temperature = temperature
                )
            except Exception as inference_error:
                logger.error(f"추중 오류 발생: {inference_error}")
                result['sentence'] = None
                result['opinion'] = None
                
            # 문장 검증
            sentence = result.get("sentence", "")
            result["verified_sentence"] = verify_sentence(text, sentence)
            
            try:
                result["verified_opinion"] = verify_opinion(
                    text, 
                    state["question"], 
                    context,
                    result,
                    llm_type=config.llm_config.llm_type,
                    temperature = temperature
                )
            except Exception as verify_error:
                logger.error(f"의견 검증 중 오류 발생: {verify_error}")
                result["verified_opinion"] = None  # 검증 실패 시 None으로 처리
            
            # 결과에 추가
            inference_results.append(result)
        
        # 상태 업데이트
        update_task_status(
            task_id, 
            "processing", 
            0.8, 
            "데이터 처리 완료"
        )
        
        return {
            **state, 
            "results": inference_results, 
            "progress": 0.8,
            "status": "completed",  # 명시적으로 completed 상태 설정
            "message": "데이터 처리 완료"
        }
        
    except Exception as e:
        error_msg = f"데이터 처리 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        
        # 재시도 카운터 증가
        retries = state["retries"] + 1
        
        if retries <= config.retry_count:
            logger.info(f"데이터 처리 재시도 중, 시도 {retries}/{config.retry_count}")
            time.sleep(1)  # 재시도 전 대기
            return {**state, "retries": retries}
        else:
            update_task_status(task_id, "failed", 0.3, error_msg)
            return {**state, "status": "failed", "error": error_msg}


def finalize_results(state: FilterState) -> FilterState:
    """결과를 마무리하고 필터링된 데이터프레임을 생성합니다."""
    task_id = state["task_id"]
    df = state["dataframe"]
    results = state["results"]
    target_column = state["target_column"]
    temperature= state["temperature"]
    config = get_config()
    
    try:
        # 상태 업데이트
        update_task_status(task_id, "processing", 0.9, "결과 마무리 중")
        
        # 결과 데이터프레임 생성
        result_df = df.copy()
        result_df["sentence"] = [result.get("sentence", "") for result in results]
        result_df["opinion"] = [result.get("opinion", "Uncertain") for result in results]
        result_df["verified_sentence"] = [result.get("verified_sentence", False) for result in results]
        
        if config.use_gpt_verification:
            result_df["verified_opinion"] = [result.get("verified_opinion", None) for result in results]
            # 최종 결정 (verified_opinion이 True이거나 None이면 opinion 사용, 그렇지 않으면 "Absent")
            result_df["final_opinion"] = result_df.apply(
                lambda row: row["opinion"] if row["verified_opinion"] == True else "Absent", 
                axis=1
            )
        else:
            result_df["final_opinion"] = result_df["opinion"]
        
        # 필터링 안하고 반환
        filtered_df = result_df.copy()
        
        # 결과 이유 추출
        reasoning = [f"전체 {len(df)}개 중 {len(filtered_df)}개의 일치하는 케이스를 찾았습니다."]
        
        # 상태 업데이트
        update_task_status(
            task_id, 
            "completed", 
            1.0, 
            f"필터링 완료: 전체 {len(df)}행 중 {len(filtered_df)}행 일치, 대상 열: {target_column}"
        )
        
        logger.info(f"필터링 완료: 전체 {len(df)}행 중 {len(filtered_df)}행 일치, 대상 열: {target_column}")
        
        return {
            **state, 
            "filtered_dataframe": filtered_df,
            "reasoning": reasoning,
            "status": "completed",
            "progress": 1.0, 
            "message": f"필터링 완료: 전체 {len(df)}행 중 {len(filtered_df)}행 일치, 대상 열: {target_column}"
        }
        
    except Exception as e:
        error_msg = f"결과 마무리 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        update_task_status(task_id, "failed", 0.9, error_msg)
        return {**state, "status": "failed", "error": error_msg}


def handle_error(state: FilterState) -> FilterState:
    """파이프라인 오류 처리"""
    task_id = state["task_id"]
    error = state.get("error", "알 수 없는 오류")
    
    logger.error(f"파이프라인 오류: {error}")
    update_task_status(task_id, "failed", 0.0, f"오류: {error}")
    
    return {**state, "status": "failed", "message": f"오류: {error}"} 