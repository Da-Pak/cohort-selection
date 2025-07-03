import logging
import time
import numpy as np
from langgraph.graph import StateGraph, END
from typing import Literal

from ..utils import get_dataframe, update_task_status, get_config
from .agents.prompt_generator import generate_context
from .agents.llm_inference import inference_llm
from ..verifier import verify_sentence
from .state import FilterState, SingleTextState, get_initial_single_text_state

logger = logging.getLogger(__name__)

def inference_single_text(state: SingleTextState) -> SingleTextState:
    """단일 텍스트에 대한 추론을 수행합니다."""
    try:
        logger.info(f"추론 시작 - 시도 {state['retry_count'] + 1}/{state['max_retries']}")
        
        # LLM 추론 실행 (verifier 피드백 제거)
        result = inference_llm(
            state["text"], 
            state["context"], 
            state["question"],
            llm_type=state["llm_type"],
            temperature=state["temperature"],
            verifier_feedback=None
        )
        
        # 결과 검증 - 필수 키가 있는지 확인
        sentence = result.get("sentence", "").strip() if result.get("sentence") else ""
        opinion = result.get("opinion", "").strip() if result.get("opinion") else ""
        
        # 추론 결과가 유효한지 확인
        if not sentence or not opinion:
            logger.warning("추론 결과가 불완전함 - 재시도 필요")
            return {
                **state,
                "sentence": None,
                "opinion": None,
                "current_step": "inference",
                "retry_count": state["retry_count"] + 1,
                "error": "추론 결과가 불완전함"
            }
        
        # 첫 번째 시도인 경우 첫 번째 결과를 저장
        first_sentence = state.get("first_sentence")
        first_opinion = state.get("first_opinion")
        if state["retry_count"] == 0:
            first_sentence = sentence
            first_opinion = opinion
            logger.info(f"첫 번째 추론 결과 저장: {first_opinion}")
        
        # CoT 분석 결과 저장
        inference_cot = {
            "step1_analysis_and_evidence": result.get("step1_analysis_and_evidence", ""),
            "step2_classification_reasoning": result.get("step2_classification_reasoning", "")
        }
        
        logger.info("추론 성공")
        return {
            **state,
            "sentence": sentence,
            "opinion": opinion,
            "first_sentence": first_sentence,
            "first_opinion": first_opinion,
            "inference_cot": inference_cot,
            "current_step": "validate_sentence",
            "error": None
        }
        
    except Exception as e:
        error_msg = f"추론 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        return {
            **state,
            "sentence": None,
            "opinion": None,
            "current_step": "inference",
            "retry_count": state["retry_count"] + 1,
            "error": error_msg
        }


def validate_sentence_node(state: SingleTextState) -> SingleTextState:
    """문장 검증을 수행합니다."""
    try:
        logger.info("문장 검증 시작")
        
        sentence = state.get("sentence", "")
        if not sentence:
            # 이 상황은 라우터에서 이미 체크되어야 하지만, 안전을 위해 처리
            logger.error("문장이 없음 - 예상치 못한 상황")
            return {
                **state,
                "verified_sentence": False,
                "current_step": "inference",
                "retry_count": state["retry_count"] + 1,
                "error": "문장이 없음"
            }
        
        # 문장 검증 실행
        verified = verify_sentence(state["text"], sentence)
        
        if verified:
            logger.info("문장 검증 성공 - 처리 완료")
            return {
                **state,
                "verified_sentence": True,
                "current_step": "completed",
                "error": None
            }
        else:
            logger.info("문장 검증 실패 - 재시도 필요")
            return {
                **state,
                "verified_sentence": False,
                "current_step": "inference",
                "retry_count": state["retry_count"] + 1,
                "error": None  # 검증 실패는 오류가 아님
            }
            
    except Exception as e:
        error_msg = f"문장 검증 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        return {
            **state,
            "verified_sentence": False,
            "current_step": "inference",
            "retry_count": state["retry_count"] + 1,
            "error": error_msg
        }


def route_after_inference(state: SingleTextState) -> Literal["validate_sentence", "inference", "completed"]:
    """추론 후 다음 단계 결정"""
    # 최대 재시도 횟수 초과 시 종료
    if state["retry_count"] >= state["max_retries"]:
        logger.warning(f"최대 재시도 횟수({state['max_retries']}) 초과 - 처리 종료")
        return "completed"
    
    # 추론 결과 확인
    sentence = state.get("sentence")
    opinion = state.get("opinion")
    error = state.get("error")
    
    # 오류가 있거나 추론 결과가 없는 경우 재시도
    if error or not sentence or not opinion:
        logger.info("추론 실패 - 재시도 필요")
        return "inference"
    
    # 추론 성공 시 문장 검증으로 진행
    logger.info("추론 성공 - 문장 검증 단계로 진행")
    return "validate_sentence"


def route_after_sentence_validation(state: SingleTextState) -> Literal["inference", "completed"]:
    """문장 검증 후 다음 단계 결정"""
    # 최대 재시도 횟수 초과 시 종료
    if state["retry_count"] >= state["max_retries"]:
        logger.warning(f"최대 재시도 횟수({state['max_retries']}) 초과 - 처리 종료")
        return "completed"
    
    verified_sentence = state.get("verified_sentence")
    
    # 문장 검증 성공 시 완료
    if verified_sentence is True:
        logger.info("문장 검증 성공 - 처리 완료")
        return "completed"
    
    # 문장 검증 실패 시 추론부터 다시 시작
    elif verified_sentence is False:
        logger.info("문장 검증 실패 - 추론 단계로 재시도")
        return "inference"
    
    # 예상치 못한 상황 (verified_sentence가 None 등)
    logger.warning("문장 검증 결과가 예상치 못한 값 - 추론 단계로 재시도")
    return "inference"


def create_single_text_subgraph():
    """단일 텍스트 처리용 서브그래프를 생성합니다."""
    sub = StateGraph(SingleTextState)
    
    # 노드 추가 (validate_case 제거)
    sub.add_node("inference", inference_single_text)
    sub.add_node("validate_sentence", validate_sentence_node)
    
    # 엣지 추가 (조건부 라우팅)
    sub.add_conditional_edges(
        "inference",
        route_after_inference,
        {
            "validate_sentence": "validate_sentence",
            "inference": "inference",  # 추론 실패 시 재시도
            "completed": END
        }
    )
    
    sub.add_conditional_edges(
        "validate_sentence", 
        route_after_sentence_validation,
        {
            "inference": "inference",
            "completed": END
        }
    )
    
    sub.set_entry_point("inference")
    
    return sub.compile()

single_text_subgraph = create_single_text_subgraph()

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
        logger.info(f"Context generated: {len(context)} characters")
        update_task_status(task_id, "processing", 0.3, "Context generated")
        
        return {**state, "context": context, "progress": 0.3, "message": "Context generated"}
        
    except Exception as e:
        error_msg = f"Error generating context: {str(e)}"
        logger.error(error_msg)
        update_task_status(task_id, "failed", 0.2, error_msg)
        return {**state, "status": "failed", "error": error_msg}


def process_all_texts(state: FilterState) -> FilterState:
    """서브그래프를 사용하여 모든 텍스트를 처리합니다."""
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
        update_task_status(task_id, "processing", 0.3, "데이터 처리 중")
        
        # 결과 리스트 초기화
        inference_results = []
        
        # 각 텍스트를 서브그래프로 처리
        for idx, text in enumerate(texts):
            # 진행률 계산
            progress = 0.3 + (idx / total_texts) * 0.5  # 0.3 ~ 0.8 범위
            
            logger.info(f"{idx+1}번째 텍스트 처리 시작 : 진행률 {progress:.1%}")
            
            # 상태 업데이트
            if idx % 10 == 0 or idx == len(texts) - 1:
                update_task_status(
                    task_id, 
                    "processing", 
                    progress, 
                    f"항목 처리 중 {idx+1}/{total_texts}"
                )
            
            # 서브그래프 입력 준비
            sub_input = get_initial_single_text_state(
                text=text,
                context=context,
                question=state["question"],
                temperature=temperature,
                llm_type=config.llm_config.llm_type
            )
            
            try:
                # 서브그래프 실행
                sub_result = single_text_subgraph.invoke(sub_input)
                
                # 결과 변환
                result = {
                    "sentence": sub_result.get("sentence", ""),
                    "opinion": sub_result.get("opinion", "Uncertain"),
                    "first_sentence": sub_result.get("first_sentence", ""),
                    "first_opinion": sub_result.get("first_opinion", "Uncertain"),
                    "verified_sentence": sub_result.get("verified_sentence", False),
                    "retry_count": sub_result.get("retry_count", 0),
                    "error": sub_result.get("error", None),
                    "inference_cot": sub_result.get("inference_cot", None)  # inference CoT 포함
                }
                
            except Exception as sub_error:
                logger.error(f"서브그래프 실행 중 오류 발생: {sub_error}")
                result = {
                    "sentence": "Error during processing",
                    "opinion": "Uncertain",  # 오류 발생 시 Uncertain으로 설정
                    "first_sentence": "Error during processing",
                    "first_opinion": "Uncertain",  # 첫 번째도 Uncertain으로 설정
                    "verified_sentence": False,
                    "retry_count": 5,
                    "error": str(sub_error),
                    "inference_cot": None
                }
            
            # 결과에 추가
            inference_results.append(result)
        
        # 상태 업데이트
        update_task_status(task_id, "processing", 0.8, "데이터 처리 완료")
        
        return {
            **state, 
            "results": inference_results, 
            "progress": 0.8,
            "status": "processing",
            "message": "데이터 처리 완료"
        }
        
    except Exception as e:
        error_msg = f"데이터 처리 중 오류 발생: {str(e)}"
        logger.error(error_msg)
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
        result_df["first_sentence"] = [result.get("first_sentence", "") for result in results]
        result_df["first_opinion"] = [result.get("first_opinion", "Uncertain") for result in results]
        result_df["retry_count"] = [result.get("retry_count", 0) for result in results]
        result_df["verified_sentence"] = [result.get("verified_sentence", False) for result in results]
        
        # CoT 분석 결과 포함 (요약된 형태로)
        def extract_cot_summary(cot_data, step_key):
            if not cot_data or not cot_data.get(step_key):
                return ""
            text = cot_data.get(step_key, "")
            # 너무 긴 텍스트는 요약하여 저장
            return text[:200] + "..." if len(text) > 200 else text
        
        result_df["inference_analysis_and_evidence"] = [
            extract_cot_summary(result.get("inference_cot"), "step1_analysis_and_evidence") for result in results
        ]
        result_df["inference_classification_reasoning"] = [
            extract_cot_summary(result.get("inference_cot"), "step2_classification_reasoning") for result in results
        ]
        
        # 최종 결정 (항상 inference의 opinion 사용)
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