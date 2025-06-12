import logging
from typing import Dict, Any, List, Optional, Tuple, Literal
from langgraph.graph import StateGraph, END
import pandas as pd

from ..utils import get_config, save_dataframe
from .state import FilterState, get_initial_state
from .nodes import load_data, generate_medical_context, inference_case, validate_case, validate_sentence, finalize_results, handle_error

logger = logging.getLogger(__name__)


def should_retry(state: FilterState) -> Literal["retry", "error", "continue"]:
    """Decide whether to retry processing or fail"""
    # 먼저 실패한 경우 처리
    if state["status"] == "failed":
        return "error"
    
    # 성공적으로 처리된 경우 (status가 completed인 경우)
    if state["status"] == "completed":
        return "continue"
    
    # 결과가 생성된 경우도 성공으로 간주
    if "results" in state and len(state["results"]) > 0:
        return "continue"
    
    # 재시도 횟수 확인
    config = get_config()
    if state.get("retries", 0) <= config.retry_count:
        return "retry"
    else:
        return "error"


def create_filter_graph() -> StateGraph:
    """Create the filtering graph structure"""
    # Create the graph
    graph = StateGraph(FilterState)
    
    # Add nodes
    graph.add_node("load_data", load_data)
    graph.add_node("generate_context", generate_medical_context)
    graph.add_node("inference_case", inference_case)
    graph.add_node("finalize_results", finalize_results)
    graph.add_node("handle_error", handle_error)
    
    # Define the edges
    graph.add_edge("load_data", "generate_context")
    graph.add_edge("generate_context", "inference_data")
    
    # Conditional edges for processing
    graph.add_conditional_edges(
        "inference_case",
        should_retry,
        {
            "retry": "inference_case",
            "error": "handle_error",
            "continue": "finalize_results"
        }
    )
    
    # Connect error handler and finalize to end
    graph.add_edge("handle_error", END)
    graph.add_edge("finalize_results", END)
    
    # Set the entry point
    graph.set_entry_point("load_data")
    
    return graph.compile()


class FilterRunner:
    """Runner for the filtering graph"""
    
    def __init__(self):
        """Initialize the filter runner"""
        self.graph = create_filter_graph()
        logger.info("Filter graph initialized")
    
    def run_filter(self, question: str, data_id: str, task_id: str, target_column: str, temperature: float) -> Tuple[pd.DataFrame, str]:
        """
        필터링 프로세스를 실행합니다.
        
        Args:
            question (str): 사용자 질문
            data_id (str): 필터링할 데이터프레임의 ID
            task_id (str): 상태 추적을 위한 작업 ID
            target_column (str): 필터링할 대상 열 이름 (기본값: "text")
            
        Returns:
            Tuple[pd.DataFrame, str]: 필터링된 데이터프레임과 결과 데이터 ID
        """
        try:
            # 초기 상태 생성
            initial_state = get_initial_state(question, data_id, task_id, target_column, temperature)
            
            # 그래프 실행
            logger.info(f"필터 그래프 실행 시작: {task_id}, 대상 열: {target_column}")
            final_state = self.graph.invoke(initial_state)
            
            # 필터링된 데이터프레임 추출
            filtered_df = final_state.get("filtered_dataframe")
            
            if filtered_df is None or final_state.get("status") == "failed":
                logger.error(f"필터 그래프 실패: {final_state.get('error')}")
                # 실패 시 빈 데이터프레임 반환
                return pd.DataFrame(), ""
            
            # 메타데이터 설정 (대상 열 정보 보존)
            metadata = {"target_column": target_column}
            
            # 필터링된 데이터프레임 저장
            result_data_id = save_dataframe(filtered_df, metadata)
            logger.info(f"필터 그래프 실행 완료: {task_id}, 결과: {result_data_id}, 행 수: {len(filtered_df)}")
            
            return filtered_df, result_data_id
            
        except Exception as e:
            logger.error(f"필터 그래프 실행 중 오류 발생: {e}")
            # 오류 발생 시 빈 데이터프레임 반환
            return pd.DataFrame(), ""


# Create singleton instance
filter_runner = FilterRunner() 