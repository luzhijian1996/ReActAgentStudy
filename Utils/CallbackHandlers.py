from typing import Optional, Union, Any, Dict
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk, LLMResult

from Utils.PrintUtils import *


class ColoredPrintHandler(BaseCallbackHandler):
    """自定义回调处理器，用于在控制台打印彩色输出"""
    
    def __init__(self, color: str):
        BaseCallbackHandler.__init__(self)
        self._color = color

    def on_llm_new_token(
            self,
            token: str,
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        """LLM 每生成一个 token 时，实时彩色打印"""
        color_print(token, self._color, end="")
        return token

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """LLM 生成完毕时换行"""
        color_print("\n", self._color, end="")
        return response

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """工具执行完毕时打印返回结果"""
        print()
        color_print("\n[Tool Return]", RETURN_COLOR)
        color_print(output, OBSERVATION_COLOR)
        return output

    @staticmethod
    def on_thought_start(index: int, **kwargs: Any) -> Any:
        """新一轮思考开始时打印轮次号"""
        color_print(f"\n[Thought: {index}]", ROUND_COLOR)
        return index

