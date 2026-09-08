from colorama import init, Fore, Back, Style
import sys

THOUGHT_COLOR = Fore.GREEN  # 思考过程
OBSERVATION_COLOR = Fore.YELLOW  # 工具返回结果
ROUND_COLOR = Fore.BLUE  # 轮次标识
RETURN_COLOR = Fore.CYAN  # 最终返回
CODE_COLOR = Fore.WHITE  # 生成的代码


def color_print(text, color=None, end="\n"):
    """带颜色的终端输出"""
    if color is not None:
        content = color + text + Style.RESET_ALL + end
    else:
        content = text + end
    sys.stdout.write(content)
    sys.stdout.flush()
