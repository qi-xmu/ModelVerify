"""终端表格显示工具库

提供 CJK 字符感知的字符串显示宽度计算和对齐函数，
用于在终端中正确对齐包含中文的表格。
"""

import unicodedata


def display_width(text: str) -> int:
    """计算字符串的终端显示宽度（CJK字符宽度=2, ASCII=1）"""
    w = 0
    for ch in text:
        ea = unicodedata.east_asian_width(ch)
        w += 2 if ea in ("W", "F") else 1
    return w


def pad_center(text: str, width: int, fill: str = " ") -> str:
    """按显示宽度居中对齐"""
    dw = display_width(text)
    if dw >= width:
        return text
    left = (width - dw) // 2
    right = width - dw - left
    return fill * left + text + fill * right


def pad_left(text: str, width: int, fill: str = " ") -> str:
    """按显示宽度左对齐（右补空格）"""
    dw = display_width(text)
    return text + fill * (width - dw) if dw < width else text


def pad_right(text: str, width: int, fill: str = " ") -> str:
    """按显示宽度右对齐（左补空格）"""
    dw = display_width(text)
    return fill * (width - dw) + text if dw < width else text
