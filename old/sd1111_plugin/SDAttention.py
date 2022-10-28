from enum import Enum


class SDAttention(Enum):
    LDM = 0
    SPLIT_BASUJINDAL = 1
    SPLIT_INVOKE = 2
    SPLIT_DOGGETT = 3
    XFORMERS = 4