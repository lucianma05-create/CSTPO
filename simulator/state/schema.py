"""用户认知状态数据结构（开发文档 §3、§4、§15）。"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

STRENGTH_MIN, STRENGTH_MAX = 0.0, 4.0
MAX_ITEMS = {"belief": 4, "desire": 3, "intention": 3}
EMOTION_CATEGORIES = [
    "neutral", "sadness", "anxiety", "frustration",
    "interest", "hope", "relief", "satisfaction", "anger",
]
POLARITIES = {"approach", "avoid"}
DEACTIVATE_THRESHOLD = 0.5   # 非核心节点强度低于该值则 deactivate（文档 §3.1）


@dataclass
class BDIItem:
    id: str
    type: str            # belief / desire / intention
    content: str         # 用户第一人称表达（文档 §3.1）
    strength: float      # [0,4]
    core: bool = True    # 非核心节点优先淘汰（文档 §14 数量限制）
    active: bool = True
    # Desire/Intention 的极性：approach=希望实现/去做，avoid=希望避免/不去做。
    # Belief 不使用该字段（命题真伪无所谓极性）。
    # 使文档 §15.1 的 gc_i（促进/阻碍）可程序化计算，避免依赖文本解析。
    polarity: str = "approach"


@dataclass
class Emotion:
    valence: float = 0.0   # [-1,1]
    arousal: float = 0.3   # [0,1]
    category: str = "neutral"


@dataclass
class Appraisal:
    """当轮中间变量，不作为长期状态（文档 §15）。"""
    goal_congruence: float = 0.0     # [-1,1]
    coping_potential: float = 0.0    # [-1,1]
    future_expectancy: float = 0.0   # [-1,1]


@dataclass
class CognitiveProfile:
    """Theta_u = (eta_R, tau_A, tau_R)，全部保存在程序侧（文档 §4）。"""
    eta_R: float = 0.6    # 加工深度倾向 [0,1]
    tau_A: float = 0.35   # 接受阈值，0 <= tau_A < tau_R <= 1
    tau_R: float = 0.70   # 拒绝阈值

    def __post_init__(self):
        assert 0.0 <= self.tau_A < self.tau_R <= 1.0, "要求 0 <= tau_A < tau_R <= 1"


@dataclass
class UserState:
    """完整用户状态：Persona + 习惯卡 + BDI + Emotion + 对话历史。"""
    persona: str = ""
    profile: CognitiveProfile = field(default_factory=CognitiveProfile)
    habit_card: str = ""
    beliefs: list[BDIItem] = field(default_factory=list)
    desires: list[BDIItem] = field(default_factory=list)
    intentions: list[BDIItem] = field(default_factory=list)
    emotion: Emotion = field(default_factory=Emotion)
    history: list[dict] = field(default_factory=list)  # [{role, text}]

    # ---- 便捷访问 ----
    def items(self, type_: str) -> list[BDIItem]:
        return {"belief": self.beliefs, "desire": self.desires, "intention": self.intentions}[type_]

    def find(self, item_id: str) -> BDIItem | None:
        for lst in (self.beliefs, self.desires, self.intentions):
            for it in lst:
                if it.id == item_id:
                    return it
        return None

    def bdi_dict(self) -> dict:
        return {
            "beliefs": [asdict(b) for b in self.beliefs],
            "desires": [asdict(d) for d in self.desires],
            "intentions": [asdict(i) for i in self.intentions],
        }

    def summary_text(self) -> str:
        """给 LLM 看的紧凑 BDI 摘要。"""
        lines = []
        for lst in (self.beliefs, self.desires, self.intentions):
            for it in lst:
                if not it.active:
                    continue
                tag = it.type
                if it.polarity == "avoid" and it.type != "belief":
                    tag += ",avoid"
                lines.append(f"- {it.id} [{tag}] ({it.strength:.1f}): {it.content}")
        return "\n".join(lines) or "(empty)"


@dataclass
class TurnLog:
    """单轮可审计日志（文档 §25：状态可解释、过程可审计）。"""
    turn: int
    assistant_reply: str
    mode: str
    route: str | None
    judgment: str | None
    p_central: float | None
    discrepancy: float | None
    target: str | None
    support_quality: str | None
    bdi_before: dict
    bdi_after: dict
    appraisal: dict
    emotion_before: dict
    emotion_after: dict
    reaction_plan: str | None
    user_reply: str
    proposed_bdi: dict = field(default_factory=dict)   # 引擎原始提案（审计用）
    user_done: bool = False       # 任务中立：用户是否想结束本次对话
    done_reason: str | None = None
    update_notes: list = field(default_factory=list)
