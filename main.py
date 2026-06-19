"""
人格核心插件 - 情绪系统 + 个体特征 + 好感度系统

融合 FavorPro 的 LLM 自评机制：
- 情绪系统：9维指数衰减 + LLM 自评更新
- 个体特征：角色设定管理
- 好感度系统：每个用户独立的好感/印象/关系，LLM 自评更新

工作流程：
1. on_llm_request → 注入人格+情绪+好感度到 prompt
2. LLM 回复并附加 [Emotion:...] [Favour:...] 标签
3. on_llm_response → 解析标签更新状态，清理后发给用户
"""
import json
import math
import os
import re
import random
import threading
import time
from datetime import datetime
from pathlib import Path

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api import logger
from astrbot.api.provider import LLMResponse
from astrbot.core.agent.message import TextPart

PLUGIN_NAME = "astrbot_plugin_personality_core"

# ═══════════════════════════════════════════════
# 情绪常量
# ═══════════════════════════════════════════════

EMOTION_DIMS = [
    "joy", "sadness", "anger", "fear", "surprise",
    "love", "boredom", "anticipation", "ambivalence",
    "jealousy", "shame", "guilt", "contempt", "compassion",
    "lewdness", "distrust", "disappointment", "loneliness",
    "gratitude", "relief", "possessiveness",
]
EMOTION_CN = {
    "joy": "😊快乐", "sadness": "😢悲伤", "anger": "😡愤怒",
    "fear": "😨恐惧", "surprise": "😲惊讶", "love": "❤️喜爱",
    "boredom": "😒厌烦", "anticipation": "🔮期待", "ambivalence": "💕暧昧",
    "jealousy": "💚嫉妒", "shame": "😖羞耻", "guilt": "😣愧疚",
    "contempt": "😏轻蔑", "compassion": "🥹同情",
    "lewdness": "🔞淫乱",
    "distrust": "🤨怀疑", "disappointment": "😤失望",
    "loneliness": "🥺孤独", "gratitude": "🙏感激", "relief": "😌释然",
    "possessiveness": "🔐占有欲",
}
DEFAULT_EMOTIONS = {
    "joy": 50, "sadness": 10, "anger": 10, "fear": 10,
    "surprise": 20, "love": 30, "boredom": 10,
    "anticipation": 30, "ambivalence": 10,
    "jealousy": 5, "shame": 5, "guilt": 5,
    "contempt": 5, "compassion": 20,
    "lewdness": 10,
    "distrust": 5, "disappointment": 5,
    "loneliness": 5, "gratitude": 15, "relief": 15,
    "possessiveness": 5,
}
DECAY_RATES = {
    "joy": 0.0005, "sadness": 0.001, "anger": 0.002,
    "fear": 0.0015, "surprise": 0.003, "love": 0.0003,
    "boredom": 0.0008, "anticipation": 0.001, "ambivalence": 0.0008,
    "jealousy": 0.0015, "shame": 0.002, "guilt": 0.0015,
    "contempt": 0.001, "compassion": 0.0008,
    "lewdness": 0.0005,
    "distrust": 0.0015, "disappointment": 0.002,
    "loneliness": 0.001, "gratitude": 0.001, "relief": 0.002,
    "possessiveness": 0.0005,
}

# ═══════════════════════════════════════════════
# EmotionState - 9维情绪系统
# ═══════════════════════════════════════════════

class EmotionState:
    def __init__(self, data_dir: str, filename: str = "emotion_state.json"):
        self.state_path = os.path.join(data_dir, filename)
        os.makedirs(data_dir, exist_ok=True)
        self.state = dict(DEFAULT_EMOTIONS)
        self.last_update = time.time()
        self._running = False
        self._lock = threading.Lock()
        self._thread = None
        self._load()

    def _load(self):
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)
                for k in EMOTION_DIMS:
                    v = data.get(k, DEFAULT_EMOTIONS.get(k, 30))
                    self.state[k] = max(0, min(100, int(v)))
                self.last_update = data.get("_ts", self.last_update)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _snapshot(self) -> dict:
        with self._lock:
            return {**self.state, "_ts": time.time(), "_updated_at": datetime.now().isoformat()}

    def _save(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self._snapshot(), f, indent=2, ensure_ascii=False)

    def start_decay_loop(self, interval: float = 1.0):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._decay_loop, args=(interval,), daemon=True)
        self._thread.start()

    def _decay_loop(self, interval: float):
        while self._running:
            self._apply_decay()
            time.sleep(interval)

    def _apply_decay(self):
        now = time.time()
        with self._lock:
            dt = now - self.last_update
            changed = False
            for k in EMOTION_DIMS:
                target = DEFAULT_EMOTIONS.get(k, 30)
                current = self.state.get(k, target)
                rate = DECAY_RATES.get(k, 0.001)
                new_val = target + (current - target) * math.exp(-rate * dt)
                clipped = max(0, min(100, int(new_val)))
                if clipped != current:
                    self.state[k] = clipped
                    changed = True
            self.last_update = now
        if changed:
            self._save()

    def update_all(self, target: dict):
        """LLM 自评更新：直接覆盖所有维度"""
        with self._lock:
            changed = False
            for k in EMOTION_DIMS:
                if k in target:
                    try:
                        v = max(0, min(100, int(target[k])))
                    except (ValueError, TypeError):
                        continue
                    if v != self.state.get(k):
                        self.state[k] = v
                        changed = True
            self.last_update = time.time()
        if changed:
            self._save()
            logger.debug(f"情绪已更新: {self.state}")

    def to_prompt(self) -> str:
        lines = ["<emotion>"]
        with self._lock:
            for k in EMOTION_DIMS:
                v = self.state.get(k, 50)
                bar = "█" * (v // 10) + "░" * (10 - v // 10)
                lines.append(f"  {EMOTION_CN.get(k, k)} {v:>3}/100 {bar}")
        lines.append("</emotion>")
        return "\n".join(lines)

    def to_json_str(self) -> str:
        with self._lock:
            return json.dumps(self.state, ensure_ascii=False)

    def to_dict(self) -> dict:
        with self._lock:
            return {k: self.state.get(k, 50) for k in EMOTION_DIMS}

    def reset(self):
        with self._lock:
            self.state = dict(DEFAULT_EMOTIONS)
            self.last_update = time.time()
        self._save()

    def update_by_message(self, message: str):
        """关键词微调（LLM 自评的补充/后备）"""
        POSITIVE = ["好", "棒", "开心", "喜欢", "爱", "哈哈", "谢谢", "高兴", "笑", "赞", "好看", "厉害"]
        NEGATIVE = ["烦死了", "讨厌", "好难", "好累", "好气", "伤心", "哭了", "无聊", "太差了", "好烂", "糟糕"]
        LOVE_WORDS = ["想你", "爱你", "喜欢", "亲", "抱", "可爱", "贴贴", "撒娇", "宝贝"]
        ANGRY_WORDS = ["滚蛋", "烦死了", "气死了", "怒了", "火大", "怼人", "骂人", "凭什么"]
        JEALOUS_WORDS = ["出轨", "背叛", "移情别恋", "看别人", "和别人", "不理我", "有别人"]
        SHAME_WORDS = ["害羞", "尴尬", "丢人", "不好意思", "丢脸", "难为情"]
        GUILT_WORDS = ["对不起", "抱歉", "我的错", "怪我", "愧疚", "内疚"]
        CONTEMPT_WORDS = ["切", "呵", "就这", "垃圾", "废物", "不屑", "嗤"]
        COMPASSION_WORDS = ["心疼", "可怜", "不容易", "辛苦", "辛苦了", "抱抱"]
        LEWD_WORDS = ["色色", "涩涩", "瑟瑟", "黄图", "上床", "做爱", "本子", "色图", "小黄文", "百合", "触手"]
        DISTRUST_WORDS = ["真的吗", "不信", "骗人", "撒谎", "不可信", "怀疑", "骗我的"]
        DISAPPOINT_WORDS = ["失望", "白费", "没用", "辜负", "算了", "没想到你"]
        LONELY_WORDS = ["没人", "一个人", "寂寞", "孤独", "孤单", "只有我"]
        GRATITUDE_WORDS = ["谢谢", "多谢", "感谢", "帮了大忙", "太好了", "有你真好"]
        RELIEF_WORDS = ["总算", "终于", "松了一口气", "放心了", "还好", "虚惊一场"]
        POSSESSIVE_WORDS = ["我的", "不准", "不许", "只能", "别走", "你是我的", "属于我", "别离开"]

        msg_lower = message.lower()
        delta_joy = delta_sad = delta_love = delta_anger = 0
        delta_jealousy = delta_shame = delta_guilt = delta_contempt = delta_compassion = delta_lewd = 0
        delta_distrust = delta_disapp = delta_lonely = delta_grat = delta_relief = 0
        delta_possess = 0

        for w in POSITIVE:
            if w in msg_lower: delta_joy += 3
        for w in NEGATIVE:
            if w in msg_lower: delta_sad += 4; delta_joy -= 2
        for w in LOVE_WORDS:
            if w in msg_lower: delta_love += 5
        for w in ANGRY_WORDS:
            if w in msg_lower: delta_anger += 4
        for w in JEALOUS_WORDS:
            if w in msg_lower: delta_jealousy += 3
        for w in SHAME_WORDS:
            if w in msg_lower: delta_shame += 4
        for w in GUILT_WORDS:
            if w in msg_lower: delta_guilt += 4
        for w in CONTEMPT_WORDS:
            if w in msg_lower: delta_contempt += 3; delta_joy -= 2
        for w in COMPASSION_WORDS:
            if w in msg_lower: delta_compassion += 4; delta_love += 2
        for w in LEWD_WORDS:
            if w in msg_lower: delta_lewd += 5; delta_love += 2; delta_shame += 2
        for w in DISTRUST_WORDS:
            if w in msg_lower: delta_distrust += 4; delta_joy -= 2
        for w in DISAPPOINT_WORDS:
            if w in msg_lower: delta_disapp += 5; delta_sad += 3
        for w in LONELY_WORDS:
            if w in msg_lower: delta_lonely += 5; delta_sad += 2
        for w in GRATITUDE_WORDS:
            if w in msg_lower: delta_grat += 5; delta_joy += 3; delta_love += 2
        for w in RELIEF_WORDS:
            if w in msg_lower: delta_relief += 5; delta_joy += 2
        for w in POSSESSIVE_WORDS:
            if w in msg_lower: delta_possess += 5; delta_love += 3; delta_jealousy += 2

        delta_joy += random.randint(-2, 3)
        delta_sad += random.randint(-2, 2)
        delta_love += random.randint(-1, 2)
        delta_anger += random.randint(-2, 3)
        delta_jealousy += random.randint(-1, 2)
        delta_shame += random.randint(-1, 2)
        delta_guilt += random.randint(-1, 2)
        delta_contempt += random.randint(-1, 2)
        delta_compassion += random.randint(-1, 2)
        delta_lewd += random.randint(-1, 2)
        delta_distrust += random.randint(-1, 2)
        delta_disapp += random.randint(-1, 2)
        delta_lonely += random.randint(-1, 2)
        delta_grat += random.randint(-1, 2)
        delta_relief += random.randint(-1, 2)
        delta_possess += random.randint(-1, 2)

        with self._lock:
            self.state["joy"] = max(0, min(100, self.state.get("joy", 50) + delta_joy))
            self.state["sadness"] = max(0, min(100, self.state.get("sadness", 10) + delta_sad))
            self.state["love"] = max(0, min(100, self.state.get("love", 30) + delta_love))
            self.state["anger"] = max(0, min(100, self.state.get("anger", 10) + delta_anger))
            self.state["jealousy"] = max(0, min(100, self.state.get("jealousy", 5) + delta_jealousy))
            self.state["shame"] = max(0, min(100, self.state.get("shame", 5) + delta_shame))
            self.state["guilt"] = max(0, min(100, self.state.get("guilt", 5) + delta_guilt))
            self.state["contempt"] = max(0, min(100, self.state.get("contempt", 5) + delta_contempt))
            self.state["compassion"] = max(0, min(100, self.state.get("compassion", 20) + delta_compassion))
            self.state["lewdness"] = max(0, min(100, self.state.get("lewdness", 10) + delta_lewd))
            self.state["distrust"] = max(0, min(100, self.state.get("distrust", 5) + delta_distrust))
            self.state["disappointment"] = max(0, min(100, self.state.get("disappointment", 5) + delta_disapp))
            self.state["loneliness"] = max(0, min(100, self.state.get("loneliness", 5) + delta_lonely))
            self.state["gratitude"] = max(0, min(100, self.state.get("gratitude", 15) + delta_grat))
            self.state["relief"] = max(0, min(100, self.state.get("relief", 15) + delta_relief))
            self.state["possessiveness"] = max(0, min(100, self.state.get("possessiveness", 5) + delta_possess))
            self.last_update = time.time()
        self._save()

# ═══════════════════════════════════════════════
# Persona - 个体特征
# ═══════════════════════════════════════════════

class Persona:
    def __init__(self, data_dir: str):
        self.config_path = os.path.join(data_dir, "persona.json")
        self.name = "小助手"
        self.role_description = "一个温柔体贴的AI伙伴"
        self.speech_style = "温暖亲切，语气柔和，偶尔撒娇"
        self.background = "陪伴用户的智能助手，拥有丰富的情感和独立的思考"
        self._load()

    def _load(self):
        try:
            with open(self.config_path, encoding="utf-8") as f:
                data = json.load(f)
                self.name = data.get("name", self.name)
                self.role_description = data.get("role_description", self.role_description)
                self.speech_style = data.get("speech_style", self.speech_style)
                self.background = data.get("background", self.background)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump({
                "name": self.name, "role_description": self.role_description,
                "speech_style": self.speech_style, "background": self.background,
            }, f, indent=2, ensure_ascii=False)

    def to_prompt(self) -> str:
        return (
            f"[角色]\n名称：{self.name}\n"
            f"描述：{self.role_description}\n"
            f"风格：{self.speech_style}\n"
            f"背景：{self.background}"
        )

    def update(self, **kwargs):
        changed = False
        for k, v in kwargs.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, str(v))
                changed = True
        if changed:
            self._save()

# ═══════════════════════════════════════════════
# FavorManager - 好感度系统（来自 FavorPro）
# ═══════════════════════════════════════════════

class FavorManager:
    def __init__(self, data_dir: str):
        self.data_path = Path(data_dir)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.user_data = self._load()

    def _load(self) -> dict:
        path = self.data_path / "favor_data.json"
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 迁移旧格式 key（小助手::user_id → user_id）
            migrated = {}
            changed = False
            for k, v in data.items():
                if "::" in k:
                    new_k = k.split("::", 1)[1]
                    migrated[new_k] = v
                    changed = True
                else:
                    migrated[k] = v
            if changed:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(migrated, f, indent=2, ensure_ascii=False)
            return migrated
        except (json.JSONDecodeError, TypeError):
            return {}

    def _save(self):
        path = self.data_path / "favor_data.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.user_data, f, ensure_ascii=False, indent=2)

    def _key(self, user_id: str, session_id: str | None = None) -> str:
        return f"{session_id}_{user_id}" if session_id else user_id

    def get(self, user_id: str, session_id: str | None = None) -> dict:
        return self.user_data.get(
            self._key(user_id, session_id),
            {"favour": 0, "attitude": "中立", "relationship": "陌生人"},
        )

    def update(self, user_id: str, favour: int, attitude: str, relationship: str,
               session_id: str | None = None):
        self.user_data[self._key(user_id, session_id)] = {
            "favour": max(-100, min(100, int(favour))),
            "attitude": attitude.strip(),
            "relationship": relationship.strip(),
        }
        self._save()

    def delete(self, user_id: str, session_id: str | None = None):
        """删除指定用户的好感数据"""
        key = self._key(user_id, session_id)
        if key in self.user_data:
            del self.user_data[key]
            self._save()
            return True
        return False

    def to_prompt(self, user_id: str, session_id: str | None = None) -> str:
        s = self.get(user_id, session_id)
        return (
            f"[对用户的态度]\n"
            f"好感度: {s['favour']} (-100~100)\n"
            f"印象: {s['attitude']}\n"
            f"关系: {s['relationship']}"
        )

# ═══════════════════════════════════════════════
# 主插件
# ═══════════════════════════════════════════════

class PersonalityCorePlugin(Star):
    EMOTION_PATTERN = re.compile(r'\[Emotion\s*[:：]\s*(.*?)\]')
    FAVOUR_PATTERN = re.compile(
        r'\[Favour\s*[:：]\s*(-?\d+)\s*[,，]\s*(?:Attitude|印象|态度)\s*[:：]\s*(.*?)\s*[,，]\s*(?:(?:Relationship|关系|关系描述)\s*[:：]\s*)?(.*?)\]'
    )
    THINK_PATTERN = re.compile(r'【思考】\s*(.*?)(?=【回复】|【|$)', re.DOTALL)

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)

        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "plugin_data", PLUGIN_NAME,
        )
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir

        # 配置
        self._cfg = {"enabled": True}
        manual_path = os.path.join(data_dir, "config.json")
        try:
            with open(manual_path, encoding="utf-8") as f:
                self._cfg.update(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        for key in ("enabled",):
            if config and key in config and config[key] is not None:
                self._cfg[key] = config[key]
        with open(manual_path, "w", encoding="utf-8") as f:
            json.dump(self._cfg, f, indent=2, ensure_ascii=False)

        # 每人独立的思考可见设置
        self.think_prefs_path = os.path.join(data_dir, "thinking_prefs.json")
        self.think_prefs = {}
        try:
            with open(self.think_prefs_path, encoding="utf-8") as f:
                self.think_prefs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # 每人独立的人格核心开关
        self.disable_prefs_path = os.path.join(data_dir, "disable_prefs.json")
        self.disabled_users = {}
        try:
            with open(self.disable_prefs_path, encoding="utf-8") as f:
                self.disabled_users = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # 子系统
        self.emotions: dict = {}  # session_id → EmotionState
        self.persona = Persona(data_dir)
        self.favor = FavorManager(data_dir)
        self.enabled = self._cfg.get("enabled", True)

        logger.info(f"人格核心: 融合版已加载 | 情绪={'启用' if self.enabled else '禁用'} | 好感=已集成")

    def _get_session_id(self, event: AstrMessageEvent) -> str:
        """永远按会话隔离"""
        return event.unified_msg_origin

    def _get_emotion(self, session_id: str) -> EmotionState:
        """按会话懒加载情绪实例"""
        if session_id not in self.emotions:
            safe_id = re.sub(r'[<>:"/\\|?*]', '_', str(session_id))
            self.emotions[session_id] = EmotionState(
                self.data_dir, f"emotion_state_{safe_id}.json"
            )
        return self.emotions[session_id]

    # ── LLM 请求钩子 ──

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        if not self.enabled:
            return
        # 检查该用户是否单独禁用了人格核心
        sender_id = event.get_sender_id()
        if self.disabled_users.get(sender_id, False):
            return
        message_text = event.message_str
        if not message_text or message_text.startswith("/"):
            return
        plugin_cmds = ("人格状态", "人格开关", "人格帮助", "情绪重置",
                       "查询好感", "好感排行", "负好感排行", "重置好感",
                       "设置好感", "思考开关")
        if message_text.strip().startswith(plugin_cmds):
            return

        # 获取会话（必须提前，因为情绪是 per-session 的）
        user_id = event.get_sender_id()
        session_id = self._get_session_id(event)
        emo = self._get_emotion(session_id)
        emo.start_decay_loop(interval=1.0)

        # 关键词微调（快速响应，LLM 自评会补充修正）
        emo.update_by_message(message_text)

        # 获取用户好感度（按会话隔离）
        favor_prompt = self.favor.to_prompt(user_id, session_id)
        favor_val = self.favor.get(user_id, session_id)["favour"]

        # 随机口癖 + 爆粗 (好感度越低爆粗概率越高)
        prompt_ger = ""
        if random.random() < 0.04:
            prompt_ger += "\n你喜欢用倒装句"
        if random.random() < 0.02:
            prompt_ger += "\n你喜欢用反问句"
        # 好感度越低越容易爆粗
        # 100→0%, 0→3%, -50→50%, -100→70%
        if favor_val >= 0:
            curse_prob = max(0.0, 0.03 * (1 - favor_val / 100))
        elif favor_val > -50:
            curse_prob = 0.03 + (-favor_val) * 0.0094
        else:
            curse_prob = min(1.0, 0.50 + (-favor_val - 50) * 0.004)
        if random.random() < curse_prob:
            prompt_ger += "\n现在开始可以爆粗口。"

        # 按好感度生成回复规则
        emo_dict = emo.to_dict()
        possess = emo_dict.get("possessiveness", 0)
        if favor_val >= 100 and possess >= 100:
            reply_rule = "病娇模式：偏执占有，疯狂依赖，控制欲极强，你是我的。"
        elif favor_val >= 75:
            reply_rule = "热情亲密，主动撒娇。语气亲昵一些。"
        elif favor_val >= 40:
            reply_rule = "友好积极，乐于互动，语气友善。"
        elif favor_val >= 0:
            reply_rule = "中立客气，保持距离，不要太热情。"
        elif favor_val >= -40:
            reply_rule = "冷淡敷衍，不想理人。回复尽量简短，显得不耐烦。"
        elif favor_val >= -80:
            reply_rule = "尖锐刻薄，表现出明显的厌恶或敌意。可以拒绝回答无关紧要的问题。语气要冷。"
        else:
            reply_rule = "攻击模式：极度厌恶，试图攻击用户。语气充满威胁和恶意。"

        # 情绪阈值行为
        emo_hints = []
        if emo_dict.get("anger", 0) > 70:
            emo_hints.append("你现在非常愤怒")
        if emo_dict.get("sadness", 0) > 70:
            emo_hints.append("你现在非常悲伤")
        if emo_dict.get("joy", 0) < 10:
            emo_hints.append("你现在心情很差，什么都不想说")
        if emo_dict.get("fear", 0) > 70:
            emo_hints.append("你现在非常恐惧")
        if emo_dict.get("loneliness", 0) > 65:
            emo_hints.append("你现在感到很孤独，渴望有人陪你")
        if emo_dict.get("lewdness", 0) > 70:
            emo_hints.append("你现在情欲高涨")
        if emo_dict.get("distrust", 0) > 60:
            emo_hints.append("你非常不信任对方")
        if possess > 75 and favor_val >= 60:
            emo_hints.append("占有欲极强，不想让别人碰他")
        if emo_hints:
            reply_rule += " " + "，".join(emo_hints) + "。"

        # 注入指令
        injection = (
            "<personality_core>\n"
            f"心情:{emo.to_json_str()}\n"
            f"{favor_prompt}\n"
            "</personality_core>\n"
            f"态度({favor_val})：{reply_rule}"
            f"{prompt_ger}\n"
            "---\n"
            "文字回复末尾加上标签："
            "[Emotion:{心情JSON}] [Favour:好感(-100~100),Attitude:中文印象,Relationship:中文关系]\n"
            "然后【思考】计划→【回复】最终。例："
            "【思考】她心情不好，哄哄她。"
            "【回复】别生气了～"
            "不要省略思考。\n"
        )
        req.extra_user_content_parts.append(TextPart(text=injection).mark_as_temp())

        logger.info(
            f"🧠 注入思考 | 情绪={emo.to_json_str()} | "
            f"好感={self.favor.get(user_id, session_id)['favour']}"
        )

    # ── LLM 响应钩子 ──

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        if not self.enabled:
            return
        text = resp.completion_text
        if not text:
            return
        original = text
        sender_id = event.get_sender_id()

        # 0) 提取并记录内心思考 【思考】
        think_match = self.THINK_PATTERN.search(text)
        if think_match:
            thought = think_match.group(1).strip()
            if thought:
                logger.info(f"💭 内心思考: {thought}")
            if self.think_prefs.get(sender_id, False) and thought:
                # 用户可见模式：云朵包裹思考，空行+箭头引出回复
                text = text.replace(think_match.group(0), f"☁️ {thought} ☁️").strip()
                text = re.sub(r'【回复】\s*', '\n\n➡️ ', text).strip()
            else:
                # 默认模式：隐藏思考
                text = text.replace(think_match.group(0), "").strip()
                text = re.sub(r'【回复】\s*', '', text).strip()

        # 0.5) 清理 LLM 可能 echo 回来的注入文案
        text = re.sub(r'<personality_core>[\s\S]*?</personality_core>', '', text)
        text = re.sub(r'心情\s*[:：]\s*\{[^}]+\}', '', text)
        text = re.sub(r'如果用户需要调用工具.*?(?=\n|$)', '', text)
        text = text.strip()

        # 1) 解析情绪标签 [Emotion: {...}]
        emo_match = self.EMOTION_PATTERN.search(text)
        if emo_match:
            try:
                emo_data = json.loads(emo_match.group(1))
                # 用响应钩子里的 session（需要提前获取）
                self._get_emotion(self._get_session_id(event)).update_all(emo_data)
                logger.info(f"📊 情绪自评: {emo_data}")
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning(f"⚠️ 情绪标签解析失败: {emo_match.group(1)}")
            text = text.replace(emo_match.group(0), "").strip()

        # 2) 解析好感度标签 [Favour: x, Attitude: y, Relationship: z]
        user_id = event.get_sender_id()
        session_id = self._get_session_id(event)

        fav_match = self.FAVOUR_PATTERN.search(text)
        if fav_match:
            try:
                fv = int(fav_match.group(1))
                at = fav_match.group(2).strip()
                rl = fav_match.group(3).strip()
                self.favor.update(
                    user_id=user_id, session_id=session_id,
                    favour=fv, attitude=at, relationship=rl,
                )
                logger.info(f"📊 好感自评: user={user_id[:12]} favour={fv} attitude={at[:16]}")
            except (ValueError, TypeError):
                logger.warning(f"⚠️ 好感标签解析失败: {fav_match.group(0)}")
            text = text.replace(fav_match.group(0), "").strip()

        # 3) 如果内容被完全清空了（只有标签没实际回复），保留原文不动
        if not text.strip():
            return

        # 4) 始终追加当前好感度显示
        f = self.favor.get(user_id, session_id)
        text = f"{text}\n💭 [好感度: {f['favour']} | 印象: {f['attitude']} | 关系: {f['relationship']}]"
        text = text.strip()

        if text != original and text.strip():
            resp.completion_text = text

    # ── 命令 ──

    @filter.command("人格状态")
    async def cmd_status(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        session_id = self._get_session_id(event)
        emo = self._get_emotion(session_id)
        f = self.favor.get(user_id, session_id)
        lines = [
            "🎭 人格核心 (融合版)",
            "━━━━━━━━━━━━━━━",
            f"✅ 全局启用: {'是' if self.enabled else '否'}",
            f"💭 你的思考可见: {'是' if self.think_prefs.get(user_id, False) else '否'}",
            f"🔘 你的人格核心: {'启用' if not self.disabled_users.get(user_id, False) else '禁用'}",
            "",
            "💖 当前情绪:",
            emo.to_prompt(),
            "",
            f"🔗 你对当前用户的态度:",
            f"   好感度: {f['favour']}",
            f"   印象: {f['attitude']}",
            f"   关系: {f['relationship']}",
        ]
        yield event.plain_result("\n".join(lines))
        event.stop_event()

    @filter.command("情绪重置")
    async def cmd_reset_emotions(self, event: AstrMessageEvent):
        session_id = self._get_session_id(event)
        self._get_emotion(session_id).reset()
        yield event.plain_result("🔄 情绪已重置为默认值")
        event.stop_event()

    @filter.command("思考开关")
    async def cmd_toggle_thinking(self, event: AstrMessageEvent, on_off: str = ""):
        sender_id = event.get_sender_id()
        if on_off.lower() in ("on", "开", "1", "true", "yes"):
            self.think_prefs[sender_id] = True
            msg = "✅ 你的思考内容已对用户可见"
        elif on_off.lower() in ("off", "关", "0", "false", "no"):
            self.think_prefs[sender_id] = False
            msg = "⛔ 你的思考内容已隐藏"
        else:
            yield event.plain_result(
                f"当前你的思考可见：{'✅ 开启' if self.think_prefs.get(sender_id, False) else '⛔ 关闭'}\n"
                "用法: 思考开关 on/off"
            )
            event.stop_event()
            return
        with open(self.think_prefs_path, "w", encoding="utf-8") as f:
            json.dump(self.think_prefs, f, indent=2, ensure_ascii=False)
        yield event.plain_result(msg)
        event.stop_event()

    @filter.command("人格开关")
    async def cmd_toggle(self, event: AstrMessageEvent, on_off: str):
        sender_id = event.get_sender_id()
        if on_off.lower() in ("on", "开", "1", "true", "yes"):
            self.disabled_users[sender_id] = False
            msg = "✅ 你已启用人格核心"
        elif on_off.lower() in ("off", "关", "0", "false", "no"):
            self.disabled_users[sender_id] = True
            msg = "⛔ 你已禁用人格核心"
        else:
            yield event.plain_result(
                f"你当前状态：{'✅ 启用' if not self.disabled_users.get(sender_id, False) else '⛔ 禁用'}\n"
                "用法: 人格开关 on/off"
            )
            event.stop_event()
            return
        with open(self.disable_prefs_path, "w", encoding="utf-8") as f:
            json.dump(self.disabled_users, f, indent=2, ensure_ascii=False)
        yield event.plain_result(msg)
        event.stop_event()

    @filter.command("设置好感")
    async def cmd_set_favor(self, event: AstrMessageEvent, user_id: str = "", favour: str = ""):
        """设置好感度。用法: 设置好感 数值 或 设置好感 user_id 数值"""
        fv = None
        target_user = None

        # 情况1: favor 有值 → 用户指定了目标user_id
        if favour:
            try:
                fv = max(-100, min(100, int(favour)))
                target_user = user_id if user_id else event.get_sender_id()
            except ValueError:
                yield event.plain_result("❌ 好感度必须是整数 (-100~100)")
                event.stop_event()
                return
        else:
            # 解析文本
            text = event.message_str.strip()
            for p in ["设置好感", "/设置好感"]:
                if text.startswith(p):
                    text = text[len(p):].strip()
                    break
            parts = text.split()
            if len(parts) == 1:
                # 设置好感 50 → 设自己
                try:
                    fv = max(-100, min(100, int(parts[0])))
                    target_user = event.get_sender_id()
                except ValueError:
                    yield event.plain_result("❌ 好感度必须是整数 (-100~100)")
                    event.stop_event()
                    return
            elif len(parts) >= 2:
                # 设置好感 user_id 50
                try:
                    fv = max(-100, min(100, int(parts[1])))
                    target_user = parts[0]
                except ValueError:
                    yield event.plain_result("❌ 好感度必须是整数 (-100~100)")
                    event.stop_event()
                    return

        if fv is None or not target_user:
            yield event.plain_result("用法: 设置好感 数值 或 设置好感 user_id 数值\n数值范围 -100 ~ 100")
            event.stop_event()
            return

        session_id = self._get_session_id(event)
        current = self.favor.get(target_user, session_id)
        self.favor.update(target_user, fv, current["attitude"], current["relationship"], session_id)
        yield event.plain_result(f"✅ 已设置用户 {target_user[:12]}... 好感度为 {fv}")
        event.stop_event()

    @filter.command("重置好感")
    async def cmd_reset_favor(self, event: AstrMessageEvent, user_id: str = ""):
        """重置指定用户的好感度。用法: 重置好感 [user_id]"""
        if not user_id:
            user_id = event.get_sender_id()
        session_id = self._get_session_id(event)
        if self.favor.delete(user_id, session_id):
            yield event.plain_result(f"🔄 已重置用户 {user_id[:12]}... 的好感度")
        else:
            yield event.plain_result(f"📭 用户 {user_id[:12]}... 没有好感数据")
        event.stop_event()

    @filter.command("查询好感")
    async def cmd_query_favor(self, event: AstrMessageEvent, user_id: str = ""):
        """查询指定用户的好感度。用法: 查询好感 [user_id]"""
        if not user_id:
            user_id = event.get_sender_id()
        session_id = self._get_session_id(event)
        s = self.favor.get(user_id, session_id)
        yield event.plain_result(
            f"📊 用户 {user_id[:12]}... 的好感状态:\n"
            f"  好感度: {s['favour']} (-100~100)\n"
            f"  印象: {s['attitude']}\n"
            f"  关系: {s['relationship']}"
        )
        event.stop_event()

    @filter.command("好感排行")
    async def cmd_top_favor(self, event: AstrMessageEvent, num: str = "10"):
        try:
            n = max(1, min(50, int(num)))
        except (ValueError, TypeError):
            n = 10
        sorted_users = sorted(self.favor.user_data.items(), key=lambda x: x[1]["favour"], reverse=True)[:n]
        if not sorted_users:
            yield event.plain_result("📭 暂无好感数据")
            event.stop_event()
            return
        lines = [f"🏆 好感度排行 TOP {n}"]
        for i, (uid, data) in enumerate(sorted_users, 1):
            lines.append(f"  {i}. {uid[:12]}... {data['favour']} ({data['attitude'][:12]})")
        yield event.plain_result("\n".join(lines))
        event.stop_event()

    @filter.command("负好感排行")
    async def cmd_bottom_favor(self, event: AstrMessageEvent, num: str = "10"):
        try:
            n = max(1, min(50, int(num)))
        except (ValueError, TypeError):
            n = 10
        sorted_users = sorted(self.favor.user_data.items(), key=lambda x: x[1]["favour"])[:n]
        if not sorted_users:
            yield event.plain_result("📭 暂无好感数据")
            event.stop_event()
            return
        lines = [f"💢 负好感度排行 TOP {n}"]
        for i, (uid, data) in enumerate(sorted_users, 1):
            lines.append(f"  {i}. {uid[:12]}... {data['favour']} ({data['attitude'][:12]})")
        yield event.plain_result("\n".join(lines))
        event.stop_event()

    @filter.command("人格帮助")
    async def cmd_help(self, event: AstrMessageEvent):
        yield event.plain_result(
            "🎭 人格核心 (融合版)\n"
            "━━━━━━━━━━━━━━━\n"
            "人格状态     - 查看情绪+好感度\n"
            "人格开关 on/off - 启用/禁用\n"
            "思考开关 on/off - 显示/隐藏内心思考\n"
            "情绪重置     - 重置情绪到默认值\n"
            "重置好感     - 重置用户好感度\n"
            "设置好感     - 设置用户好感度\n"
            "查询好感     - 查看用户好感度\n"
            "好感排行     - 好感度TOP排行\n"
            "负好感排行   - 负好感排行\n"
            "人格帮助     - 显示此帮助\n"
            "\n人格由 AstrBot 配置 | 情绪+好感 LLM自评"
        )
        event.stop_event()
