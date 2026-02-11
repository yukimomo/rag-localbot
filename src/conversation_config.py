FOLLOW_UP_MARKERS = [
    "これ",
    "それ",
    "あれ",
    "この",
    "その",
    "あの",
    "それ",
    "それは",
    "どこ",
    "どれ",
    "どの",
    "何分",
    "いつ",
    "回数",
    "具体的",
    "詳細",
    "教えて",
    "もう少し",
    "続き",
    "さっき",
    "前の",
    "先の",
    "前述",
    "上の",
    "上記",
    "今の",
    "先ほど",
    "soreha",
    "kore",
    "are",
    "this",
    "that",
    "it",
    "there",
    "where",
    "when",
    "previous",
    "earlier",
    "above",
    "that",
    "where",
    "when",
]

INTENT_FEEDBACK = "FEEDBACK"
INTENT_CONTROL = "CONTROL"
INTENT_QUESTION = "QUESTION"
INTENT_META_QUESTION = "META_QUESTION"
INTENT_CHAT = "CHAT"

FEEDBACK_TERMS = [
    "いいね",
    "ok",
    "ありがとう",
    "なるほど",
    "了解",
    "助かった",
]

QUESTION_TERMS = [
    "?",
    "？",
    "何",
    "どこ",
    "回",
    "対処",
    "いつ",
    "どう",
    "どれ",
    "どの",
]

META_QUESTION_TERMS = [
    "別の質問",
    "他の質問",
    "質問していい",
    "別の質問でもいい",
    "別の質問していい",
    "使い方",
    "どう使う",
    "次どうする",
    "進め方",
    "リセット",
    "さっきの",
    "前の回答",
    "もう一回",
]

PRONOUN_MARKERS = [
    "それ",
    "それは",
    "あれ",
    "これ",
    "どれ",
    "上の",
    "前の",
]

REFERENCE_TERMS = [
    "同じ",
    "さっき",
    "前回",
    "先ほど",
    "先の",
]

GENERIC_TOPIC_TERMS = {
    "タイムライン",
    "フェーズ",
    "対処",
    "回数",
    "どこ",
    "それ",
}

TOPIC_VARIANT_RULES = [
    ("サーバー", "サーバ"),
    ("ユーザー", "ユーザ"),
    ("コンピューター", "コンピュータ"),
    ("システム", "システムズ"),
    ("インフラ", "インフラストラクチャ"),
    ("データベース", "DB"),
    ("アプリケーション", "アプリ"),
    ("マネージャー", "マネージャ"),
    ("オペレーター", "オペレータ"),
    ("セキュリティ", "セキュア"),
    ("バージョン", "ver"),
    ("バージョン", "version"),
    ("エラー", "ERR"),
    ("エラー", "error"),
    ("インシデント", "INC"),
    ("インシデント", "incident"),
    ("クロー", "クロウ"),
]

ASSISTANT_HEADER = "あなたは社内資料を根拠に回答するアシスタントです。\n"

BASE_EVIDENCE_POLICY = (
    "以下の根拠以外から推測しないでください。根拠内に質問対象が"
    "見つからない場合は『不明』と返し、必要な追加情報(別表記など)を提案してください。\n"
    "根拠に基づく旨を必ず明記し、回答内で参照した根拠(ファイル/シート/行)を引用してください。\n"
)

INFERENCE_POLICY = (
    "根拠不足の場合は推測として回答してよいが、必ず【推測】を付け、断定しないでください。\n"
)

CONFIRMATION_GUIDE_LINE = "確認すべき追加情報/検索キーワード候補を提示してください。"

BASE_PROMPT_HEADER = ASSISTANT_HEADER + BASE_EVIDENCE_POLICY + "\n"
INFERENCE_PROMPT_HEADER = (
    ASSISTANT_HEADER + INFERENCE_POLICY + CONFIRMATION_GUIDE_LINE + "\n\n"
)

SYSTEM_PROMPT_JA = "日本語で回答してください。"
SYSTEM_PROMPT_JA_BRIEF = "日本語で簡潔に回答してください。"
SYSTEM_PROMPT_NO_INFERENCE = "根拠以外から推測しないでください。"
SYSTEM_PROMPT_TIME2 = "time2はタイムライン時間です。"
SYSTEM_PROMPT_NO_ASSERT = "断定口調は禁止です。"

INFERENCE_BLOCK_FORMAT = (
    "出力は次の3ブロック構造で固定してください。\n"
    "(a) 【推測】で始まる回答\n"
    "(b) 推測の根拠(根拠documentsから引用)\n"
    "(c) 確認すべき点(3つ程度)\n"
)

BASE_SYSTEM_PROMPT = (
    SYSTEM_PROMPT_JA_BRIEF + SYSTEM_PROMPT_NO_INFERENCE + SYSTEM_PROMPT_TIME2
)

INFERENCE_SYSTEM_PROMPT = (
    SYSTEM_PROMPT_JA
    + "根拠不足なら推測として回答してよいが、必ず【推測】を付け、断定しないでください。"
    + CONFIRMATION_GUIDE_LINE
    + INFERENCE_BLOCK_FORMAT
    + SYSTEM_PROMPT_NO_ASSERT
)


def build_variants(text: str, rules: list[tuple[str, str]]) -> list[str]:
    variants = {text}
    for src, dst in rules:
        if src in text:
            variants.add(text.replace(src, dst))
        if dst in text:
            variants.add(text.replace(dst, src))
    return sorted(variants)
