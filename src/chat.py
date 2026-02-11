import argparse
import logging
import math
import re
import unicodedata
from typing import List, Tuple

import chromadb
import ollama
from chromadb.utils import embedding_functions

from src.conversation_config import BASE_PROMPT_HEADER, BASE_SYSTEM_PROMPT, FEEDBACK_TERMS, FOLLOW_UP_MARKERS, GENERIC_TOPIC_TERMS, INFERENCE_PROMPT_HEADER, INFERENCE_SYSTEM_PROMPT, INTENT_CHAT, INTENT_CONTROL, INTENT_FEEDBACK, INTENT_META_QUESTION, INTENT_QUESTION, META_QUESTION_TERMS, PRONOUN_MARKERS, QUESTION_TERMS, REFERENCE_TERMS, TOPIC_VARIANT_RULES, build_variants


LOGGER = logging.getLogger(__name__)



def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Chat with Excel chunks in Chroma")
	parser.add_argument("--db", required=True, help="Chroma DB directory")
	parser.add_argument("--model", default="gemma3:12b", help="Ollama model name")
	parser.add_argument("--topk", type=int, default=5, help="Top-K results to retrieve")
	parser.add_argument(
		"--memory-turns",
		type=int,
		default=3,
		help="Number of recent turns to keep in memory",
	)
	parser.add_argument(
		"--distance-threshold",
		type=float,
		default=0.75,
		help="Distance threshold for filtering results",
	)
	parser.add_argument(
		"--show-evidence",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Show evidence document snippets",
	)
	parser.add_argument(
		"--allow-inference",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Allow inference mode when evidence is insufficient",
	)
	return parser.parse_args()


def format_score(distance: float) -> str:
	if distance is None:
		return "N/A"
	return f"{distance:.4f}"


def extract_keyword(question: str) -> str:
	katakana_matches = re.findall(r"[\u30a0-\u30ff]+", question)
	if katakana_matches:
		return max(katakana_matches, key=len)

	word_matches = re.findall(r"[A-Za-z0-9\u3040-\u30ff\u4e00-\u9fff]+", question)
	if word_matches:
		return max(word_matches, key=len)

	return question.strip()


def extract_topic_candidate(question: str) -> str:
	# Prefer longest katakana term as topic candidate.
	katakana_matches = re.findall(r"[\u30a0-\u30ff]+", question)
	if katakana_matches:
		return max(katakana_matches, key=len)

	return ""


def is_follow_up(question: str) -> bool:
	if any(marker in question for marker in FOLLOW_UP_MARKERS):
		return True

	short_question = len(question) <= 12
	weak_noun = extract_topic_candidate(question) == ""
	return short_question and weak_noun


def has_pronoun_ref(question: str) -> bool:
	return any(marker in question for marker in PRONOUN_MARKERS)


def has_reference_term(question: str) -> bool:
	return any(marker in question for marker in REFERENCE_TERMS)


def classify_follow_up(question: str, current_topic: str) -> Tuple[bool, str, bool]:
	pronoun_ref = has_pronoun_ref(question)
	ref_term = has_reference_term(question)
	if not current_topic:
		return False, "", pronoun_ref
	if pronoun_ref:
		return True, "pronoun_ref", pronoun_ref
	if ref_term:
		return True, "reference_term", pronoun_ref
	return False, "", pronoun_ref


def make_topic_variants(topic: str) -> List[str]:
	if not topic:
		return []
	return build_variants(topic, TOPIC_VARIANT_RULES)


def match_topic_in_doc(topic: str, doc: str) -> bool:
	variants = make_topic_variants(topic)
	if not variants:
		return False
	normalized_doc = normalize_for_match(doc)
	return any(normalize_for_match(var) in normalized_doc for var in variants)


def extract_time_candidates(
	documents: List[str],
	metadatas: List[dict],
) -> List[Tuple[str, dict]]:
	candidates: List[Tuple[str, dict]] = []
	pattern = re.compile(r"time2=\d{2}:\d{2}:\d{2}")
	for doc, meta in zip(documents, metadatas):
		for match in pattern.findall(doc):
			time_value = match.split("=", maxsplit=1)[1]
			candidates.append((time_value, meta))
	return candidates


def normalize_for_match(text: str) -> str:
	# Normalize for simple matching: NFKC + strip separators/punctuations.
	cleaned = unicodedata.normalize("NFKC", str(text))
	cleaned = re.sub(
		r'[\s・･\-_/／、。,．，:：;；!！?？\(\)\[\]{}＜＞<>「」『』"\'’‘`~〜ｰー]+',
		"",
		cleaned,
	)
	return cleaned.lower()


def make_snippet(text: str, limit: int = 200) -> str:
	cleaned = " ".join(str(text).split())
	if len(cleaned) <= limit:
		return cleaned
	return f"{cleaned[:limit].rstrip()}..."


def classify_intent(user_input: str) -> str:
	stripped = user_input.strip()
	if not stripped:
		return INTENT_CHAT
	if stripped.startswith("/"):
		return INTENT_CONTROL

	normalized = normalize_for_match(stripped)
	if len(stripped) <= 12:
		for term in FEEDBACK_TERMS:
			if normalize_for_match(term) in normalized:
				return INTENT_FEEDBACK

	for term in META_QUESTION_TERMS:
		if term in stripped:
			return INTENT_META_QUESTION
		if normalize_for_match(term) in normalized:
			return INTENT_META_QUESTION

	if stripped.endswith("か"):
		return INTENT_QUESTION
	for term in QUESTION_TERMS:
		if term in stripped:
			return INTENT_QUESTION
		if normalize_for_match(term) in normalized:
			return INTENT_QUESTION

	return INTENT_CHAT


def is_generic_topic(term: str) -> bool:
	normalized = normalize_for_match(term)
	if not normalized:
		return True
	return normalized in {normalize_for_match(value) for value in GENERIC_TOPIC_TERMS}


def filter_by_distance(
	documents: List[str],
	metadatas: List[dict],
	distances: List[float],
	threshold: float,
) -> Tuple[List[str], List[dict], List[float]]:
	filtered_docs: List[str] = []
	filtered_metas: List[dict] = []
	filtered_dists: List[float] = []

	for doc, meta, dist in zip(documents, metadatas, distances):
		if dist is not None and dist > threshold:
			continue
		filtered_docs.append(doc)
		filtered_metas.append(meta)
		filtered_dists.append(dist)

	return filtered_docs, filtered_metas, filtered_dists


def cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
	dot = 0.0
	norm_a = 0.0
	norm_b = 0.0
	for a, b in zip(vec_a, vec_b):
		dot += a * b
		norm_a += a * a
		norm_b += b * b
	if norm_a == 0.0 or norm_b == 0.0:
		return 1.0
	return 1.0 - (dot / (math.sqrt(norm_a) * math.sqrt(norm_b)))


def two_stage_search(
	collection: chromadb.Collection,
	question: str,
	keyword: str,
	max_get: int,
	max_candidates: int,
) -> Tuple[List[str], List[dict], List[float], int]:
	# Note: For large datasets, fetching many documents client-side can be slow and
	# memory-heavy. Consider server-side filters or smaller limits when scaling up.
	initial = collection.get(
		limit=max_get,
		include=["documents", "metadatas"],
	)
	all_docs = initial.get("documents")
	if all_docs is None:
		all_docs = []
	all_metas = initial.get("metadatas")
	if all_metas is None:
		all_metas = []
	all_ids = initial.get("ids")
	if all_ids is None:
		all_ids = []

	filtered_ids: List[str] = []
	normalized_keyword = normalize_for_match(keyword)
	for doc, doc_id in zip(all_docs, all_ids):
		if not doc:
			continue
		normalized_doc = normalize_for_match(doc)
		if normalized_keyword and normalized_keyword in normalized_doc:
			filtered_ids.append(doc_id)

	prefilter_count = len(filtered_ids)
	if prefilter_count == 0:
		return [], [], [], 0

	if prefilter_count > max_candidates:
		filtered_ids = filtered_ids[:max_candidates]

	filtered = collection.get(
		ids=filtered_ids,
		include=["documents", "metadatas", "embeddings"],
	)
	filtered_docs = filtered.get("documents")
	if filtered_docs is None:
		filtered_docs = []
	filtered_metas = filtered.get("metadatas")
	if filtered_metas is None:
		filtered_metas = []
	filtered_embeddings = filtered.get("embeddings")
	if filtered_embeddings is None:
		filtered_embeddings = []

	embedding_fn = embedding_functions.DefaultEmbeddingFunction()
	query_embedding = embedding_fn([question])[0]

	scored = []
	for doc, meta, emb in zip(filtered_docs, filtered_metas, filtered_embeddings):
		distance = cosine_distance(query_embedding, emb)
		scored.append((distance, doc, meta))

	scored.sort(key=lambda item: item[0])
	sorted_docs = [item[1] for item in scored]
	sorted_metas = [item[2] for item in scored]
	sorted_dists = [item[0] for item in scored]

	return sorted_docs, sorted_metas, sorted_dists, prefilter_count


def build_prompt(
	question: str,
	documents: List[str],
	metadatas: List[dict],
	history: List[Tuple[str, str]],
) -> str:
	history_lines = []
	for user_q, assistant_a in history:
		history_lines.append(f"User: {user_q}")
		history_lines.append(f"Assistant: {assistant_a}")

	history_text = "\n".join(history_lines).strip()
	if history_text:
		history_text = f"\n\n会話履歴:\n{history_text}"

	evidence_blocks = []
	for doc, meta in zip(documents, metadatas):
		source = f"{meta.get('file_name')} / {meta.get('sheet_name')} / {meta.get('row_index')}"
		block = f"[根拠]\n出典: {source}\n内容: {doc}"
		evidence_blocks.append(block)

	evidence_text = "\n\n---\n\n".join(evidence_blocks)

	return (
		f"{BASE_PROMPT_HEADER}"
		f"{history_text}\n\n"
		f"今回の質問: {question}\n\n"
		"根拠(この区切り内のみ参照可):\n"
		"==== BEGIN EVIDENCE ====\n"
		f"{evidence_text}\n"
		"==== END EVIDENCE ====\n"
	)


def build_inference_prompt(
	question: str,
	documents: List[str],
	metadatas: List[dict],
	history: List[Tuple[str, str]],
	current_topic: str,
) -> str:
	history_lines = []
	for user_q, assistant_a in history:
		history_lines.append(f"User: {user_q}")
		history_lines.append(f"Assistant: {assistant_a}")

	history_text = "\n".join(history_lines).strip()
	if history_text:
		history_text = f"\n\n会話履歴:\n{history_text}"

	evidence_blocks = []
	for doc, meta in zip(documents, metadatas):
		source = f"{meta.get('file_name')} / {meta.get('sheet_name')} / {meta.get('row_index')}"
		block = f"[根拠]\n出典: {source}\n内容: {doc}"
		evidence_blocks.append(block)

	evidence_text = "\n\n---\n\n".join(evidence_blocks)

	return (
		f"{INFERENCE_PROMPT_HEADER}"
		f"{history_text}\n\n"
		f"今回の質問: {question}\n"
		f"現在の主題: {current_topic}\n\n"
		"根拠(この区切り内のみ参照可):\n"
		"==== BEGIN EVIDENCE ====\n"
		f"{evidence_text}\n"
		"==== END EVIDENCE ====\n"
	)


def collect_confirmation_points(current_topic: str) -> List[str]:
	points = []
	if current_topic:
		points.append(f"{current_topic} の別表記/略称")
	points.append("該当シート名/フェーズ名")
	points.append("発動条件/トリガーの列名")
	return points


def main() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)
	LOGGER.setLevel(logging.DEBUG)
	args = parse_args()

	client = chromadb.PersistentClient(path=args.db)
	collection = client.get_or_create_collection(name="excel_chunks")

	print(f"対話モード（会話履歴 最大{args.memory_turns}ターン）")
	print("質問を入力してください。Ctrl+Cで終了します。")
	memory: List[Tuple[str, str]] = []
	current_topic = ""
	last_evidence = None
	try:
		while True:
			question = input("> ").strip()
			if not question:
				continue

			intent = classify_intent(question)
			if intent == INTENT_CONTROL:
				command = question.strip().lower()
				if command.startswith("/reset"):
					memory = []
					current_topic = ""
					last_evidence = None
					print("リセットしました。")
					continue
				if command.startswith("/topic"):
					print(f"現在の主題: {current_topic or '(未設定)'}")
					continue
				if command.startswith("/help"):
					print("利用可能コマンド: /reset /topic /help")
					continue
				print("不明なコマンドです。/help を確認してください。")
				continue

			if intent in (INTENT_FEEDBACK, INTENT_CHAT, INTENT_META_QUESTION):
				print(
					"もちろんOK。別の質問をどうぞ。必要なら対象（スキル名/フェーズ）も一緒に書いてください。"
				)
				continue

			topic_candidate = extract_topic_candidate(question)
			topic_updated = False
			if topic_candidate and not is_generic_topic(topic_candidate):
				current_topic = topic_candidate
				topic_updated = True

			follow_up, followup_reason, pronoun_ref = classify_follow_up(
				question, current_topic
			)

			query_text = question
			if follow_up and current_topic:
				query_text = f"{current_topic} {question}".strip()

			if pronoun_ref and not current_topic:
				LOGGER.debug(
					"debug_state intent=%s followup=%s followup_reason=%s pronoun_ref=%s current_topic=%s topic_candidate=%s topic_updated=%s query_text=%s prefilter_count=%s retrieved_count=%s",
					intent,
					follow_up,
					followup_reason or None,
					pronoun_ref,
					current_topic or None,
					topic_candidate or None,
					topic_updated,
					query_text,
					0,
					0,
				)
				print("対象が分からないので、何についての『それ』か教えてください。")
				continue

			keyword = extract_keyword(question)
			if not keyword:
				keyword = question
			normalized_keyword = normalize_for_match(keyword)

			search_query = query_text
			if follow_up and current_topic:
				display_keyword = f"{current_topic} + {question}"
			else:
				display_keyword = keyword

			documents, metadatas, distances, prefilter_count = two_stage_search(
				collection=collection,
				question=search_query,
				keyword=keyword,
				max_get=5000,
				max_candidates=300,
			)
			last_evidence = {
				"documents": documents,
				"metadatas": metadatas,
				"distances": distances,
			}

			inference_mode = False
			evidence_sufficient = True
			reason_message = ""

			if prefilter_count == 0:
				evidence_sufficient = False
				reason_message = "根拠から該当語が見つからないため回答不能。表記揺れ/別名で再検索してください。"

			if not documents:
				evidence_sufficient = False
				if not reason_message:
					reason_message = "該当する根拠が見つかりませんでした。"

			documents, metadatas, distances = filter_by_distance(
				documents, metadatas, distances, args.distance_threshold
			)
			retrieved_count = len(documents)
			LOGGER.debug(
				"debug_state intent=%s followup=%s followup_reason=%s pronoun_ref=%s current_topic=%s topic_candidate=%s topic_updated=%s query_text=%s prefilter_count=%s retrieved_count=%s",
				intent,
				follow_up,
				followup_reason or None,
				pronoun_ref,
				current_topic or None,
				topic_candidate or None,
				topic_updated,
				search_query,
				prefilter_count,
				retrieved_count,
			)
			if not documents:
				evidence_sufficient = False
				if not reason_message:
					reason_message = "閾値で除外され0件になりました。"

			print("\n参照した根拠:")
			for idx, (meta, dist) in enumerate(zip(metadatas, distances), start=1):
				file_name = meta.get("file_name")
				sheet_name = meta.get("sheet_name")
				row_index = meta.get("row_index")
				score = format_score(dist)
				print(
					f"{idx}. {file_name} / {sheet_name} / {row_index} (distance: {score})"
				)

			if args.show_evidence:
				print("\n根拠テキスト抜粋:")
				for idx, doc in enumerate(documents, start=1):
					snippet = make_snippet(doc, limit=200)
					print(f"{idx}. {snippet}")

			topic_ok = False
			if follow_up:
				topic_ok = any(match_topic_in_doc(current_topic, doc) for doc in documents)
				if not topic_ok:
					evidence_sufficient = False
					if not reason_message:
						reason_message = (
							"根拠から該当語が見つからないため回答不能。"
							"表記揺れ/別名で再検索してください。"
						)
			else:
				if not any(
					normalized_keyword in normalize_for_match(doc) for doc in documents
				):
					evidence_sufficient = False
					if not reason_message:
						reason_message = (
							"根拠から該当語が見つからないため回答不能。"
							"表記揺れ/別名で再検索してください。"
						)

			if not evidence_sufficient:
				if follow_up and current_topic and args.allow_inference:
					print("根拠不足のため推測モードで回答します。")
					inference_mode = True
				else:
					print(reason_message)
					print(f"検索キーワード: {display_keyword}")
					continue

			time_candidates = extract_time_candidates(documents, metadatas)
			if time_candidates:
				print("\nタイムライン候補(time2):")
				for idx, (time_value, meta) in enumerate(time_candidates, start=1):
					sheet_name = meta.get("sheet_name")
					row_index = meta.get("row_index")
					print(f"{idx}. {time_value} / {sheet_name} / {row_index}")

			if inference_mode and args.allow_inference:
				confirmation_points = collect_confirmation_points(current_topic)
				prompt = build_inference_prompt(
					question,
					documents,
					metadatas,
					memory,
					current_topic,
				)
				response = ollama.chat(
					model=args.model,
					messages=[
						{
							"role": "system",
							"content": INFERENCE_SYSTEM_PROMPT,
						},
						{"role": "user", "content": prompt},
						{
							"role": "user",
							"content": "確認すべき点の候補: " + " / ".join(confirmation_points),
						},
					],
				)
			else:
				prompt = build_prompt(question, documents, metadatas, memory)
				response = ollama.chat(
					model=args.model,
					messages=[
						{
							"role": "system",
							"content": BASE_SYSTEM_PROMPT,
						},
						{"role": "user", "content": prompt},
					],
				)

			answer = response.get("message", {}).get("content", "").strip()
			print("\n回答:")
			print(answer or "(回答が空でした)")
			print()

			memory.append((question, answer))
			if len(memory) > args.memory_turns:
				memory = memory[-args.memory_turns :]
	except KeyboardInterrupt:
		print("\n終了します。")


if __name__ == "__main__":
	main()
