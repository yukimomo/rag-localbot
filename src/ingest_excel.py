import argparse
import logging
import os
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm
import chromadb


LOGGER = logging.getLogger(__name__)


def normalize_text(value: str) -> str:
	return " ".join(value.split())


def is_short_noise(value: str) -> bool:
	if len(value) <= 1:
		return not value.isdigit()
	return False


def build_documents_from_sheet(
	df: pd.DataFrame,
	rel_path: str,
	file_name: str,
	sheet_name: str,
) -> Tuple[List[str], List[dict], List[str]]:
	documents: List[str] = []
	metadatas: List[dict] = []
	ids: List[str] = []

	for row_idx, row in enumerate(df.itertuples(index=False), start=2):
		value_tokens: List[str] = []
		parts: List[str] = []
		for col_name, value in zip(df.columns, row):
			if pd.isna(value):
				continue
			value_str = normalize_text(str(value))
			if not value_str:
				continue
			if is_short_noise(value_str):
				continue
			value_tokens.append(value_str)
			if str(col_name).startswith("Unnamed:"):
				continue
			parts.append(f"{col_name}={value_str}")

		if not parts:
			continue

		values_text = " ".join(value_tokens)
		# Design intent: put value-only tokens first (including Unnamed columns) to boost
		# named-entity matching while retaining col=value details for context/filtering.
		documents.append(f"VALUES: {values_text} / " + " / ".join(parts))
		metadatas.append(
			{
				"file_path": rel_path,
				"file_name": file_name,
				"sheet_name": sheet_name,
				"row_index": row_idx,
			}
		)
		ids.append(f"{rel_path}::{sheet_name}::{row_idx}")

	return documents, metadatas, ids


def iter_excel_files(input_dir: str) -> List[str]:
	excel_files: List[str] = []
	for root, _, files in os.walk(input_dir):
		for file_name in files:
			if file_name.lower().endswith(".xlsx"):
				excel_files.append(os.path.join(root, file_name))
	return excel_files


def ingest_excel_folder(input_dir: str, db_dir: str) -> None:
	client = chromadb.PersistentClient(path=db_dir)
	collection = client.get_or_create_collection(name="excel_chunks")

	excel_files = iter_excel_files(input_dir)
	if not excel_files:
		LOGGER.warning("No .xlsx files found under: %s", input_dir)
		return

	for file_path in tqdm(excel_files, desc="Excel files", unit="file"):
		rel_path = os.path.relpath(file_path, start=input_dir)
		file_name = os.path.basename(file_path)

		try:
			excel = pd.ExcelFile(file_path, engine="openpyxl")
		except Exception as exc:  # noqa: BLE001
			LOGGER.warning("Skipping unreadable file: %s (%s)", rel_path, exc)
			continue

		collection.delete(where={"file_path": rel_path})

		for sheet_name in excel.sheet_names:
			try:
				df = excel.parse(sheet_name=sheet_name, header=0)
			except Exception as exc:  # noqa: BLE001
				LOGGER.warning(
					"Skipping unreadable sheet: %s [%s] (%s)",
					rel_path,
					sheet_name,
					exc,
				)
				continue

			documents, metadatas, ids = build_documents_from_sheet(
				df=df,
				rel_path=rel_path,
				file_name=file_name,
				sheet_name=sheet_name,
			)

			if not documents:
				continue

			batch_size = 1000
			for i in tqdm(
				range(0, len(documents), batch_size),
				desc=f"Rows: {file_name}::{sheet_name}",
				unit="batch",
				leave=False,
			):
				collection.add(
					documents=documents[i : i + batch_size],
					metadatas=metadatas[i : i + batch_size],
					ids=ids[i : i + batch_size],
				)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Ingest Excel files into Chroma")
	parser.add_argument("--input", required=True, help="Input folder with .xlsx files")
	parser.add_argument("--db", required=True, help="Chroma DB directory")
	return parser.parse_args()


def main() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)
	args = parse_args()
	ingest_excel_folder(args.input, args.db)


if __name__ == "__main__":
	# Edge cases:
	# - Merged cells: pandas reads only the top-left cell; remaining merged cells appear empty.
	# - Empty rows: skipped because no non-empty "col=value" parts are produced.
	# - Multiple header rows: only the first row is treated as header; additional header-like rows
	#   are ingested as data rows.
	main()
