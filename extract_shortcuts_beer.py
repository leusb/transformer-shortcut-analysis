#!/usr/bin/env python3
"""Extract Beer occurrence1 and correlation1 inserted text + original rationale.

Uses PUBLISHED train.csv files and their matching ORIGINAL Excel sources.
Run from any directory; downloads the source datasets directly from GitHub:
    python extract_beer_rationale_shortcuts.py

Optional: --repo /path/to/shortcut-learning-in-text-classification
Requires: pip install pandas openpyxl
Outputs: beer_shortcuts_output/*.csv (six files)

Important: source rationale=1 is aspect evidence; it is not a guarantee that
an individual token is predictive of Palate. Never use the modified CSV's
rationale to annotate inserted text: the authors replace that with zeros.
"""

from __future__ import annotations

import argparse
import csv
import urllib.error
import urllib.request
import re
import string
from collections import Counter, defaultdict, deque
from pathlib import Path
from urllib.parse import quote

import pandas as pd

PUNCT = {".", "?", "!"}
WORD_TRIM = string.punctuation + "“”‘’…"
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/yuqing-zhou/"
    "shortcut-learning-in-text-classification/master/"
)


def norm(value: object) -> str:
    """Normalize whitespace ONLY, retaining source token spellings."""
    return " ".join(str(value).split())


def label(value: object) -> float:
    return round(float(value), 4)


def remove_suffix(full: object, ending: object, where: str) -> str:
    full_norm = norm(full)
    ending_norm = norm(ending)
    if not ending_norm or not full_norm.endswith(ending_norm):
        raise ValueError(
            f"Cannot identify original text at {where}. "
            "Check that the published train.csv matches the original XLSX files.\n"
            f"Last 180 characters: {full_norm[-180:]}\n"
            f"Expected suffix: {ending_norm[-180:]}"
        )
    prefix = full_norm[: -len(ending_norm)].strip()
    if not prefix:
        raise ValueError(f"Empty inserted text at {where}")
    return prefix


def original_index(df: pd.DataFrame) -> dict[str, deque]:
    """Index by whole-text match; retain duplicate source rows separately."""
    result: dict[str, deque] = defaultdict(deque)
    for idx, row in df.iterrows():
        result[norm(row["text"])].append((idx, row))
    return result


def take_source(index: dict[str, deque], text: str, where: str, expected_rating=None):
    candidates = index.get(norm(text))
    if not candidates:
        raise ValueError(f"No exact original source review for {where}: {norm(text)[:200]!r}")
    if expected_rating is None:
        _, row = candidates.popleft()
        return row

    # On correlation1 train, Aroma label must equal the Palate label.
    for pos, (_, row) in enumerate(candidates):
        if label(row["label"]) == label(expected_rating):
            chosen = candidates[pos][1]
            del candidates[pos]
            return chosen
    raise ValueError(f"{where}: source text exists, but not at Aroma rating {expected_rating}")


def paired_source(df: pd.DataFrame, i: int, expected_text: str, where: str):
    """Correlation uses the sampled Look review at the same row index."""
    row = df.iloc[i]
    if norm(row["text"]) != norm(expected_text):
        raise ValueError(f"Look review differs at {where}")
    return row


def validate_rationale(row: pd.Series, where: str):
    tokens = str(row["text"]).split()
    mask = str(row["rationale"]).split()
    if len(tokens) != len(mask):
        raise ValueError(
            f"Token/rationale length mismatch at {where}: "
            f"{len(tokens)} tokens versus {len(mask)} rationale values"
        )
    if any(bit not in ("0", "1") for bit in mask):
        raise ValueError(f"Rationale contains a non-binary value at {where}")
    return tokens, list(map(int, mask))


def split_annotated_sentences(tokens: list[str], rationale: list[int]):
    """Use the same standalone punctuation markers as the author's preprocessing."""
    start = 0
    for pos, token in enumerate(tokens):
        if token in PUNCT:
            yield tokens[start: pos + 1], rationale[start: pos + 1]
            start = pos + 1
    if start < len(tokens):
        yield tokens[start:], rationale[start:]


def normalize_token(token: str) -> str:
    """Remove punctuation at token boundaries, lowercase, preserve internal hyphens."""
    return token.lower().strip(WORD_TRIM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=None,
                        help="Optional local clone; if omitted, download from GitHub URLs")
    parser.add_argument("--cache", type=Path, default=Path(".beer_shortcut_cache"),
                        help="Cache downloaded input files (reused on later runs)")
    parser.add_argument("--out", type=Path, default=Path("beer_shortcuts_output"),
                        help="Output folder for six CSV files")
    args = parser.parse_args()

    # The authors' files are located under Dataset/beer_new/target/.
    # In URL mode, download the *same paths* into a local cache, then parse them.
    source_root = Path("Dataset") / "beer_new" / "target"

    def source_file(relative_path: str) -> Path:
        relative_path = Path(relative_path)
        if args.repo is not None:
            file = args.repo / source_root / relative_path
            if not file.is_file():
                raise FileNotFoundError(f"Required local source file not found: {file}")
            return file

        destination = args.cache / source_root / relative_path
        if destination.is_file() and destination.stat().st_size > 0:
            return destination

        url = GITHUB_RAW_BASE + quote((source_root / relative_path).as_posix(), safe="/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading: {url}", flush=True)
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "beer-shortcut-rationale-extractor/1.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not download {url} ({exc}). "
                "Check the URL and your network connection."
            ) from exc
        if not data:
            raise RuntimeError(f"GitHub returned an empty file: {url}")
        destination.write_bytes(data)
        return destination

    def xlsx(relative_path: str) -> pd.DataFrame:
        file = source_file(relative_path)
        df = pd.read_excel(file, engine="openpyxl")
        for column in ("text", "label", "rationale"):
            if column not in df.columns:
                raise ValueError(f"Missing column '{column}' in {file}")
        return df

    def published_csv(name: str) -> pd.DataFrame:
        file = source_file(f"concept/split/{name}/{name}1/train.csv")
        return pd.read_csv(file)

    palate = xlsx("concept/sampled_beer_Palate_train_4_ratings_1.xlsx")
    occur_aroma = original_index(xlsx("concept/top2ratings_beer_Aroma_train.xlsx"))
    occur_look = original_index(xlsx("concept/top2ratings_beer_Look_train.xlsx"))
    corr_aroma = original_index(xlsx("beer_Aroma_train.xlsx"))
    corr_look = xlsx("concept/sampled_beer_Look_train_4_ratings_1.xlsx")

    sentence_rows: dict[str, list[dict]] = {
        "occurrence_aroma": [],
        "occurrence_appearance": [],
        "correlation_low": [],
        "correlation_high": [],
    }
    # Count a token once PER OCCURRENCE where the original rationale bit is 1.
    frequencies: dict[str, Counter] = {group: Counter() for group in sentence_rows}
    doc_frequency: dict[str, Counter] = {group: Counter() for group in sentence_rows}

    for dataset in ("occurrence", "correlation"):
        df = published_csv(dataset)
        if len(df) != len(palate):
            raise ValueError(
                f"{dataset}: {len(df)} published examples but {len(palate)} palate rows. "
                "The sources may not correspond to this generated dataset."
            )

        for i, record in df.iterrows():
            rating = label(record["label"])
            original = palate.iloc[i]
            if rating != label(original["label"]):
                raise ValueError(f"{dataset} row {i}: Palate rating mismatch")

            # Both datasets end with the unmodified original Palate review.
            prefix = remove_suffix(record["text"], original["text"],
                                   f"{dataset} row {i} Palate")

            if dataset == "occurrence":
                flag = int(record["combine_flag"])
                if flag == 0:
                    group = "occurrence_appearance"
                    source = take_source(occur_look, prefix, f"occurrence row {i}, Look")
                elif flag in (1, 2):
                    group = "occurrence_aroma"
                    source = take_source(occur_aroma, prefix, f"occurrence row {i}, Aroma")
                else:
                    raise ValueError(f"Unknown combine_flag {flag} in row {i}")
                inserted = prefix
            else:
                # Correlation concatenation order: Aroma + Look + Palate.
                # Match the ORIGINAL sampled Look at the SAME row index.
                look_row = corr_look.iloc[i]
                inserted = remove_suffix(prefix, look_row["text"],
                                         f"correlation row {i} Look")
                # Only keep two extreme Palate/Aroma ratings.
                if rating not in (0.4, 1.0):
                    continue
                group = "correlation_low" if rating == 0.4 else "correlation_high"
                source = take_source(corr_aroma, inserted,
                                     f"correlation row {i}, Aroma", expected_rating=rating)
                flag = ""

            original_tokens, bits = validate_rationale(source, f"{dataset} row {i}")
            if norm(source["text"]) != norm(inserted):
                raise ValueError(f"Source does not match inserted text at {dataset} row {i}")

            review_words = set()
            sentences = list(split_annotated_sentences(original_tokens, bits))
            for sentence_no, (tokens, mask) in enumerate(sentences, start=1):
                selected_tokens = [normalize_token(tok) for tok, m in zip(tokens, mask) if m == 1]
                selected_tokens = [tok for tok in selected_tokens if any(ch.isalnum() for ch in tok)]
                frequencies[group].update(selected_tokens)
                review_words.update(selected_tokens)

                sentence_rows[group].append({
                    "dataset": dataset + "1",
                    "train_row": i,
                    "palate_rating": rating,
                    "source_rating": label(source["label"]),
                    "combine_flag": flag,
                    "sentence_number": sentence_no,
                    "sentence": " ".join(tokens),
                    "sentence_rationale": " ".join(map(str, mask)),
                    "rationale_1_tokens": " ".join(tok for tok, bit in zip(tokens, mask) if bit == 1),
                    "n_tokens": len(tokens),
                    "n_rationale_1_tokens": sum(mask),
                    "inserted_review": norm(source["text"]),
                    "original_review_rationale": norm(source["rationale"]),
                })
            doc_frequency[group].update(review_words)

    args.out.mkdir(parents=True, exist_ok=True)
    for group, rows in sentence_rows.items():
        outfile = args.out / f"{group}_sentences.csv"
        pd.DataFrame(rows, columns=[
            "dataset", "train_row", "palate_rating", "source_rating",
            "combine_flag", "sentence_number", "sentence", "sentence_rationale",
            "rationale_1_tokens", "n_tokens", "n_rationale_1_tokens",
            "inserted_review", "original_review_rationale",
        ]).to_csv(outfile, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"{outfile}: {len(rows)} sentences")

    for dataset, groups in (
        ("occurrence", ("occurrence_aroma", "occurrence_appearance")),
        ("correlation", ("correlation_low", "correlation_high")),
    ):
        records = []
        for group in groups:
            for word, count in frequencies[group].most_common():
                records.append({
                    "group": group,
                    "token": word,
                    "token_occurrences_rationale_1": count,
                    "reviews_containing_token_rationale_1": doc_frequency[group][word],
                })
        outfile = args.out / f"{dataset}_token_frequencies.csv"
        pd.DataFrame(records, columns=[
            "group", "token", "token_occurrences_rationale_1",
            "reviews_containing_token_rationale_1",
        ]).to_csv(outfile, index=False)
        print(f"{outfile}: {len(records)} word/group pairs")

    print("\nDone. All matches were verified against the original Excel review text.")


if __name__ == "__main__":
    main()
