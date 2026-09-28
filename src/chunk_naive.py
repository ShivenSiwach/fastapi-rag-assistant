import json
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.jsonl"
SOURCES = [(RAW / "tutorial", ""), (RAW / "advanced", "advanced/")]

CHUNK_SIZE_WORDS = 150


def chunk_text(text: str, size: int = CHUNK_SIZE_WORDS):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks, idx, n_files = [], 0, 0
    for root, prefix in SOURCES:
        for path in sorted(root.rglob("*.md")):
            n_files += 1
            text = path.read_text(encoding="utf-8", errors="ignore")
            rel = prefix + path.relative_to(root).as_posix()
            for piece in chunk_text(text):
                if len(piece.strip()) < 30:
                    continue
                chunks.append({"chunk_id": f"chunk_{idx:04d}", "source_file": rel, "text": piece})
                idx += 1

    with open(OUT_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"{n_files} files -> {len(chunks)} chunks")


if __name__ == "__main__":
    main()