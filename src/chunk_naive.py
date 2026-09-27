import json
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "tutorial"
OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.jsonl"

CHUNK_SIZE_WORDS = 150


def chunk_text(text: str, size: int = CHUNK_SIZE_WORDS):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    md_files = sorted(RAW_DIR.rglob("*.md"))

    chunks = []
    idx = 0
    for path in md_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(RAW_DIR).as_posix()
        for piece in chunk_text(text):
            if len(piece.strip()) < 30:
                continue
            chunks.append({
                "chunk_id": f"chunk_{idx:04d}",
                "source_file": rel,
                "text": piece,
            })
            idx += 1

    with open(OUT_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"{len(md_files)} files -> {len(chunks)} chunks")
    print("\nExample chunk (notice it just starts/stops mid-thought):\n")
    print(f"  [{chunks[5]['source_file']}]")
    print(" ", chunks[5]["text"][:300], "...")


if __name__ == "__main__":
    main()