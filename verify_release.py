"""Verify the archived scientific files from anywhere using only Python's standard library."""
from hashlib import sha256
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    records = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    for record in records:
        expected, relative = record.split("  ", maxsplit=1)
        with (root / relative).open("rb") as data:
            digest = sha256()
            for chunk in iter(lambda: data.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise SystemExit(f"Checksum mismatch for {relative}")
    print(f"Verified {len(records)} scientific files.")


if __name__ == "__main__":
    main()
