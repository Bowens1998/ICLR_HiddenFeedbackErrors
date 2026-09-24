#!/usr/bin/env python3
"""Reassemble and verify the unchanged v9 reproduction ZIP (Python 3 only)."""
import argparse
import hashlib
import json
from pathlib import Path


ARCHIVE = "when_predictions_become_inputs_anonymous_reproduction_v9.zip"
EXPECTED_SHA256 = "037b202de1e19f76f5cb460ed1d5c3678f36b3ee017eed819547fa95d20ea6d3"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.parts_dir / "REPRODUCTION_PARTS.json").read_text())
    if manifest["output"] != ARCHIVE or manifest["sha256"] != EXPECTED_SHA256:
        raise ValueError("The part manifest does not describe the accepted v9 archive.")
    parts = manifest["parts"]
    if len(parts) != 14:
        raise ValueError("Expected all 14 archive parts.")
    for i, part in enumerate(parts, 1):
        if part["file"] != f"{ARCHIVE}.part{i:03d}":
            raise ValueError("Unexpected archive part name or order.")
        path = args.parts_dir / part["file"]
        if not path.is_file() or path.stat().st_size != part["bytes"]:
            raise ValueError(f"Missing or incomplete part: {path.name}")

    destination_dir = args.output_dir or args.parts_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / ARCHIVE
    if destination.exists():
        if checksum(destination) == EXPECTED_SHA256:
            print(f"Verified existing archive: {destination}")
            return
        raise FileExistsError(f"Output exists with a different checksum: {destination}")

    temporary = destination.with_name(destination.name + ".assembling")
    total = hashlib.sha256()
    created = False
    try:
        with temporary.open("xb") as output:
            created = True
            for part in parts:
                digest = hashlib.sha256()
                with (args.parts_dir / part["file"]).open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                        total.update(block)
                        output.write(block)
                if digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"Checksum mismatch: {part['file']}")
                print(f"Verified {part['file']}")
        if temporary.stat().st_size != manifest["bytes"] or total.hexdigest() != EXPECTED_SHA256:
            raise ValueError("Combined archive failed verification.")
        if destination.exists():
            raise FileExistsError(f"Output appeared during assembly: {destination}")
        temporary.rename(destination)
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise
    print(f"Created and verified: {destination}")
    print(f"SHA256: {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
