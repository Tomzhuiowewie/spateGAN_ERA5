#!/usr/bin/env python3
"""
Download all files from a Zenodo record with resume support and MD5 checks.

Default record:
https://zenodo.org/records/17417589?by=history&from=kkframenew
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_URL = "https://zenodo.org/records/17417589?by=history&from=kkframenew"
API_URL_TEMPLATE = "https://zenodo.org/api/records/{record_id}"
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download files from a Zenodo record with resume and MD5 verification."
    )
    parser.add_argument(
        "record",
        nargs="?",
        default=DEFAULT_URL,
        help="Zenodo record URL or record ID. Default: %(default)s",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default="zenodo_17417589",
        help="Directory to save files. Default: %(default)s",
    )
    parser.add_argument(
        "-f",
        "--file",
        action="append",
        dest="files",
        help="Download only this file key. Can be used multiple times.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list files in the record, then exit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of resuming or skipping.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip MD5 checksum verification.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Download retry count per file. Default: %(default)s",
    )
    return parser.parse_args()


def record_id_from(value: str) -> str:
    if value.isdigit():
        return value
    match = re.search(r"/records/(\d+)", value)
    if match:
        return match.group(1)
    parsed = urlparse(value)
    if parsed.path:
        path_match = re.search(r"(\d+)", parsed.path)
        if path_match:
            return path_match.group(1)
    raise ValueError(f"Could not find Zenodo record ID in: {value}")


def fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "zenodo-downloader/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_md5(checksum: str | None) -> str | None:
    if not checksum:
        return None
    if checksum.startswith("md5:"):
        return checksum.split(":", 1)[1]
    return None


def print_progress(name: str, downloaded: int, total: int, start_time: float) -> None:
    elapsed = max(time.time() - start_time, 0.001)
    speed = downloaded / elapsed
    percent = (downloaded / total * 100) if total else 0.0
    message = (
        f"\r{name}: {percent:6.2f}% "
        f"({human_size(downloaded)} / {human_size(total)}) "
        f"at {human_size(int(speed))}/s"
    )
    sys.stderr.write(message)
    sys.stderr.flush()


def download_file(file_info: dict[str, Any], out_dir: Path, overwrite: bool) -> Path:
    key = file_info["key"]
    url = file_info["links"]["self"]
    expected_size = int(file_info.get("size") or 0)
    target = out_dir / key
    target.parent.mkdir(parents=True, exist_ok=True)

    existing_size = target.stat().st_size if target.exists() and not overwrite else 0
    if expected_size and existing_size == expected_size:
        print(f"skip existing complete file: {target}")
        return target

    mode = "ab" if existing_size else "wb"
    headers = {"User-Agent": "zenodo-downloader/1.0"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"resume: {key} from {human_size(existing_size)}")
    else:
        print(f"download: {key} ({human_size(expected_size)})")

    request = Request(url, headers=headers)
    with urlopen(request, timeout=120) as response:
        if existing_size and response.status == 200:
            # Server did not honor Range. Start over to avoid corrupting the file.
            existing_size = 0
            mode = "wb"
            print(f"server ignored resume for {key}; restarting")

        downloaded = existing_size
        start_time = time.time()
        with target.open(mode + "") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                print_progress(key, downloaded, expected_size, start_time)

    sys.stderr.write("\n")
    if expected_size and target.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {key}: got {target.stat().st_size}, expected {expected_size}"
        )
    return target


def verify_file(file_info: dict[str, Any], target: Path) -> None:
    expected_md5 = checksum_md5(file_info.get("checksum"))
    if not expected_md5:
        print(f"no MD5 checksum for {target.name}; skipping verification")
        return
    print(f"verify md5: {target.name}")
    actual_md5 = md5_file(target)
    if actual_md5.lower() != expected_md5.lower():
        raise RuntimeError(
            f"MD5 mismatch for {target.name}: got {actual_md5}, expected {expected_md5}"
        )
    print(f"ok: {target.name}")


def main() -> int:
    args = parse_args()
    record_id = record_id_from(args.record)
    metadata = fetch_json(API_URL_TEMPLATE.format(record_id=record_id))
    files = metadata.get("files") or []

    if args.files:
        wanted = set(args.files)
        files = [item for item in files if item.get("key") in wanted]
        missing = wanted - {item.get("key") for item in files}
        if missing:
            raise SystemExit(f"File key(s) not found: {', '.join(sorted(missing))}")

    if not files:
        raise SystemExit("No files found in this Zenodo record.")

    print(f"record: {metadata.get('title', record_id)}")
    print(f"id: {record_id}")
    print("files:")
    for item in files:
        checksum = item.get("checksum") or "no checksum"
        print(f"  - {item['key']}  {human_size(int(item.get('size') or 0))}  {checksum}")

    if args.list:
        return 0

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_dir}")

    for item in files:
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                target = download_file(item, out_dir, args.overwrite)
                if not args.no_verify:
                    verify_file(item, target)
                break
            except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                last_error = exc
                print(f"attempt {attempt}/{args.retries} failed for {item['key']}: {exc}")
                if attempt < args.retries:
                    time.sleep(min(30, 2**attempt))
        else:
            raise SystemExit(f"failed: {item['key']}: {last_error}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
