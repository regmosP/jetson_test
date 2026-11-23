#!/usr/bin/env python3
import argparse
import json
import mimetypes
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def iter_images(paths, recursive):
    for path_text in paths:
        path = Path(path_text)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for candidate in sorted(path.glob(pattern)):
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                    yield candidate
        elif path.is_file():
            yield path


def upload_image(server_url, path, timeout):
    data = path.read_bytes()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    request = Request(
        urljoin(server_url.rstrip("/") + "/", "upload"),
        data=data,
        method="POST",
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(data)),
            "X-Filename": path.name,
        },
    )
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    client_elapsed_s = time.perf_counter() - started
    result = json.loads(body)
    result["client_elapsed_s"] = client_elapsed_s
    result["client_mb_per_s"] = (len(data) / (1024 * 1024) / client_elapsed_s) if client_elapsed_s > 0 else 0.0
    result["source_path"] = os.fspath(path)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Send images to image_transfer_server.py.")
    parser.add_argument("--server", required=True, help="Server URL, for example: http://192.168.10.200:8000")
    parser.add_argument("--image", action="append", required=True, help="Image file or directory. Repeat as needed.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan image directories")
    parser.add_argument("--repeat", type=int, default=1, help="Upload each image N times, default: 1")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds, default: 60")
    return parser.parse_args()


def main():
    args = parse_args()
    images = list(iter_images(args.image, args.recursive))
    if not images:
        raise SystemExit("No image files found")

    for round_index in range(args.repeat):
        for image in images:
            try:
                result = upload_image(args.server, image, args.timeout)
            except HTTPError as exc:
                raise SystemExit(f"Upload failed for {image}: HTTP {exc.code} {exc.reason}") from exc
            except URLError as exc:
                raise SystemExit(f"Upload failed for {image}: {exc.reason}") from exc
            print(
                f"round={round_index + 1} file={image} bytes={result['bytes']} "
                f"server_elapsed={result['elapsed_s']:.6f}s "
                f"server_speed={result['mb_per_s']:.3f}MiB/s "
                f"client_elapsed={result['client_elapsed_s']:.6f}s "
                f"client_speed={result['client_mb_per_s']:.3f}MiB/s"
            )


if __name__ == "__main__":
    main()
