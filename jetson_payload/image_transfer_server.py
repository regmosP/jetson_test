#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote


def safe_name(value):
    name = os.path.basename(unquote(value or "image.bin"))
    return name or "image.bin"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TransferHandler(BaseHTTPRequestHandler):
    server_version = "ImageTransferServer/1.0"

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404, "Not found")
            return
        self.write_json(200, {"status": "ok", "time": utc_now()})

    def do_POST(self):
        if not self.path.startswith("/upload"):
            self.send_error(404, "Not found")
            return

        content_length = self.headers.get("Content-Length")
        if not content_length:
            self.send_error(411, "Content-Length required")
            return

        try:
            expected_bytes = int(content_length)
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return

        filename = safe_name(self.headers.get("X-Filename"))
        output_path = self.unique_path(filename)
        started = time.perf_counter()
        sha256 = hashlib.sha256()
        received_bytes = 0

        with open(output_path, "wb") as output_file:
            remaining = expected_bytes
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                output_file.write(chunk)
                sha256.update(chunk)
                received_bytes += len(chunk)
                remaining -= len(chunk)

        elapsed_s = time.perf_counter() - started
        mb = received_bytes / (1024 * 1024)
        mbps = (mb / elapsed_s) if elapsed_s > 0 else 0.0
        mbit_s = mbps * 8
        status = "ok" if received_bytes == expected_bytes else "incomplete"
        result = {
            "time": utc_now(),
            "client": self.client_address[0],
            "filename": filename,
            "saved_path": output_path,
            "status": status,
            "bytes": received_bytes,
            "expected_bytes": expected_bytes,
            "elapsed_s": elapsed_s,
            "mb_per_s": mbps,
            "mbit_per_s": mbit_s,
            "sha256": sha256.hexdigest(),
        }
        self.server.write_log(result)
        self.write_json(200 if status == "ok" else 400, result)

    def log_message(self, fmt, *args):
        if self.server.quiet:
            return
        super().log_message(fmt, *args)

    def unique_path(self, filename):
        base, ext = os.path.splitext(filename)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return os.path.join(self.server.save_dir, f"{base}_{stamp}{ext}")

    def write_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ImageTransferServer(ThreadingHTTPServer):
    def __init__(self, address, handler_class, save_dir, log_dir, quiet=False):
        super().__init__(address, handler_class)
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.quiet = quiet
        self.csv_path = os.path.join(log_dir, "image_transfer_log.csv")
        self.jsonl_path = os.path.join(log_dir, "image_transfer_log.jsonl")
        self.csv_fields = [
            "time",
            "client",
            "filename",
            "saved_path",
            "status",
            "bytes",
            "expected_bytes",
            "elapsed_s",
            "mb_per_s",
            "mbit_per_s",
            "sha256",
        ]
        self.ensure_logs()

    def ensure_logs(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as csv_file:
                csv.DictWriter(csv_file, fieldnames=self.csv_fields).writeheader()

    def write_log(self, result):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as csv_file:
            csv.DictWriter(csv_file, fieldnames=self.csv_fields).writerow(result)
        with open(self.jsonl_path, "a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        if not self.quiet:
            print(
                f"{result['time']} {result['filename']} {result['bytes']} bytes "
                f"{result['elapsed_s']:.6f}s {result['mb_per_s']:.3f} MiB/s "
                f"{result['mbit_per_s']:.3f} Mbit/s",
                flush=True,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Receive images and log transfer time/bandwidth.")
    parser.add_argument("--host", default="0.0.0.0", help="Listen address, default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=8000, help="Listen port, default: 8000")
    parser.add_argument("--save-dir", default="received_images", help="Directory for uploaded images")
    parser.add_argument("--log-dir", default="transfer_logs", help="Directory for CSV/JSONL logs")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-upload logs")
    return parser.parse_args()


def main():
    args = parse_args()
    server = ImageTransferServer(
        (args.host, args.port),
        TransferHandler,
        os.path.abspath(args.save_dir),
        os.path.abspath(args.log_dir),
        args.quiet,
    )
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"Saving images to {server.save_dir}")
    print(f"Writing logs to {server.csv_path} and {server.jsonl_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
