# Biscuit Jetson Runtime

This folder is a Jetson runnable copy of the `D:\bis` inference program. It uses `models/best.pt` by default and records inference timing logs.

## Copy To Jetson

From Windows PowerShell:

```powershell
scp -r .\bis_jetson_runtime jetson@192.168.10.20:/home/jetson/
```

## Install Dependencies On Jetson

```bash
cd /home/jetson/bis_jetson_runtime
python3 -m pip install -r requirements-runtime.txt
```

If Jetson already has OpenCV installed through JetPack, avoid reinstalling `opencv-python` if it causes conflicts.

## Run Timing Test

```bash
cd /home/jetson/bis_jetson_runtime
bash run_jetson_timing.sh
```

Run each image 10 times:

```bash
REPEAT=10 WARMUP=2 bash run_jetson_timing.sh
```

Use another image folder:

```bash
IMAGE_SOURCE=/home/jetson/images REPEAT=10 bash run_jetson_timing.sh
```

## Timing Logs

The script writes:

- `runs/timing_*/timing.csv`: per-image timing.
- `runs/timing_*/inference_time.log`: plain text inference time log, one line per image run.
- `runs/timing_*/summary.json`: min/avg/p50/p90/p95/max summary.

Important fields:

- `infer_ms`: YOLO model inference call time.
- `total_ms`: inference plus post-processing and optional output saving.
- `model_load_ms`: model loading time, not included in per-image timing.

Pull logs back to Windows:

```powershell
scp -r jetson@192.168.10.20:/home/jetson/bis_jetson_runtime/runs .\bis_jetson_runtime_runs
```
