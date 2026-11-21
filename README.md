# Video Indexing

Reads FineVideo Parquet files, runs YOLO11l on every frame, and writes one JSON line per second of video with the objects present and the people count.

## Install

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

No apt-get packages are needed.

## Preparation command (needs internet, run once before the timed run)

    python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"

Run it from the repository folder so that yolo11l.pt is saved next to run.py.

## Run

    python run.py --input <folder of .parquet files> --output results.jsonl

## Notes

- Developed and tested on Kaggle with 2x Tesla T4 GPUs.
- big_run.py is only a helper I used on Kaggle to process the 50 files in two parallel halves. It is not needed to run run.py.
