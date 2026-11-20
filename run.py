import argparse, glob, json, os, queue, tempfile, threading, time
from collections import defaultdict

import av
import numpy as np
import pyarrow.parquet as pq
from ultralytics import YOLO
from ultralytics.utils import LOGGER

LOGGER.setLevel("ERROR")

NAMES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
CLASSES = list(NAMES.keys())
BATCH = 32


def decode_video(path):
    """Yield (timestamp_seconds, BGR frame) for every frame, rotated if tagged."""
    c = av.open(path)
    stream = c.streams.video[0]
    rot = 0
    try:
        for sd in stream.side_data:
            if "DISPLAYMATRIX" in str(sd.type).upper():
                rot = int(round(float(sd.rotation))) % 360 if hasattr(sd, "rotation") else 0
    except Exception:
        pass
    if rot == 0:
        r = stream.metadata.get("rotate")
        rot = int(r) % 360 if r else 0
    t0 = None
    for fr in c.decode(stream):
        if fr.pts is None:
            continue
        t = float(fr.pts * fr.time_base)
        if t0 is None:
            t0 = t
        img = fr.to_ndarray(format="bgr24")
        if rot == 90:
            img = np.rot90(img, 1)
        elif rot == 180:
            img = np.rot90(img, 2)
        elif rot == 270:
            img = np.rot90(img, 3)
        yield t - t0, np.ascontiguousarray(img)
    c.close()


def prefetch(gen, size=256):
    """Run a generator in a background thread so decoding overlaps with the GPU."""
    q = queue.Queue(maxsize=size)
    END = object()

    def worker():
        try:
            for item in gen:
                q.put(item)
        except Exception as e:
            q.put(e)
        q.put(END)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is END:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def summarize(by_sec):
    """Turn {second: [per-frame class lists]} into output rows."""
    rows = []
    last = max(by_sec) if by_sec else -1
    for sec in range(last + 1):
        fl = by_sec.get(sec, [])
        n = len(fl)
        objects = []
        people = 0
        if n:
            for k in CLASSES:
                if sum(1 for d in fl if k in d) * 2 >= n:
                    objects.append(NAMES[k])
            counts = sorted(d.count(0) for d in fl)
            people = counts[n // 2] if n % 2 else (counts[n // 2 - 1] + counts[n // 2]) // 2
        rows.append({"video": None, "second": sec, "objects": objects, "people": people})
    return rows


def process_video(model, mp4_bytes, tmp_path):
    with open(tmp_path, "wb") as f:
        f.write(mp4_bytes)
    by_sec = defaultdict(list)
    buf_f, buf_t = [], []

    def flush():
        if not buf_f:
            return
        res = model.predict(buf_f, imgsz=640, conf=0.4, iou=0.7,
                            classes=CLASSES, half=True, verbose=False)
        for t, r in zip(buf_t, res):
            by_sec[int(t)].append([int(b) for b in r.boxes.cls.tolist()])
        buf_f.clear()
        buf_t.clear()

    err = None
    try:
        for t, img in prefetch(decode_video(tmp_path)):
            buf_f.append(img)
            buf_t.append(t)
            if len(buf_f) >= BATCH:
                flush()
        flush()
    except Exception as e:
        err = str(e)[:200]
        try:
            flush()
        except Exception:
            pass
    return summarize(by_sec), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model = YOLO("yolo11l.pt")
    tmp_path = os.path.join(tempfile.gettempdir(), f"_video_{os.getpid()}.mp4")
    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    start = time.time()
    with open(args.output, "w") as out:
        for fpath in files:
            fname = os.path.basename(fpath)
            try:
                table = pq.ParquetFile(fpath).read(columns=["mp4"])
            except Exception as e:
                out.write(json.dumps({"video": fname + ":0", "error": "bad parquet: " + str(e)[:100]}) + "\n")
                continue
            for i in range(table.num_rows):
                vid = f"{fname}:{i}"
                try:
                    mp4 = table.column("mp4")[i].as_py()
                    rows, err = process_video(model, mp4, tmp_path)
                except Exception as e:
                    rows, err = [], str(e)[:200]
                for r in rows:
                    r["video"] = vid
                    out.write(json.dumps(r) + "\n")
                if err or not rows:
                    out.write(json.dumps({"video": vid, "error": err or "could not decode video"}) + "\n")
                out.flush()
                print(vid, len(rows), "seconds", round(time.time() - start, 1), "s elapsed", flush=True)


if __name__ == "__main__":
    main()