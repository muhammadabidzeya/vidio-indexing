import argparse, glob, gzip, os, shutil, subprocess, sys, time

REPO = "HuggingFaceFV/finevideo"
START = int(os.environ.get("START", 0))
END = int(os.environ.get("END", 50))
GPUS = [0, 1]
OUT = "/kaggle/working/results_parts"


def worker(gpu):
    from huggingface_hub import hf_hub_download
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    files = list(range(START, END))[gpu::len(GPUS)]
    for i in files:
        name = f"train-{i:05d}-of-01357.parquet"
        final = f"{OUT}/{name}.jsonl"
        if os.path.exists(final):
            print("skip", name, flush=True)
            continue
        d = f"/kaggle/working/data_gpu{gpu}"
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        ok = False
        for attempt in range(3):
            try:
                hf_hub_download(REPO, f"data/{name}", repo_type="dataset", local_dir=d)
                ok = True
                break
            except Exception as e:
                print("download failed", name, attempt, e, flush=True)
                time.sleep(5)
        if not ok:
            print("GAVE UP on", name, flush=True)
            continue
        tmp = final + ".tmp"
        r = subprocess.run([sys.executable, "run.py", "--input", f"{d}/data", "--output", tmp], env=env)
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, final)
            print("done", name, round(time.time() - t0), "s", flush=True)
        else:
            print("FAILED", name, "returncode", r.returncode, flush=True)
            if os.path.exists(tmp):
                os.remove(tmp)
        shutil.rmtree(d, ignore_errors=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    from ultralytics import YOLO
    YOLO("yolo11l.pt")
    procs = []
    for g in GPUS:
        log = open(f"/kaggle/working/log_gpu{g}.txt", "a")
        procs.append(subprocess.Popen([sys.executable, "big_run.py", "--gpu", str(g)],
                                      stdout=log, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    parts = sorted(glob.glob(f"{OUT}/*.jsonl"))
    with gzip.open("/kaggle/working/results.jsonl.gz", "wt") as z:
        for part in parts:
            with open(part) as f:
                shutil.copyfileobj(f, z)
    wanted = {f"train-{i:05d}-of-01357.parquet.jsonl" for i in range(START, END)}
    missing = sorted(wanted - {os.path.basename(p) for p in parts})
    print("merged", len(parts), "parts; missing:", missing or "none")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int)
    a = ap.parse_args()
    worker(a.gpu) if a.gpu is not None else main()
