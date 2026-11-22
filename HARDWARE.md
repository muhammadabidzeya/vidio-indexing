# Hardware

All 50 files were processed on Kaggle (free notebook, "Save \& Run All" background runs), not on my own computer.

* GPU: 2x NVIDIA Tesla T4, 15 GB memory each (15360 MiB)
* CPU: AMD EPYC 7B12, 4 vCPUs
* RAM: 31 GB
* Disk: working disk of 20 GB (the data was downloaded one file at a time and deleted after processing). The disk reports as non-rotational, so it is an SSD.
* Operating system: Ubuntu 24.04.5 LTS
* Python packages: see requirements.txt

## How the run was done

The 50 files were processed in two separate runs, one per half. In each run, two worker processes ran at the same time, one on each T4 GPU, and each worker processed every second file.

* Run 1 (files 0 to 24): about 9 h 27 min (33,992 s)
* Run 2 (files 25 to 49): about 8 h 31 min (8 h 30 min 55 s)
* Total: about 17 h 57 min of run time (two separate runs; time waiting in the Kaggle queue is not counted)

These times are on 2 Tesla T4 GPUs, so they are not comparable with the RTX 4090 grading machine.

