# Write-up

## 1\. Your timings (100 words)

#### Ans:The full run time is about 17hours and 57 minutes total on two t4 GPU on kagle



## 

## 2\. Your pipeline (150 words)

#### Ans:

#### 1.It download one Parquet file 

#### 2\.read its rows 

#### 3\.each row's mp4 bytes decoded frame by 4.frame frames batched into YOLO11l

#### 5\.keep only person/car/dog/laptop/cell phone group frames by second

#### 6\.object present if in at least half that second frames people = median of the per-frame counts

#### 7\.one JSON line per second

#### 8\.per-file results merged and gzipped.

#### Two workers ran in parallel one per GPU

## 

## 3\. Where the time goes (150 words)

#### Ans:the pipeline has three hungry stages: 1.downloading each Parquet file 

#### 2.decoding frames (CPU work) 

#### 3.inference in YOLO (GPU work)

so all the time goes to these three steps so first the downloading of files then decoding it in frames and then inference and judge by yolo model is time taking 

## 4\. Checking correctness (100 words)

#### Ans:Checking the correctness by changing the script and looking toward the accuracy of the model while at the mean time the time to process it goes down while the accuracy remaind in good numbers 

## 

## 5\. Working with AI (100 words)

#### Ans:I worked with claude ai and give him every instruction to make the project while stay in charge claude write code and give step by step instruction to all things including with kagle and accounts making and working and the only thing i do my self is writing these answers

## 

## 6\. Next steps (100 words)

#### Ans:I faced a limitation with Kagle session time which was about 12 hours while my complete run took around 18 hours.To handle this I divided the 50 files into two runs of 25 files each and used two T4 GPUs in parallel.I also saved each files results separately to avoid losing completed work if a session stopped.With one more week I would try to run all 50 files continuously on a GPU without a session limit which would remove the need for splitting and merging the results.

