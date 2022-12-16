# t5x-train-and-test

## Team Members

Stephen Petrides (sp4076)
Zhejian Jin (zj2324)

## Project Description

Our goal is to train various sizes of the T5 model on GCP and compare the models on training performance (time, cost) as well as task performance across the models.

### Training

Since the T5 model is very large (millions or billions of parameters), we will have to use cloud compute and TPUs for training and evaluation. Further, it's not feasible to train any of these models from scratch, due to cost and time constraints, we will train the models for a one or more epochs and estimage to total time and cost of training.

### Evaluation

Google has provided pretrained models of various sizes that can be used for evaluation of various tasks. In addition to the pretrained models, Google also provides the sequence tasks for evaluation.

We run the following experiments for each model:
 - [CNN DailyMail](https://paperswithcode.com/dataset/cnn-daily-mail-1)
 - [SuperGLUE](https://paperswithcode.com/dataset/superglue)
 - [SQuAD](https://paperswithcode.com/dataset/squad)

For each evaluation, we evaluate the model and time the experiment.

## Repository Description

This repository holds the instructions and Gin config files for setting up and running the training and evaluation experiments.

## Running Experiments

### Getting Started

Create a TPU-connected VM on Google Cloud Platform (GCP).

```
$ gcloud compute tpus tpu-vm create t5x-2022-12-13 \
    --zone=us-central1-b \
    --accelerator-type=v2-8 \
    --version=tpu-vm-base
```

Connect to the VM via SSH.

```
$ gcloud alpha compute tpus tpu-vm ssh t5x-2022-12-13 --zone=us-central1-b
```

Now, in the VM, clone the t5x repository and move into it.

```
$ git clone https://github.com/google-research/t5x
$ cd t5x
```

Install depenencies and update the $PATH.

```
$ python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
$ export PATH="/home/$USER/.local/bin:$PATH"
```

```
$ export STORAGE_BUCKET=gs://t5x-store
$ export MODEL_DIR=${STORAGE_BUCKET}/model
$ export TFDS_DATA_DIR=${STORAGE_BUCKET}/data
$ export T5X_DIR=`pwd`
```

For each evaluation experiment, create or use a Gin config file and define the `MODEL_EVAL_PAIR` environment variable.

### Prepare Evaluation Datasets

Clone the original T5 repository.

```
$ cd ~
$ git clone https://github.com/google-research/text-to-text-transfer-transformer
$ cd text-to-text-transfer-transformer/
```

Edit the task file in the following way.

```
diff --git a/t5/data/tasks.py b/t5/data/tasks.py
index 4ad29f5..faeb423 100644
--- a/t5/data/tasks.py
+++ b/t5/data/tasks.py
@@ -160,7 +160,7 @@ for b in tfds.text.glue.Glue.builder_configs.values():
 # =============================== CNN DailyMail ================================
 TaskRegistry.add(
     "cnn_dailymail_v002",
-    source=seqio.TfdsDataSource(tfds_name="cnn_dailymail:3.1.0"),
+    source=seqio.TfdsDataSource(tfds_name="cnn_dailymail:3.4.0"),
     preprocessors=[
         functools.partial(
             preprocessors.summarize,
```

Install the updated t5 package.

```
pip3 install .
```

### Evalulate

```
$ export EVAL_OUTPUT_DIR=${STORAGE_BUCKET}/${MODEL_EVAL_PAIR}
$ time python3 -m t5x.eval \
  --gin_file=${MODEL_EVAL_PAIR}.gin \
  --gin.EVAL_OUTPUT_DIR=\"${EVAL_OUTPUT_DIR}\" \
  --gin.DROPOUT_RATE=\"0.2\"
```

The expected output is below.

```
I1214 22:40:49.294055 139789406698560 evaluation.py:587] Evaluating cnn_dailymail_v002
I1214 22:40:49.335327 139789406698560 utils.py:1280] length of dataset = 146
I1214 22:40:49.335492 139789406698560 utils.py:1291] Padding infer dataset with 14 examples for even per-replica shards.
I1214 22:40:49.663203 139789406698560 utils.py:1306] The infer dataset is sharded into 1 shards with per-shard batch size of 32
I1214 22:46:54.565200 139789406698560 utils.py:1359] Inference of all batches done.
I1214 22:46:54.569870 139778054727424 evaluation.py:755] Computing metrics for cnn_dailymail_v002
I1214 22:46:54.705683 139778054727424 rouge_scorer.py:83] Using default tokenizer.
I1214 22:46:55.015949 139778054727424 metrics.py:98] rouge1 = 8.14, 95% confidence [7.09, 9.18]
I1214 22:46:55.016074 139778054727424 metrics.py:98] rouge2 = 1.53, 95% confidence [1.08, 2.02]
I1214 22:46:55.016136 139778054727424 metrics.py:98] rougeLsum = 7.47, 95% confidence [6.69, 8.41]
I1214 22:46:55.016340 139778054727424 loggers.py:96] cnn_dailymail_v002/rouge1 at step 1000000: 8.140
I1214 22:46:55.016404 139778054727424 loggers.py:96] cnn_dailymail_v002/rouge2 at step 1000000: 1.529
I1214 22:46:55.016455 139778054727424 loggers.py:96] cnn_dailymail_v002/rougeLsum at step 1000000: 7.475
I1214 22:46:55.794987 139778054727424 loggers.py:375] Appending metrics to gs://t5x-store/model-small-eval/inference_eval/cnn_dailymail_v002-metrics.jsonl
I1214 22:46:56.322395 139778054727424 loggers.py:404] Writing inferences to gs://t5x-store/model-small-eval/inference_eval/cnn_dailymail_v002-1000000.jsonl
I1214 22:46:56.566674 139778054727424 loggers.py:443] Writing completed in 0.244310 seconds (8.186316 examples/sec).
I1214 22:46:56.566818 139778054727424 evaluation.py:611] Time computing metrics: 1.996996 secs.
```
Total number of parameters: 76961152

### Evaluation Inference (Optional)

Update the `t5/evaluation/eval_utils.py` file in the following way.

```
diff --git a/t5/evaluation/eval_utils.py b/t5/evaluation/eval_utils.py
index 9e7cf54..006264c 100644
--- a/t5/evaluation/eval_utils.py
+++ b/t5/evaluation/eval_utils.py
@@ -127,10 +127,12 @@ def get_eval_metric_values(events, task_name=None):
   eval_values = {}
   for tag, event_values in events.items():
     if tag.startswith("eval"):
-      if task_name:
+      if task_name.count("/") == 1:
         _, metric_name = tag.split("/")
-      else:
+      elif task_name.count("/") == 1:
         _, task_name_from_tag, metric_name = tag.split("/")
+      else:
+        raise ValueError("Something wrong with the eval and task name.")
       eval_task_name = task_name if task_name else task_name_from_tag
       eval_values["{}/{}".format(eval_task_name, metric_name)] = event_values
   return eval_values
```

Parse the evaluation results and output them as CSV file.

```
$ python3 -m t5.scripts.parse_tb \
  --summary_dir="$VAL_DIR" \
  --seqio_summaries \
  --out_file="$VAL_DIR/results.csv" \
  --alsologtostderr
```

## Results

### Training Time

Training time from scratch.

| Model | Parameters | Size Increase  | Training Time | Training Time Increase |
| --- | --- | --- | --- | --- |
| small | `76961152` | NA |  |  | 
| base | `` | 3.67x |  |  |
| large | `783150080` | 3.5x |  |  |
| xl | `` | 3.9x |  |  |
| xxl | `` | 3.67x | |  |

Fixed cost of ~$5 per training hour on `v2-8`. We only used the `v2-8` pod type; to estimate the training time and cost for the `v3-8` would require a new set of experiments.

### Evaluation

#### CNN/Daily Mail abstractive summarization

| Model | Eval Time (seconds) | Rouge1 | Rouge2 | RougeL |
| --- | --- | --- | --- |  --- |
| small | 396 | 8.13 | 1.53 | 7.47 |
| base | 1147 | 9.78 | 1.81 | 8.70 |
| large | 2921 | 13.64 | 3.45 | 12.20 |
| xl |  |  |  |  |
| xxl |  |  |  |  |

#### SuperGLUE Text Classification

| Model | Eval Time (seconds) | ASDF |
| --- | --- | --- |
| small |  |  |
| base |  |  |
| large |  |  |
| xl |  |  |  |
| xxl |  |  |  |

#### SQuAD question answering

