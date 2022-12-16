# t5x-train-and-test

## Team Members

- Stephen Petrides (sp4076)
- Zhejian Jin (zj2324)

## Project Description

Our goal is to train various sizes of the T5 model on GCP and compare the models on training (time/cost) and task performance.

### Training

Since the T5 model is very large, millions or billions of parameters, we will have to use cloud compute and TPUs for training and evaluation. Further, it's not feasible to train any of these models from scratch, due to cost and time constraints; therefore, we will train the models for a one or more epochs and estimate to total time and cost of training.

### Evaluation

Google has provided pretrained models of various sizes that can be used for evaluation of various tasks. In addition to the pretrained models, Google also provides the sequence tasks for evaluation as TF datasets and as SeqIO tasks/mixtures.

We run the following experiments for each model:
 - [CNN DailyMail](https://paperswithcode.com/dataset/cnn-daily-mail-1)
 - [SuperGLUE (Common Sense Reasoning on ReCoRD)](https://paperswithcode.com/dataset/superglue)
 - [SQuAD](https://paperswithcode.com/dataset/squad)

For each trial, we evaluate the model and time the experiment.

## Repository Description

This repository holds the instructions and Gin config files for setting up and running the training and evaluation experiments.

## Running Experiments

### Getting Started

Create a TPU-connected VM on Google Cloud Platform (GCP).

```
$ gcloud compute tpus tpu-vm create t5x-vm-1 \
    --zone=us-central1-b \
    --accelerator-type=v2-8 \
    --version=tpu-vm-base
```

Connect to the VM via SSH.

```
$ gcloud alpha compute tpus tpu-vm ssh t5x-vm-1 --zone=us-central1-b
```

Now, in the VM, clone the t5x repository.

```
$ git clone https://github.com/google-research/t5x
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

For each experiment, make sure to define a Gin file and the `MODEL_EVAL_PAIR` and `EVAL_OUTPUT_DIR` environment variables.

```
$ export EVAL_OUTPUT_DIR=${STORAGE_BUCKET}/${MODEL_EVAL_PAIR}
$ cd t5x
$ time python3 -m t5x.eval \
  --gin_file=${MODEL_EVAL_PAIR}.gin \
  --gin.EVAL_OUTPUT_DIR=\"${EVAL_OUTPUT_DIR}\" \
  --gin.DROPOUT_RATE=\"0.2\"
```

The expected output is below.

```
evaluation.py:587] Evaluating cnn_dailymail_v002
utils.py:1280] length of dataset = 146
utils.py:1291] Padding infer dataset with 14 examples for even per-replica shards.
utils.py:1306] The infer dataset is sharded into 1 shards with per-shard batch size of 32
utils.py:1359] Inference of all batches done.
evaluation.py:755] Computing metrics for cnn_dailymail_v002
rouge_scorer.py:83] Using default tokenizer.
metrics.py:98] rouge1 = 8.14, 95% confidence [7.09, 9.18]
metrics.py:98] rouge2 = 1.53, 95% confidence [1.08, 2.02]
metrics.py:98] rougeLsum = 7.47, 95% confidence [6.69, 8.41]
loggers.py:96] cnn_dailymail_v002/rouge1 at step 1000000: 8.140
loggers.py:96] cnn_dailymail_v002/rouge2 at step 1000000: 1.529
loggers.py:96] cnn_dailymail_v002/rougeLsum at step 1000000: 7.475
loggers.py:375] Appending metrics to gs://t5x-store/model-small-eval/inference_eval/cnn_dailymail_v002-metrics.jsonl
loggers.py:404] Writing inferences to gs://t5x-store/model-small-eval/inference_eval/cnn_dailymail_v002-1000000.jsonl
loggers.py:443] Writing completed in 0.244310 seconds (8.186316 examples/sec).
evaluation.py:611] Time computing metrics: 1.996996 secs.
```

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

Training time from scratch on a TPU `v2-8` pod.

| Model | Parameters | Size Increase | Training Time | Training Time Increase |
| --- | --- | --- | --- | --- |
| small | `76961152` | NA |  |  | 
| base | `247577856` | 3.22x |  |  |
| large | `783150080` | 3.16x |  |  |
| xl | `` | 3.9x |  |  |
| xxl | `` | 3.67x | |  |

Fixed cost of ~$5 per training hour on `v2-8`. We only used the `v2-8` pod type; to estimate the training time and cost for the `v3-8` would require a new set of experiments.

### Evaluation

Evaluation was done on CPU.

#### CNN/Daily Mail abstractive summarization

| Model | Eval Time (sec) | Rouge1 | Rouge2 | RougeL |
| --- | --- | --- | --- |  --- |
| small | 396 | 8.13 | 1.53 | 7.47 |
| base | 1147 | 9.78 | 1.81 | 8.70 |
| large | 2921 | 13.64 | 3.45 | 12.20 |
| xl |  |  |  |  |
| xxl |  |  |  |  |

#### SuperGLUE Text Classification

| Model | Eval Time (sec) |  |
| --- | --- | --- |
| small |  |  |
| base |  |  |
| large |  |  |
| xl |  |  |
| xxl |  |  |

#### SQuAD question answering

| Model | Eval Time (sec) | EM | FM |
| --- | --- | --- | --- |
| small | 591 | 0.000 | 1.697 |
| base | 1548 | 0.000 | 3.408 |
| large |  |  |  |
| xl |  |  |  |
| xxl |  |  |  |