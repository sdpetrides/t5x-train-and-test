# t5x-train-and-test

## Members

Stephen Petrides (sp4076)
Zhejian Jin (zj2324)

## Project Description

Our goal is to train various sizes of the T5 model on GCP and compare the models on training performance (time, cost) as well as task performance across the models.

Since the T5 model is very large (millions of parameters), we will have to use cloud compute and TPUs for training and evaluation.

## Repository Description



## Example commands to execute the code

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
python3 -m t5x.eval \
  --gin_file=small_eval_cnn-dailymail.gin \
  --gin.EVAL_OUTPUT_DIR=\"${EVAL_OUTPUT_DIR}\" \
  --gin.DROPOUT_RATE=\"0.2\"
```

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


## Results

### Training Time

Training time from scratch.

| Model | Parameters | Training Time |
| --- | --- | --- |
| tiny | | |
| small | | |
| base | | |

Fixed cost of $5.22 per training hour.

### Evaluation

#### SuperGLUE Text Classification

#### CNN/Daily Mail abstractive summarization

| Model | Metric |
| --- | --- |
| tiny | |
| small | |
| base | |

#### SQuAD question answering
