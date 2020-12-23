# -*- coding: utf-8 -*-
from bert import modeling
import os
import tensorflow as tf
from utils import load_qid_from_cv
import time
import collections
from functions import model_fn_builder, input_fn_builder
from config import checkpoint_dict, config_dict

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "device", "0",
    "CUDA device number")

flags.DEFINE_string(
    'tpu', None,
    'tpu address'
)

flags.DEFINE_string(
    'output_path', None,
    'output path'
)

flags.DEFINE_string(
    'dataset', None,
    'dataset: robust04 or gov2'
)

flags.DEFINE_integer(
    'max_seq_length', 384,
    "max sequence length for BERT"
)

flags.DEFINE_string(
    'model_size', None,
    'BERT model size in the current phase'
)

flags.DEFINE_string(
    "task", None,
    "current task: passage or chunk"
)

flags.DEFINE_integer(
    'fold', None,
    'fold index'
)

flags.DEFINE_integer(
    'kc', 10,
    "kc in the paper"
)

flags.DEFINE_integer(
    'batch_size', 32,
    'batch size'
)

if FLAGS.task == "chunk":
    assert FLAGS.fold is not None, \
        "'--fold' must be provided for cross-validation when running the 'chunk' task!"

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device

init_checkpoint = checkpoint_dict[FLAGS.model_size]  # @param {type:"string"}
print('***** BERT Init Checkpoint: {} *****'.format(init_checkpoint))
print('***** Model output directory: {} *****'.format(FLAGS.output_path))

# Parameters
use_tpu = False if FLAGS.tpu is None else True
iterations_per_loop = 1000
num_tpu_cores = 8


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    bert_config = modeling.BertConfig.from_json_file(config_dict[FLAGS.model_size])

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tpu_cluster_resolver = None
    if use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.output_path,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=2,
        init_checkpoint=init_checkpoint,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
        predict_batch_size=FLAGS.batch_size)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.batch_size)

    if FLAGS.task == "passage":
        path = FLAGS.output_path
    else:
        path = os.path.join(FLAGS.output_path, "fold-" + str(FLAGS.fold))

    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)

    split_list = ["valid", "test"] if FLAGS.task == "chunk" else [""]
    for split in split_list:
        suffix = ""
        if split is not "":
            suffix = "_" + split

        predictions_path = os.path.join(path, "{}_query_{}_score{}.tsv".format(FLAGS.dataset, FLAGS.task, suffix))
        ids_file_path = os.path.join(path, "query_{}_ids{}.txt".format(FLAGS.task, suffix))
        dataset_path = os.path.join(path, "query_{}{}.tf".format(FLAGS.task, suffix))

        query_chunks_ids = []
        with tf.gfile.Open(ids_file_path) as ids_file:
            for line in ids_file:
                qid, pid = line.strip().split("\t")
                query_chunks_ids.append([qid, pid])

        predict_input_fn = input_fn_builder(
            dataset_path=dataset_path,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        tf.logging.set_verbosity(tf.logging.WARN)

        result = estimator.predict(input_fn=predict_input_fn,
                                   yield_single_examples=True)

        start_time = time.time()
        cnt = 0
        with tf.gfile.Open(predictions_path, "w") as predictions_file:
            for item in result:

                qid = query_chunks_ids[cnt][0]
                pid = query_chunks_ids[cnt][1]
                doc_id = pid.split("_")[0]

                probs = item["probs"]
                scores = probs[1]

                predictions_file.write(
                    "\t".join((qid, doc_id, pid, str(float(scores)))) + "\n")
                cnt += 1
                if cnt % 10000 == 0:
                    print("process {} pairs  in {} secs.".format(
                        cnt, int(time.time() - start_time)))

            print("Done Evaluating!\nTotal examples:{}".format(cnt))

        if FLAGS.task == "passage":
            for fold in range(1, 6):
                cur_path = os.path.join(path, "fold-" + str(fold))
                if not tf.gfile.Exists(cur_path):
                    tf.gfile.MakeDirs(cur_path)
                for split in ["train", "valid", "test"]:
                    qid_list = load_qid_from_cv(FLAGS.dataset, fold, split)

                    with tf.gfile.Open(predictions_path, 'r') as ref_file, \
                            tf.gfile.Open(
                                os.path.join(cur_path, "{}_query_passage_score_{}.tsv".format(FLAGS.dataset, split)),
                                'w') as out_file, \
                            tf.gfile.Open(
                                os.path.join(cur_path, "{}_query_passage_{}_top1.tsv".format(FLAGS.dataset, split)),
                                'w') as top_file:
                        top_res = collections.OrderedDict()
                        for line in ref_file:
                            qid, doc_id, pid, score = line.strip().split()
                            score = float(score)
                            if qid in qid_list:
                                out_file.write(line)

                                if qid not in top_res:
                                    top_res[qid] = dict()
                                if doc_id not in top_res[qid]:
                                    top_res[qid][doc_id] = {"pid": pid, "score": score}
                                else:
                                    if score > top_res[qid][doc_id]["score"]:
                                        top_res[qid][doc_id]["pid"] = pid
                                        top_res[qid][doc_id]["score"] = score

                        for qid, docs in top_res.items():
                            sorted_docs = sorted(docs.items(), key=lambda x: x[1]["score"], reverse=True)
                            for rank, (doc_id, pid_score) in enumerate(sorted_docs):
                                top_file.write("\t".join(
                                    [qid, "Q0", doc_id, pid_score["pid"], str(rank + 1), str(pid_score["score"]),
                                     "BERT_top1_passage"]) + "\n")
        else:
            with tf.gfile.Open(predictions_path, 'r') as ref_file, \
                    tf.gfile.Open(
                        os.path.join(path, "{}_query_chunk_{}_kc-{}.tsv".format(FLAGS.dataset, split, FLAGS.kc)),
                        'w') as top_file:
                top_res = collections.OrderedDict()
                for line in ref_file:
                    qid, doc_id, pid, score = line.strip().split()
                    score = float(score)

                    if qid not in top_res:
                        top_res[qid] = dict()
                    top_res[qid][pid] = score

                for qid, pid_scores in top_res.items():
                    sorted_pids = sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)
                    top_pids = sorted_pids[:FLAGS.kc]
                    for rank, (pid, score) in enumerate(top_pids):
                        top_file.write("\t".join([qid, pid, str(rank + 1), str(score),
                                                  "BERT_top{0}_chunk".format(FLAGS.kc)]) + "\n")


if __name__ == "__main__":
    tf.flags.mark_flag_as_required("model_size")
    tf.flags.mark_flag_as_required("output_path")
    tf.flags.mark_flag_as_required("dataset")
    tf.app.run()
