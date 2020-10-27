# -*- coding: utf-8 -*-
from bert import modeling
import os
import tensorflow as tf
from scipy.special import softmax
import time
import numpy as np
import collections
from utils import load_run
from config import config_dict
from functions import model_fn_builder, input_fn_builder

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "device", "0",
    "CUDA device number")

flags.DEFINE_string(
    "output_path", None,
    "output path"
)

flags.DEFINE_integer(
    'kc', None,
    'kc in the paper'
)

flags.DEFINE_string(
    'third_model_path', None,
    'path of the third model'
)

flags.DEFINE_integer(
    'batch_size', None,
    'batch size for training and evaluation'
)

flags.DEFINE_string(
    'tpu', None,
    'tpu address'
)

flags.DEFINE_string(
    'dataset', None,
    "dataset: robust04 or gov2"
)

flags.DEFINE_integer(
    'rerank_num', None,
    "the number of documents to be re-ranked"
)

flags.DEFINE_integer(
    'max_seq_length', 384,
    "max sequence length for BERT"
)

flags.DEFINE_string(
    'model_size', None,
    "BERT model size used in the current phase"
)

flags.DEFINE_string(
    'first_model_path', None,
    "first model path"
)

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device

init_checkpoint = None  # @param {type:"string"}
print('***** BERT Init Checkpoint: {} *****'.format(init_checkpoint))

# Parameters
use_tpu = False if FLAGS.tpu is None else True
iterations_per_loop = 500
num_tpu_cores = 8


def main(_):
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
        keep_checkpoint_max=1,
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
        predict_batch_size=FLAGS.batch_size,
        params={"qc_scores": "qc_scores"})

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.batch_size)

    for split in ["valid", "test"]:

        maxp_run = load_run(os.path.join(FLAGS.first_model_path, "{}_{}_result.trec".format(FLAGS.dataset, split)))

        query_docids_map = []
        data_path = os.path.join(FLAGS.output_path, "rerank-{0}_kc-{1}".format(FLAGS.rerank_num, FLAGS.kc), "data")
        result_path = os.path.join(FLAGS.output_path, "rerank-{0}_kc-{1}".format(FLAGS.rerank_num, FLAGS.kc), "result")
        if not tf.gfile.Exists(result_path):
            tf.gfile.MakeDirs(result_path)

        with tf.gfile.Open(os.path.join(data_path, "chunk_passage_ids_{0}.txt".format(split))) as ref_file:
            for line in ref_file:
                query_docids_map.append(line.strip().split("\t"))

        predict_input_fn = input_fn_builder(
            dataset_path=os.path.join(data_path, "chunk_passage_{0}.tf".format(split)),
            is_training=False,
            seq_length=FLAGS.max_seq_length,
            drop_remainder=False)

        total_count = 0

        result_file = tf.gfile.Open(os.path.join(result_path, "{0}_{1}_result.trec".format(FLAGS.dataset, split)), 'w')

        ckpt = tf.train.latest_checkpoint(checkpoint_dir=FLAGS.third_model_path)
        print("use latest ckpt: {0}".format(ckpt))

        result = estimator.predict(input_fn=predict_input_fn,
                                   yield_single_examples=True,
                                   checkpoint_path=ckpt)

        start_time = time.time()
        results = []
        result_dict = collections.OrderedDict()
        for item in result:

            results.append((item["qc_scores"], item["probs"]))
            total_count += 1

            if total_count == len(query_docids_map) or query_docids_map[total_count][0] != \
                    query_docids_map[total_count - 1][0]:

                chunk_num = len(results) // FLAGS.rerank_num
                assert chunk_num <= FLAGS.kc

                qc_scores, probs = list(zip(*results))
                qc_scores = np.stack(qc_scores)
                cp_scores = np.stack(probs)[:, 1]

                qc_scores = np.reshape(qc_scores, [FLAGS.rerank_num, chunk_num])
                cp_scores = np.reshape(cp_scores, [FLAGS.rerank_num, chunk_num])

                # softmax normalization
                qc_scores = softmax(qc_scores, axis=-1)

                scores = np.sum(np.multiply(qc_scores, cp_scores), axis=-1, keepdims=False)

                start_idx = total_count - FLAGS.rerank_num * chunk_num
                end_idx = total_count
                query_ids, chunk_ids, passage_ids, labels, qc_scores = zip(*query_docids_map[start_idx:end_idx])
                assert len(set(query_ids)) == 1, "Query ids must be all the same."
                query_id = query_ids[0]

                candidate_docs = list()
                for pid in passage_ids:
                    doc_id = pid.split("_")[0]
                    if doc_id not in candidate_docs:
                        candidate_docs.append(doc_id)

                result_dict[query_id] = dict()

                for i, doc in enumerate(candidate_docs):
                    result_dict[query_id][doc] = scores[i]

                rerank_list = sorted(result_dict[query_id].items(), key=lambda x: x[1], reverse=True)

                last_score = rerank_list[-1][1]
                for doc in maxp_run[query_id][FLAGS.rerank_num:]:
                    current_score = last_score - 0.01
                    result_dict[query_id][doc] = current_score
                    last_score = current_score

                ranking_list = sorted(result_dict[query_id].items(), key=lambda x: x[1], reverse=True)

                for rank, (doc_id, score) in enumerate(ranking_list):
                    result_file.write(
                        "\t".join([query_id, "Q0", doc_id, str(rank + 1), str(score), "chunk_passage_PRF"]) + "\n")

                results = []

            if total_count % 1000 == 0:
                tf.logging.warn("Read {} examples in {} secs".format(
                    total_count, int(time.time() - start_time)))

        result_file.close()
        tf.logging.info("Done Evaluating!")


if __name__ == "__main__":
    flags.mark_flag_as_required("model_size")
    flags.mark_flag_as_required("output_path")
    flags.mark_flag_as_required("kc")
    flags.mark_flag_as_required("third_model_path")
    flags.mark_flag_as_required('batch_size')
    flags.mark_flag_as_required('dataset')
    flags.mark_flag_as_required('rerank_num')
    flags.mark_flag_as_required('first_model_path')
    tf.app.run()
