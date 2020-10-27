# -*- coding: utf-8 -*-
from bert import modeling
import os
import tensorflow as tf
import time
import numpy as np
import collections
from functions import model_fn_builder, input_fn_builder
from config import checkpoint_dict, config_dict

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "device", "0",
    "CUDA device number")

flags.DEFINE_string(
    "output_path", None,
    "output path"
)

flags.DEFINE_string(
    "data_path", None,
    'data path'
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
    'max_seq_length', 384,
    "max sequence length for BERT"
)

flags.DEFINE_string(
    'model_size', None,
    'BERT model size in the current phase'
)

flags.DEFINE_string(
    'passage_path', None,
    'passage path'
)

flags.DEFINE_integer(
    'batch_size', 32,
    'batch size'
)

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device

init_checkpoint = checkpoint_dict[FLAGS.model_size]  # @param {type:"string"}
print('***** BERT Init Checkpoint: {} *****'.format(init_checkpoint))

assert FLAGS.dataset in ["robust04", "gov2"], "For now, we only support robust04 and GOV2 dataset!"

# Parameters
use_tpu = False if FLAGS.tpu is None else True
do_train = True  # Whether to run training.
do_eval = True  # Whether to run evaluation.
train_batch_size = FLAGS.batch_size
eval_batch_size = FLAGS.batch_size
learning_rate = 1e-6
if FLAGS.dataset == 'robust04':
    train_examples = 150 * 1000
    eval_examples = 50 * 1000
else:
    train_examples = 90 * 1000
    eval_examples = 30 * 1000

num_train_epochs = 2
num_train_steps = int(train_examples * num_train_epochs // train_batch_size)  # must divided by batch size!
num_warmup_steps = int(num_train_steps * 0.1)
save_checkpoints_steps = 1000
iterations_per_loop = 1000
num_tpu_cores = 8

np.set_printoptions(threshold=np.inf)


def main(_):
    if not tf.gfile.Exists(FLAGS.output_path):
        tf.gfile.MakeDirs(FLAGS.output_path)

    if not do_train and not do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

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
        save_checkpoints_steps=save_checkpoints_steps,
        model_dir=FLAGS.output_path,
        keep_checkpoint_max=5,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=2,
        init_checkpoint=init_checkpoint,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_tpu,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=eval_batch_size,
        params={"train_examples": train_examples,
                "num_train_epochs": num_train_epochs})

    try:
        if do_train:
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Batch size = %d", train_batch_size)
            tf.logging.info("  Num steps = %d", num_train_steps)
            train_input_fn = input_fn_builder(
                dataset_path=os.path.join(FLAGS.data_path, "{}_query_maxp_train.tf".format(FLAGS.dataset)),
                seq_length=FLAGS.max_seq_length,
                is_training=True,
                drop_remainder=True)

            current_step = 0
            steps_per_epoch = train_examples // train_batch_size
            tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                            ' step %d.',
                            num_train_steps,
                            num_train_steps / steps_per_epoch,
                            current_step)

            start_timestamp = time.time()

            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

            elapsed_time = int(time.time() - start_timestamp)
            tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                            num_train_steps, elapsed_time)

    except KeyboardInterrupt:
        pass

    tf.logging.info("Done Training!")

    if do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", eval_batch_size)

        for split in ["valid", "test"]:
            query_docids_map = []
            with tf.gfile.Open(os.path.join(FLAGS.passage_path,
                                            "{0}_query_passage_{1}_top1.tsv".format(FLAGS.dataset, split))) as ref_file:
                for line in ref_file:
                    query_docids_map.append(line.strip().split("\t"))

            eval_input_fn = input_fn_builder(
                dataset_path=os.path.join(FLAGS.data_path, "{0}_query_maxp_{1}.tf".format(FLAGS.dataset, split)),
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=False)

            total_count = 0
            tsv_file_path = os.path.join(FLAGS.output_path, "{0}_{1}_result.tsv".format(FLAGS.dataset, split))
            trec_file_path = os.path.join(FLAGS.output_path, "{0}_{1}_result.trec".format(FLAGS.dataset, split))

            result = estimator.predict(input_fn=eval_input_fn,
                                       yield_single_examples=True)

            start_time = time.time()
            results = []
            result_dict = collections.OrderedDict()
            with tf.gfile.Open(tsv_file_path, 'w') as tsv_file, tf.gfile.Open(trec_file_path, 'w') as trec_file:
                for item in result:

                    results.append(item["probs"])
                    total_count += 1

                    if total_count == len(query_docids_map) or query_docids_map[total_count][0] != \
                            query_docids_map[total_count - 1][0]:

                        candidate_doc_num = len(results)

                        probs = np.stack(results)
                        results = probs[:, 1]

                        start_idx = total_count - candidate_doc_num
                        end_idx = total_count
                        query_ids, _, doc_ids, passage_ids, rank, _, _ = zip(*query_docids_map[start_idx:end_idx])
                        assert len(set(query_ids)) == 1, "Query ids must be all the same."
                        query_id = query_ids[0]

                        result_dict[query_id] = dict()

                        for i, doc in enumerate(doc_ids):
                            result_dict[query_id][doc] = (passage_ids[i], results[i])

                        ranking_list = sorted(result_dict[query_id].items(), key=lambda x: x[1][1], reverse=True)
                        for rank, (doc_id, (pid, score)) in enumerate(ranking_list):
                            tsv_file.write("\t".join(
                                [query_id, "Q0", doc_id, pid, str(rank + 1), str(score), "maxp_finetune"]) + "\n")
                            trec_file.write(
                                "\t".join([query_id, "Q0", doc_id, str(rank + 1), str(score), "maxp_finetune"]) + "\n")

                        results = []

                    if total_count % 1000 == 0:
                        tf.logging.info("Read {} examples in {} secs".format(
                            total_count, int(time.time() - start_time)))

                tf.logging.info("Done Evaluating!")


if __name__ == "__main__":
    flags.mark_flag_as_required("passage_path")
    flags.mark_flag_as_required("model_size")
    flags.mark_flag_as_required("output_path")
    flags.mark_flag_as_required("data_path")
    flags.mark_flag_as_required("dataset")
    tf.app.run()
