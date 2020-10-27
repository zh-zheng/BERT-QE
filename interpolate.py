import argparse
import collections
import os
import numpy as np
import tensorflow as tf
from math import log


def interpolate(inter_run, main_run, output_path, rerank_num, **kwargs):
    """ Interpolate two runs. """
    main_run_dict = dict()
    with tf.gfile.Open(main_run, 'r') as f:
        for line in f:
            query_id, Q0, doc_id, rank, score, run_name = line.strip().split()
            if query_id not in main_run_dict:
                main_run_dict[query_id] = collections.OrderedDict()
            main_run_dict[query_id][doc_id] = float(score)
    for alpha in np.linspace(0, 1, 11):
        final_dict = collections.OrderedDict()
        with tf.gfile.Open(inter_run, 'r') as f:
            for line in f:
                query_id, Q0, doc_id, rank, score, run_name = line.strip().split()
                if query_id not in main_run_dict:
                    continue
                if query_id not in final_dict:
                    final_dict[query_id] = dict()
                if len(final_dict[query_id]) == rerank_num:
                    continue
                if len(kwargs) == 0:
                    final_dict[query_id][doc_id] = alpha * log(main_run_dict[query_id][doc_id]) + (1 - alpha) * float(
                        score)
                else:
                    # Note: switch the position of alpha according to the equations in the paper
                    final_dict[query_id][doc_id] = alpha * float(score) + (1 - alpha) * main_run_dict[query_id][doc_id]

        if not tf.gfile.Exists(output_path):
            tf.gfile.MakeDirs(output_path)

        if len(kwargs) == 0:
            output_file_path = os.path.join(output_path, "{}_inter_{:.2f}".format(main_run, alpha))
        else:
            output_file_path = os.path.join(output_path,
                                            "{}_expansion_maxp_{}_merge_{:.2f}.trec".format(kwargs["dataset"],
                                                                                            kwargs["split"], alpha))
        with tf.gfile.Open(output_file_path, 'w') as out:
            for query_id in final_dict:
                ranking_list = sorted(final_dict[query_id].items(), key=lambda x: x[1], reverse=True)

                last_score = ranking_list[-1][1]
                our_run_docs = list(main_run_dict[query_id].keys())
                for doc in our_run_docs[rerank_num:]:
                    current_score = last_score - 0.01
                    final_dict[query_id][doc] = current_score
                    last_score = current_score

                ranking_list = sorted(final_dict[query_id].items(), key=lambda x: x[1], reverse=True)

                for rank, (doc_id, score) in enumerate(ranking_list):
                    if len(kwargs) == 0:
                        run_name = "BERT-QE"
                    else:
                        run_name = "expansion_maxp_merge"
                    out.write("\t".join([query_id, "Q0", doc_id, str(rank + 1), str(score), run_name]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxp_path', type=str, required=True)
    parser.add_argument('--expansion_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--rerank_num', type=int, required=True)
    args = parser.parse_args()
    for split in ["valid", "test"]:
        maxp_run = os.path.join(args.maxp_path, "{0}_{1}_result.trec".format(args.dataset, split))
        expansion_run = os.path.join(args.expansion_path, "{0}_{1}_result.trec".format(args.dataset, split))

        interpolate(expansion_run, maxp_run, args.expansion_path, args.rerank_num, dataset=args.dataset, split=split)
