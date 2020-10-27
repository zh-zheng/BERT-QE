import argparse
import os
import subprocess
from evaluation import evaluate_trec
from interpolate import interpolate

metrics = ['map', 'map_cut_100', 'P_20', 'ndcg_cut_20']


def generate_and_test(args):
    """Select the best result according to validation sets."""
    total_path = '{0}/total/rerank-{1}_kc-{2}/result/'.format(args.main_path, args.rerank_num, args.kc)
    if not os.path.exists(total_path):
        os.makedirs(total_path)
    for fold in range(1, 6):
        cur_path = '{0}/fold-{1}/rerank-{2}_kc-{3}/result/'.format(args.main_path, fold, args.rerank_num, args.kc)

        best_valid_metric = (-1.0, -1.0)
        best_test_res = None
        best_valid_res = None
        best_alpha = 0

        for alpha in range(1, 11):
            test_file = "{}_expansion_maxp_test_merge_{:.2f}.trec".format(args.dataset, 0.1 * alpha)
            valid_file = "{}_expansion_maxp_valid_merge_{:.2f}.trec".format(args.dataset, 0.1 * alpha)

            test_file_path = os.path.join(cur_path, test_file)
            valid_file_path = os.path.join(cur_path, valid_file)
            valid_metrics = evaluate_trec(args.qrels, valid_file_path, metrics)

            if valid_metrics["ndcg_cut_20"] > best_valid_metric[0] or (
                    abs(valid_metrics["ndcg_cut_20"] - best_valid_metric[0]) < 1e-6 and valid_metrics["map_cut_100"] >
                    best_valid_metric[1]):
                best_valid_metric = valid_metrics["ndcg_cut_20"], valid_metrics["map_cut_100"]
                best_test_res = test_file_path
                best_valid_res = valid_file_path
                best_alpha = 0.1 * alpha

        print("fold {}: best alpha: ".format(fold), best_alpha)

        # interpolate with initial ranking
        interpolate(args.run_file, best_valid_res, cur_path, args.rerank_num)
        interpolate(args.run_file, best_test_res, cur_path, args.rerank_num)
        best_inter_valid_metric = (-1.0, -1.0)
        best_inter_valid_res = None
        best_inter_test_res = None
        best_beta = -1

        for beta in range(1, 10):
            inter_valid_file = os.path.join(cur_path, "{}_inter_0.{}0".format(best_valid_res, beta))
            inter_test_file = os.path.join(cur_path, "{}_inter_0.{}0".format(best_test_res, beta))

            inter_valid_metrics = evaluate_trec(args.qrels, inter_valid_file, metrics)

            if inter_valid_metrics["ndcg_cut_20"] > best_inter_valid_metric[0] or (
                    abs(inter_valid_metrics["ndcg_cut_20"] - best_inter_valid_metric[0]) < 1e-6 and inter_valid_metrics[
                "map_cut_100"] >
                    best_inter_valid_metric[1]):
                best_inter_valid_metric = inter_valid_metrics["ndcg_cut_20"], inter_valid_metrics["map_cut_100"]
                best_inter_valid_res = inter_valid_file
                best_inter_test_res = inter_test_file
                best_beta = 0.1 * beta

        print("fold {}: best beta: ".format(fold), best_beta)
        subprocess.run(
            'cat {0} >> {1}'.format(best_inter_test_res,
                                    os.path.join(total_path, "final_test.trec")),
            shell=True)
        subprocess.run(
            'cat {0} >> {1}'.format(best_inter_valid_res,
                                    os.path.join(total_path, "final_valid.trec")),
            shell=True)

    final_res = os.path.join(total_path, "final_test.trec")

    evaluation_res = evaluate_trec(args.qrels, final_res, metrics)
    print(evaluation_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qrels', type=str, required=True)
    parser.add_argument('--run_file', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--rerank_num', type=int, required=True)
    parser.add_argument('--kc', type=int, required=True)
    parser.add_argument('--main_path', type=str, required=True)
    args = parser.parse_args()

    generate_and_test(args)
