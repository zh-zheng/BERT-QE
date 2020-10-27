import re
import sys
import subprocess
from config import trec_eval_script_path
from scipy import stats
from collections import OrderedDict


def run(command, get_ouput=False):
    try:
        if get_ouput:
            process = subprocess.Popen(command, stdout=subprocess.PIPE)
            output, err = process.communicate()
            return output
        else:
            subprocess.call(command)
    except subprocess.CalledProcessError as e:
        print(e)


def evaluate_trec(qrels, res, metrics):
    """Calculate TREC metrics."""
    command = [trec_eval_script_path, '-m', 'all_trec', '-M', '1000', qrels, res]
    output = run(command, get_ouput=True)
    output = str(output, encoding='utf-8')

    metrics_dict = OrderedDict()
    for metric in metrics:
        metrics_dict[metric] = float(re.findall(r'{0}\s+all.+\d+'.format(metric), output)[0].split('\t')[2].strip())

    return metrics_dict


def evaluate_trec_perquery(qrels, res, metrics):
    """Calculate TREC metrics per query."""
    command = [trec_eval_script_path, '-m', 'all_trec', '-q', '-M', '1000', qrels, res]
    output = run(command, get_ouput=True)
    output = str(output, encoding='utf-8')

    metrics_val = []
    for metric in metrics:
        cur_res = re.findall(r'{0}\s+\t\d+.+\d+'.format(metric), output)
        cur_res = list(map(lambda x: float(x.split('\t')[-1]), cur_res))
        metrics_val.append(cur_res)

    return OrderedDict(zip(metrics, metrics_val))


def tt_test(qrels, res1, res2, metrics):
    """ Perform a paired two-tailed t-test. """
    met_dict1 = evaluate_trec_perquery(qrels, res1, metrics)
    met_dict2 = evaluate_trec_perquery(qrels, res2, metrics)

    avg_met_dict1 = evaluate_trec(qrels, res1, metrics)
    avg_met_dict2 = evaluate_trec(qrels, res2, metrics)
    print(avg_met_dict1)
    print(avg_met_dict2)

    test_dict = OrderedDict()
    for met in met_dict1.keys():
        p_value = stats.ttest_rel(met_dict1.get(met), met_dict2.get(met))[1]
        test_dict.update({met: p_value})

    return test_dict


if __name__ == '__main__':
    argv = sys.argv
    print(tt_test(argv[1], argv[2], argv[3], ['map', 'map_cut_100', 'P_20', 'ndcg_cut_20']))
