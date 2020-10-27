import json
import tensorflow as tf
import collections
from config import cv_folder_path
import os


def load_queries(path, type, dataset, fold=None, split=None):
    """Load queries into a dict of key: query_id, value: query text."""

    qid_list = list()
    if fold and split:
        qid_list = load_qid_from_cv(dataset, fold, split)

    queries = {}
    with tf.gfile.Open(path) as f:
        for i, line in enumerate(f):
            content = json.loads(line)
            query_id = content['qid']
            if qid_list and query_id not in qid_list:
                continue
            if type != 'desc' and type != 'title' and type != 'narr':
                raise KeyError("query type must be desc, title or narr")
            query = content[type]
            queries[query_id] = query
            if i % 1000 == 0:
                print('Loading queries {}'.format(i))
    return queries


def load_qid_from_cv(dataset, fold, split):
    """ Load query_id in a cross-validation partition. """
    qid_list = list()

    assert dataset in ["robust04", "gov2"], "For now, we only support robust04 and GOV2 dataset!"

    cv_path = cv_folder_path[dataset]

    with tf.gfile.Open(os.path.join(cv_path, str(fold), "{}_query".format(split)), "r") as cv_file:
        for line in cv_file:
            qid_list.append(line.strip())

    return qid_list


def load_run(path, has_pid=False, return_pid=False):  # TREC format run (provided)
    """Load run into a dict of key: query_id, value: list of candidate doc ids."""

    run = collections.OrderedDict()
    with tf.gfile.Open(path) as f:
        for i, line in enumerate(f):
            if not has_pid:
                query_id, _, doc_id, rank, score, _ = line.strip().split()
            else:
                query_id, _, doc_id, pid, rank, score, _ = line.strip().split()
            if query_id not in run:
                run[query_id] = []
            if return_pid:
                run[query_id].append(pid)
            else:
                run[query_id].append(doc_id)
            if i % 1000 == 0:
                print('Loading run {}'.format(i))

    return run


def load_qrels(path):
    """Load qrels into a dict of key: query_id, value: list of relevant doc ids."""
    qrels = collections.defaultdict(set)
    with tf.gfile.Open(path) as f:
        for i, line in enumerate(f):
            query_id, _, doc_id, relevance = line.rstrip().split()
            if int(relevance) >= 1:
                qrels[query_id].add(doc_id)
            if i % 1000 == 0:
                print('Loading qrels {}'.format(i))
    return qrels


def load_two_columns_file(path):
    """Load tsv collection into a dict of key: doc id, value: doc text."""
    collection = {}
    with tf.gfile.Open(path) as f:
        for i, line in enumerate(f):
            id, text = line.split('\t')
            collection[id] = text.strip()

    return collection


def get_pieces(text, plen, overlap):
    """ Split a document into text pieces."""
    words = text.split(' ')
    s, e = 0, 0
    chunks = []
    while s < len(words):
        e = s + plen
        if len(words) - e < overlap:
            e = len(words)
        p = ' '.join(words[s:e])
        chunks.append(p)
        if e == len(words):
            break
        s = s + plen - overlap
    return chunks


def load_collection(path, dataset, useful_docids):
    """ Load Robust04 or GOV2 collections into a dict. """
    collection = {}

    assert dataset in ["robust04", "gov2"], "For now, we only support robust04 and GOV2 dataset!"

    with tf.gfile.Open(path) as f:
        for i, line in enumerate(f):
            if dataset == 'robust04':
                doc_id, title, doc_text = line.split('\t')
                if doc_id not in useful_docids:
                    continue
                collection[doc_id] = title + "\t" + doc_text.replace('\n', ' ').strip()
            else:
                doc_id, doc_text = line.split('\t')
                if doc_id not in useful_docids:
                    continue
                collection[doc_id] = doc_text.replace('\n', ' ').strip()

    return collection
