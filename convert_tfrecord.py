# -*- coding: utf-8 -*-
import collections
import os
import tensorflow as tf
from bert import tokenization
from utils import load_queries, load_run, load_qrels, load_two_columns_file, load_collection, get_pieces

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_path", None,
    "Folder where the TFRecord files will be writen.")

flags.DEFINE_string(
    "vocab", None,
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "collection_file", None,
    "Path to the collection file.")

flags.DEFINE_string(
    "queries", None,
    "Path to the <query id; query text> pairs.")

flags.DEFINE_string(
    "run_file", None,
    "Path to the initial ranking.")

flags.DEFINE_string(
    "first_model_path", None,
    "Path to the output of the first model")

flags.DEFINE_string(
    "qrels", None,
    "Path to the query id / relevant doc ids pairs.")

flags.DEFINE_integer(
    "window_size", 100,
    "The sliding window size before WordPiece tokenization.")

flags.DEFINE_integer(
    "stride", 50,
    "The stride size before WordPiece tokenization."
)

flags.DEFINE_integer(
    "max_title_length", 30,
    "The maximum title sequence length before WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_integer(
    "max_query_length", 128,
    "The maximum query sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_integer(
    "doc_depth", 1000,
    "The number of docs per query.")

flags.DEFINE_integer(
    "max_passage_length", 256,
    "The maximum length of a passage after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_string(
    'dataset', None,
    'dataset: robust04 or gov2'
)

flags.DEFINE_string(
    "task", None,
    "current task: passage or chunk"
)

flags.DEFINE_integer(
    'fold', None,
    'fold index'
)

assert FLAGS.task in ["passage", "chunk"], "task must be 'passage' or 'chunk'!"
assert FLAGS.dataset in ["robust04", "gov2"], "For now, we only support robust04 and GOV2 dataset!"

if FLAGS.task == "passage":
    assert FLAGS.run_file is not None, \
        "--run_file must be provided when running the 'passage' task, as we need the initial ranking!"
else:
    assert FLAGS.fold is not None, \
        "'--fold' must be provided for cross-validation when running the 'chunk' task!"
    assert FLAGS.first_model_path is not None, \
        "'--first_model_path' must be provided to get the ranking result of phase 1 when running the 'chunk' task!"


def convert_dataset(main_path, data, collection, tokenizer, split=""):
    """ Split a document into passages/chunks and convert <query, passage/chunk> pairs to TFRecord."""
    suffix = ""
    if split != "":
        suffix = "_" + split

    if not tf.gfile.Exists(main_path):
        tf.gfile.MakeDirs(main_path)
    id_file = tf.gfile.Open(os.path.join(main_path, 'query_{}_ids{}.txt'.format(FLAGS.task, suffix)), 'w')
    text_file = tf.gfile.Open(os.path.join(main_path, '{}_id_text{}.txt').format(FLAGS.task, suffix), 'w')
    out_tf_path = os.path.join(main_path, 'query_{}{}.tf'.format(FLAGS.task, suffix))
    id_set = set()
    with tf.python_io.TFRecordWriter(out_tf_path) as writer:
        for i, query_id in enumerate(data):
            query, qrels, doc_ids = data[query_id]

            query = tokenization.convert_to_unicode(query)

            query_tokens = tokenization.convert_to_bert_input(
                text=query,
                max_seq_length=FLAGS.max_query_length,
                tokenizer=tokenizer,
                add_cls=True,
                add_sep=True)

            query_token_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=query_tokens))

            # here doc_depth is the top_docs_num in chunk file
            doc_ids = doc_ids[:FLAGS.doc_depth]

            if i + 1 % 1000 == 0:
                print("process {} queries".format(i))

            for doc_id in doc_ids:

                title = None
                if FLAGS.dataset == 'robust04' and FLAGS.task == 'passage':
                    title, body = collection[doc_id].split("\t")
                    title = " ".join(title.split(" ")[:FLAGS.max_title_length]).strip()  # truncate title
                    if title == '' or title == '.':  # if title is invalid
                        title = None
                else:
                    body = collection[doc_id]

                pieces = get_pieces(body, FLAGS.window_size, FLAGS.stride)

                for j, piece in enumerate(pieces):
                    piece_id = doc_id + "_{}-{}".format(FLAGS.task, j)

                    id_file.write('{}\t{}\n'.format(query_id, piece_id))

                    if title:
                        piece = title + ' ' + piece

                    if FLAGS.task == "passage":
                        max_piece_length = FLAGS.max_passage_length
                    else:
                        max_piece_length = FLAGS.max_query_length

                    piece_tokens = tokenization.convert_to_bert_input(
                        text=tokenization.convert_to_unicode(piece),
                        max_seq_length=max_piece_length,
                        tokenizer=tokenizer,
                        add_cls=False,
                        add_sep=True)

                    if piece_id not in id_set:
                        id_set.add(piece_id)
                        text_file.write(
                            piece_id + "\t" + piece + "\n")

                    piece_token_ids_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=piece_tokens))

                    labels_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[0]))  # fake label

                    features = tf.train.Features(feature={
                        'query_token_ids': query_token_ids_tf,
                        'label': labels_tf,
                        'piece_token_ids': piece_token_ids_tf
                    })
                    example = tf.train.Example(features=features)
                    writer.write(example.SerializeToString())

            if i % 1000 == 0:
                print('wrote {} of {} queries'.format(i, len(data)))

    id_file.close()
    text_file.close()


def merge(qrels, run, queries):
    """Merge qrels and runs into a single dict of key: query,
    value: tuple(relevant_doc_ids, candidate_doc_ids)"""
    data = collections.OrderedDict()
    for query_id, candidate_doc_ids in run.items():
        try:
            query = queries[query_id]
        except KeyError:
            continue
        relevant_doc_ids = set()
        if qrels:
            relevant_doc_ids = qrels[query_id]
        data[query_id] = (query, relevant_doc_ids, candidate_doc_ids)
    return data


def main(_):
    print('Loading Tokenizer...')
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab, do_lower_case=True)

    if not tf.gfile.Exists(FLAGS.output_path):
        tf.gfile.MakeDirs(FLAGS.output_path)

    qrels = None
    if FLAGS.qrels:
        qrels = load_qrels(path=FLAGS.qrels)

    if FLAGS.task == "passage":
        run = load_run(path=FLAGS.run_file)

        useful_docids = set()
        for ids in run.values():
            for docid in ids:
                useful_docids.add(docid)

        queries = load_queries(path=FLAGS.queries, type="title", dataset=FLAGS.dataset)
        data = merge(qrels=qrels, run=run, queries=queries)

        print('Loading Collection...')
        collection = load_collection(FLAGS.collection_file, FLAGS.dataset, useful_docids)

        print("queries_num:{}".format(len(queries)))

        print('Converting to TFRecord...')
        convert_dataset(main_path=FLAGS.output_path, data=data, collection=collection, tokenizer=tokenizer)
    else:
        for split in ["valid", "test"]:
            run_file = os.path.join(FLAGS.first_model_path, "{0}_{1}_result.tsv".format(FLAGS.dataset, split))

            run = load_run(path=run_file, has_pid=True, return_pid=True)
            queries = load_queries(path=FLAGS.queries, type="title", dataset=FLAGS.dataset, fold=FLAGS.fold,
                                   split=split)
            data = merge(qrels=qrels, run=run, queries=queries)

            print('Loading Collection...')
            collection = load_two_columns_file(FLAGS.collection_file)

            print("queries_num:{}".format(len(queries)))

            print('Converting to TFRecord...')
            convert_dataset(main_path=os.path.join(FLAGS.output_path, "fold-" + str(FLAGS.fold)), data=data,
                            collection=collection, tokenizer=tokenizer, split=split)

    print('done!')


if __name__ == '__main__':
    flags.mark_flag_as_required('output_path')
    flags.mark_flag_as_required('collection_file')
    flags.mark_flag_as_required('vocab')
    flags.mark_flag_as_required('queries')
    flags.mark_flag_as_required('qrels')
    flags.mark_flag_as_required('dataset')
    flags.mark_flag_as_required('task')
    tf.app.run(main)
