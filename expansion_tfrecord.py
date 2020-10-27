import collections
import os
import tensorflow as tf
from bert import tokenization
import random
from utils import load_queries, load_qrels, load_two_columns_file

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "vocab", None,
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_path", None,
    "output path"
)

flags.DEFINE_string(
    "qrels", None,
    "Path to the query id / relevant doc ids pairs.")

flags.DEFINE_integer(
    "max_query_length", 128,
    "The maximum query sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_integer(
    "max_passage_length", 256,
    "The maximum total sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_string(
    "queries", None,
    "Path to the queries file")

flags.DEFINE_integer(
    "fold", None,
    "fold index"
)

flags.DEFINE_integer(
    'rerank_num', None,
    "the number of documents to be re-ranked"
)

flags.DEFINE_integer(
    'kc', None,
    "kc in the paper"
)

flags.DEFINE_string(
    'dataset', None,
    "dataset: robust04 or gov2"
)

flags.DEFINE_string(
    'first_model_path', None,
    'first model path'
)

flags.DEFINE_string(
    'passage_path', None,
    'passage path'
)


def convert_dataset(data, passages, chunks, qc_scores, tokenizer, fold, split):
    """ Convert <chunk, passage> pairs to TFRecord."""
    output_path = os.path.join(FLAGS.output_path, "fold-" + str(fold),
                               "rerank-{0}_kc-{1}".format(FLAGS.rerank_num, FLAGS.kc), "data")
    if not tf.gfile.Exists(output_path):
        tf.gfile.MakeDirs(output_path)

    out_chunk_passage = os.path.join(output_path, 'chunk_passage_{0}.tf'.format(split))
    with tf.python_io.TFRecordWriter(out_chunk_passage) as writer, \
            tf.gfile.Open(os.path.join(output_path, 'chunk_passage_ids_{0}.txt'.format(split)),
                          'w') as chunk_passage_ids_file:
        qids = list(data.keys())
        if split == "train":
            random.shuffle(qids)
        for i, query_id in enumerate(qids):
            query, chunk_id_list, passage_ids, labels = data[query_id]
            pid_labels = list(zip(passage_ids, labels))
            pid_labels = pid_labels[:FLAGS.rerank_num]

            for pid, label in pid_labels:
                p_content = passages[pid]

                passage_tokens = tokenization.convert_to_bert_input(
                    text=tokenization.convert_to_unicode(p_content),
                    max_seq_length=FLAGS.max_passage_length,
                    tokenizer=tokenizer,
                    add_cls=False,
                    add_sep=True)

                passage_token_ids_tf = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=passage_tokens))

                labels_tf = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label]))

                for chunk_id in chunk_id_list:
                    chunk_content = chunks[chunk_id]
                    qc_score = qc_scores[query_id][chunk_id]
                    query_tokens = tokenization.convert_to_bert_input(
                        text=tokenization.convert_to_unicode(chunk_content),
                        max_seq_length=FLAGS.max_query_length,
                        tokenizer=tokenizer,
                        add_cls=True,
                        add_sep=True)

                    query_token_ids_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=query_tokens))

                    qc_score_tf = tf.train.Feature(
                        float_list=tf.train.FloatList(value=[qc_score])
                    )

                    chunk_passage_ids_file.write(
                        query_id + "\t" + chunk_id + "\t" + pid + "\t" + str(label) + "\t" + str(qc_score) + "\n")

                    features = tf.train.Features(feature={
                        'query_token_ids': query_token_ids_tf,
                        'piece_token_ids': passage_token_ids_tf,
                        'label': labels_tf,
                        'qc_score': qc_score_tf
                    })
                    example = tf.train.Example(features=features)
                    writer.write(example.SerializeToString())


def load_query_chunk_score(path):
    """ Load scores of <query, chunk> pairs."""
    qc = collections.OrderedDict()
    with tf.gfile.Open(path) as f:
        for line in f:
            query_id, chunk_id, rank, score, run_name = line.split("\t")
            if query_id not in qc:
                qc[query_id] = collections.OrderedDict()
            qc[query_id][chunk_id] = float(score)

    return qc


def load_query_passage(path):
    """ Load <query, passage> pairs. """
    qp = collections.OrderedDict()
    with tf.gfile.Open(path) as f:
        for line in f:
            query_id, Q0, doc_id, passage_id, rank, score, run_name = line.split("\t")
            if query_id not in qp:
                qp[query_id] = list()
            qp[query_id].append(passage_id)

    return qp


def merge(queries, qp, qc, qrels):
    """ Merge queries, qrels, <query, passage> pairs, <query, chunk> pairs into a single dict. """
    data = collections.OrderedDict()

    for qid in qc:
        passage_ids = list()
        labels = list()
        for passage_id in qp[qid]:
            doc_id = passage_id.split("_")[0]
            label = 0
            if doc_id in qrels[qid]:  # leave unjudged documents as non-relevant
                label = 1
            passage_ids.append(passage_id)
            labels.append(label)
        assert len(passage_ids) == len(labels)

        chunk_id_list = list(qc[qid].keys())
        data[qid] = (queries[qid], chunk_id_list, passage_ids, labels)

    return data


def main(_):
    print('Loading Tokenizer...')
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab, do_lower_case=True)

    qrels = None
    if FLAGS.qrels:
        qrels = load_qrels(path=FLAGS.qrels)

    split_list = ["valid", "test"]

    print('Loading Collection...')
    passage = load_two_columns_file(path=os.path.join(FLAGS.passage_path, "passage_id_text.txt"))

    for split in split_list:
        chunk = load_two_columns_file(
            path=os.path.join(FLAGS.output_path, "fold-" + str(FLAGS.fold), "chunk_id_text_{}.txt".format(split)))

        qp = load_query_passage(
            path=os.path.join(FLAGS.first_model_path, "{0}_{1}_result.tsv".format(FLAGS.dataset, split)))

        qc = load_query_chunk_score(path=os.path.join(FLAGS.output_path, "fold-" + str(FLAGS.fold),
                                                      "{0}_query_chunk_{1}_kc-{2}.tsv".format(FLAGS.dataset, split,
                                                                                              FLAGS.kc)))

        queries = load_queries(path=FLAGS.queries, fold=FLAGS.fold, split=split, type="title", dataset=FLAGS.dataset)
        data = merge(queries, qp, qc, qrels)

        print('Converting to TFRecord...')
        convert_dataset(data=data, passages=passage, chunks=chunk, qc_scores=qc, tokenizer=tokenizer, fold=FLAGS.fold,
                        split=split)

        print('{} done!'.format(split))


if __name__ == '__main__':
    flags.mark_flag_as_required('passage_path')
    flags.mark_flag_as_required('output_path')
    flags.mark_flag_as_required('first_model_path')
    flags.mark_flag_as_required('vocab')
    flags.mark_flag_as_required('qrels')
    flags.mark_flag_as_required('queries')
    flags.mark_flag_as_required('fold')
    flags.mark_flag_as_required('rerank_num')
    flags.mark_flag_as_required('kc')
    tf.app.run(main)
