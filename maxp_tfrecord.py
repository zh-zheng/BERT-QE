import os
import tensorflow as tf
from bert import tokenization
from utils import load_queries, load_qrels, load_two_columns_file

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "vocab", None,
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_path", None,
    "output path")

flags.DEFINE_string(
    "qrels", None,
    "Path to the query id / relevant doc ids pairs.")

flags.DEFINE_string(
    "queries", None,
    "Path to the queries file")

flags.DEFINE_integer(
    "max_query_length", 128,
    "The maximum query sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_integer(
    "max_passage_length", 256,
    "The maximum total sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_integer(
    "fold", None,
    "fold index"
)

flags.DEFINE_string(
    'dataset', None,
    "dataset: robust04 or gov2"
)

flags.DEFINE_string(
    'passage_path', None,
    "passage path"
)

assert FLAGS.dataset in ["robust04", "gov2"], "For now, we only support robust04 and GOV2 dataset!"


def convert_dataset(queries, passages, qrels, tokenizer, fold, split):
    """ Convert <query, passage> pairs to TFRecord. """
    main_path = os.path.join(FLAGS.output_path, "fold-" + str(fold))
    if not tf.gfile.Exists(main_path):
        tf.gfile.MakeDirs(main_path)
    out_query_passage = os.path.join(main_path, '{}_query_maxp_{}.tf'.format(FLAGS.dataset, split))
    with tf.python_io.TFRecordWriter(out_query_passage) as writer, \
            tf.gfile.Open(os.path.join(FLAGS.passage_path, "fold-" + str(fold),
                                       '{}_query_passage_{}_top1.tsv'.format(FLAGS.dataset, split)), 'r') as qp_file:
        for i, line in enumerate(qp_file):
            qid, Q0, doc_id, pid, rank, score, run_name = line.split("\t")
            query = queries[qid]

            query = tokenization.convert_to_unicode(query)
            query_tokens = tokenization.convert_to_bert_input(
                text=query,
                max_seq_length=FLAGS.max_query_length,
                tokenizer=tokenizer,
                add_cls=True,
                add_sep=True)

            query_token_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=query_tokens))

            passage_content = passages[pid]

            passage_tokens = tokenization.convert_to_bert_input(
                text=tokenization.convert_to_unicode(passage_content),
                max_seq_length=FLAGS.max_passage_length,
                tokenizer=tokenizer,
                add_cls=False,
                add_sep=True)

            passage_token_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=passage_tokens))

            label = 1 if doc_id in qrels[qid] else 0

            labels_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label]))

            features = tf.train.Features(feature={
                'query_token_ids': query_token_ids_tf,
                'piece_token_ids': passage_token_ids_tf,
                'label': labels_tf,
            })

            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

            if (i + 1) % 1000 == 0:
                print("process {} examples".format(i + 1))


def main(_):
    print('Loading Tokenizer...')
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab, do_lower_case=True)

    qrels = None
    if FLAGS.qrels:
        qrels = load_qrels(path=FLAGS.qrels)

    print('Loading Collection...')
    passage = load_two_columns_file(path=os.path.join(FLAGS.passage_path, "passage_id_text.txt"))

    for split in ["train", "valid", "test"]:
        queries = load_queries(path=FLAGS.queries, fold=FLAGS.fold, split=split, type="title", dataset=FLAGS.dataset)

        print('Converting to TFRecord...')
        convert_dataset(queries=queries, passages=passage, qrels=qrels, tokenizer=tokenizer, fold=FLAGS.fold,
                        split=split)

        print('{} done!'.format(split))


if __name__ == '__main__':
    flags.mark_flag_as_required('passage_path')
    flags.mark_flag_as_required('output_path')
    flags.mark_flag_as_required('vocab')
    flags.mark_flag_as_required('qrels')
    flags.mark_flag_as_required('queries')
    flags.mark_flag_as_required('fold')
    flags.mark_flag_as_required('dataset')
    tf.app.run(main)
