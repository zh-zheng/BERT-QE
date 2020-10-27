import tensorflow as tf
from bert import optimization, modeling


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probs = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return loss, probs


def model_fn_builder(bert_config, num_labels, init_checkpoint, use_tpu,
                     use_one_hot_embeddings, learning_rate=None,
                     num_train_steps=None, num_warmup_steps=None):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, probs = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        scaffold_fn = None
        initialized_variable_names = []
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:

            predictions_dict = {"probs": probs}
            if "qc_scores" in params:
                predictions_dict["qc_scores"] = features["qc_score"]

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions_dict,
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN, EVAL and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(dataset_path, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        def extract_fn(data_record):
            features = {
                "query_token_ids": tf.FixedLenSequenceFeature(
                    [], tf.int64, allow_missing=True),
                "piece_token_ids": tf.FixedLenSequenceFeature(
                    [], tf.int64, allow_missing=True),
                "label": tf.FixedLenFeature([], tf.int64)
            }

            if "qc_scores" in params:
                features["qc_score"] = tf.FixedLenFeature([], tf.float32)

            sample = tf.parse_single_example(data_record, features)

            a_token_ids = tf.cast(sample["query_token_ids"], tf.int32)
            b_token_ids = tf.cast(sample["piece_token_ids"], tf.int32)
            label_ids = tf.cast(sample["label"], tf.int32)

            input_ids = tf.concat((a_token_ids, b_token_ids), 0)

            a_segment_id = tf.zeros_like(a_token_ids)
            b_segment_id = tf.ones_like(b_token_ids)
            segment_ids = tf.concat((a_segment_id, b_segment_id), 0)

            input_mask = tf.ones_like(input_ids)

            features_dict = {
                "input_ids": input_ids,
                "segment_ids": segment_ids,
                "input_mask": input_mask,
                "label_ids": label_ids
            }

            if "qc_scores" in params:
                features_dict["qc_score"] = tf.cast(sample["qc_score"], tf.float32)

            return features_dict

        dataset = tf.data.TFRecordDataset([dataset_path])

        if is_training:
            dataset = dataset.shuffle(buffer_size=params["train_examples"], reshuffle_each_iteration=True,
                                      seed=1234).repeat(params["num_train_epochs"])

        dataset = dataset.map(extract_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        padded_shapes_dict = {
            "input_ids": [seq_length],
            "segment_ids": [seq_length],
            "input_mask": [seq_length],
            "label_ids": []
        }

        padding_values_dict = {
            "input_ids": 0,
            "segment_ids": 0,
            "input_mask": 0,
            "label_ids": 0
        }

        if "qc_scores" in params:
            padded_shapes_dict["qc_score"] = []
            padding_values_dict["qc_score"] = 0.0

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=padded_shapes_dict,
            padding_values=padding_values_dict,
            drop_remainder=drop_remainder)

        return dataset

    return input_fn
