import os
import tempfile
import tensorflow_ranking as tfr
import tensorflow as tf
import zipfile

from collect_features import log_features, build_features_judgments_file
from load_features import init_default_store, load_features
from log_conf import Logger
from utils import elastic_connection, ES_HOST, ES_AUTH, JUDGMENTS_FILE, INDEX_NAME, JUDGMENTS_FILE_FEATURES, \
    FEATURE_SET_NAME, RANKLIB_JAR


def train_model(num_features, judgments_with_features_file):
    loss_name = 'pairwise_logistic_loss'
    list_size = 100
    batch_size = 32
    hidden_layer_dims = [10, 4]
    activation = "tanh"

    def input_fn(path):
        return (
            tf.data.Dataset.from_generator(
                tfr.data.libsvm_generator(path, num_features, list_size),
                output_types=(
                    {str(k): tf.float32 for k in range(1, num_features + 1)},
                    tf.float32),
                output_shapes=(
                    {str(k): tf.TensorShape([list_size, 1])
                     for k in range(1, num_features + 1)},
                    tf.TensorShape([list_size])))
            .shuffle(1000)
            .repeat()
            .batch(batch_size)
            .make_one_shot_iterator()
            .get_next()
        )

    def example_feature_columns():
        feature_names = ["%d" % (i + 1) for i in range(0, num_features)]
        return {
            name: tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
            for name in feature_names
        }

    def make_score_fn():
        def _score_fn(context_features, group_features, mode, params, config):
            if mode == tf.contrib.learn.ModeKeys.INFER:
                input_layer = group_features['input']
            else:
                example_input = [
                    tf.layers.flatten(group_features[name])
                    for name in sorted(example_feature_columns())
                ]
                input_layer = tf.concat(example_input, 1)
            cur_layer = input_layer
            for i, layer_width in enumerate(int(d) for d in hidden_layer_dims):
                cur_layer = tf.layers.dense(
                    cur_layer,
                    units=layer_width,
                    activation=activation)
            logits = tf.layers.dense(cur_layer, units=1)
            return logits
        return _score_fn

    def eval_metric_fns():
        return {
            "metric/ndcg@{}".format(topn): tfr.metrics.make_ranking_metric_fn(
                tfr.metrics.RankingMetricKey.NDCG, topn=topn)
            for topn in [1, 3, 5, 10]
        }

    def fix_export_outputs(ranking_head):
        # The default creates a RegressionOutput, which requires a string
        # input at serving time. Replace so we can input floats directly.
        orig_create_estimator_spec = ranking_head.create_estimator_spec

        def create_estimator_spec(features, mode, logits, labels=None, regularization_losses=None):
            if mode != tf.estimator.ModeKeys.PREDICT:
                return orig_create_estimator_spec(features, mode, logits, labels, regularization_losses)
            logits = tf.convert_to_tensor(logits)
            with tf.name_scope(ranking_head._name, 'head'):
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=logits,
                    export_outputs={
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        tf.estimator.export.PredictOutput(logits),
                    })

        ranking_head.create_estimator_spec = create_estimator_spec

    def get_estimator(hparams):
        def _train_op_fn(loss):
            return tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=hparams.learning_rate,
                optimizer='Adagrad')

        ranking_head = tfr.head.create_ranking_head(
            loss_fn=tfr.losses.make_loss_fn(loss_name),
            eval_metric_fns=eval_metric_fns(),
            train_op_fn=_train_op_fn)
        fix_export_outputs(ranking_head)

        return tf.estimator.Estimator(
            model_fn=tfr.model.make_groupwise_ranking_fn(
                group_score_fn=make_score_fn(),
                group_size=1,
                transform_fn=None,
                ranking_head=ranking_head),
            params=hparams)

    hparams = tf.contrib.training.HParams(learning_rate=0.05)
    ranker = get_estimator(hparams)
    ranker.train(input_fn=lambda: input_fn(judgments_with_features_file), steps=100)
    return ranker


def zip_dir(dir_base):
    with tempfile.TemporaryFile() as f:
        zipf = zipfile.ZipFile(f, mode='w')
        import pdb; pdb.set_trace()
        for root, dirs, files in os.walk(dir_base):
            for file in dirs + files:
                arcname = os.path.join(root[len(dir_base):], file)
                zipf.write(os.path.join(root, file), arcname)
        zipf.close()
        f.flush()
        f.seek(0)
        return f.read()


def save_model(num_features, script_name, feature_set, ranker):
    """ Save the ranklib model in Elasticsearch """
    import base64
    import requests
    import json
    from urllib.parse import urljoin
    import shutil

    inputs = tf.placeholder(
        dtype=tf.float32,
        shape=[None, None, num_features])
    export_dir_base = None
    try:
        export_dir_base = tempfile.mkdtemp()
        export_dir_bytes = ranker.export_saved_model(
            export_dir_base,
            tf.estimator.export.build_raw_serving_input_receiver_fn({
                'input': inputs
            }))
        # why bytes?
        export_dir = export_dir_bytes.decode('utf8')
        # Zip up the output
        model_bytes = zip_dir(export_dir)
    finally:
        if export_dir_base is not None:
            shutil.rmtree(export_dir_base)

    model_payload = {
        "model": {
            "name": script_name,
            "model": {
                "type": "model/tensorflow",
                "definition": base64.b64encode(model_bytes).decode('ascii'),
            }
        }
    }

    path = "_ltr/_featureset/%s/_createmodel" % feature_set
    full_path = urljoin(ES_HOST, path)
    Logger.logger.info("POST %s" % full_path)
    head = {'Content-Type': 'application/json'}
    resp = requests.post(full_path, data=json.dumps(model_payload), headers=head, auth=ES_AUTH)
    Logger.logger.info(resp.status_code)
    if resp.status_code >= 300:
        Logger.logger.error(resp.text)


if __name__ == "__main__":
    from judgments import judgments_from_file, judgments_by_qid
    num_features = 3

    es = elastic_connection(timeout=1001)
    # Load features into Elasticsearch
    init_default_store()
    load_features(FEATURE_SET_NAME)
    # Parse a judgments
    movieJudgments = judgments_by_qid(judgments_from_file(filename=JUDGMENTS_FILE))
    # Use proposed Elasticsearch queries (1.json.jinja ... N.json.jinja) to generate a training set
    # output as "sample_judgments_wfeatures.txt"
    log_features(es, judgments_dict=movieJudgments, search_index=INDEX_NAME)
    build_features_judgments_file(movieJudgments, filename=JUDGMENTS_FILE_FEATURES)
    # Train each ranklib model type
    tf.logging.set_verbosity(tf.logging.INFO)
    ranker = train_model(
        num_features=num_features,
        judgments_with_features_file=JUDGMENTS_FILE_FEATURES)
    modelType = 'tf_ranking'
    save_model(
        num_features=num_features,
        script_name="test_%s" % modelType,
        feature_set=FEATURE_SET_NAME,
        ranker=ranker)
