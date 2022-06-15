# -*-coding:utf-8 -*-
import argparse
import tensorflow as tf
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import DeepFM, xDeepFM, DCNMix, DeepFEFM, DIFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from utils.json_utils import read_json


def parse_argvs():
    parser = argparse.ArgumentParser(description='[CTR]')
    parser.add_argument("--topic", type=str, default="movie")
    parser.add_argument("--train_data", type=str, default="./train_data/movie_sample.csv")
    parser.add_argument("--train_data_sep", type=str, default=",")
    parser.add_argument("--params_file", type=str, default="./params/movie_params.json")
    parser.add_argument("--model", type=str, default="DeepFM", choices=["xDeepFM", "DeepFM", "DCNMix", "DeepFEFM", "DIFM"])
    parser.add_argument("--monitor", type=str, default="val_accuracy", choices=["val_accuracy", "val_auc"])
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dnn_dropout", type=float, default=0.2)
    parser.add_argument("--dnn_use_bn", type=bool, default=True)
    parser.add_argument("--export_version", type=int, default=1)

    args = parser.parse_args()
    print('[input params] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()
    topic = args.topic
    train_data = args.train_data
    train_data_sep = args.train_data_sep
    params_file = args.params_file
    use_model = args.model
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim
    monitor = args.monitor
    learning_rate = args.learning_rate
    epochs = args.epochs
    dnn_dropout = args.dnn_dropout
    dnn_use_bn = args.dnn_use_bn
    export_version = args.export_version

    data = pd.read_csv(train_data, sep=train_data_sep)

    params_json = read_json(file_path=params_file)
    target = params_json['label']
    sparse_features = params_json["sparse_features"]
    dense_features = params_json["dense_features"]

    data[sparse_features] = data[sparse_features].fillna('null')
    data[dense_features] = data[dense_features].fillna(0)

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field, and record dense feature field name
    fixlen_feature_columns = \
        [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=embedding_dim) for i, feat in enumerate(sparse_features)] \
        + [DenseFeat(feat, 1,) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.1, random_state=2022)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model
    print("[use_model] {} ".format(use_model))
    if use_model == "DeepFM":
        model = DeepFM(linear_feature_columns=linear_feature_columns,
                       dnn_feature_columns=dnn_feature_columns,
                       # fm_group=sparse_features,
                       task='binary',
                       dnn_dropout=dnn_dropout,
                       dnn_use_bn=dnn_use_bn)
    elif use_model == "DCNMix":
        model = DCNMix(linear_feature_columns=linear_feature_columns,
                       dnn_feature_columns=dnn_feature_columns,
                       task='binary',
                       dnn_dropout=dnn_dropout,
                       dnn_use_bn=dnn_use_bn)
    elif use_model == "DeepFEFM":
        model = DeepFEFM(linear_feature_columns=linear_feature_columns,
                         dnn_feature_columns=dnn_feature_columns,
                         task='binary',
                         dnn_dropout=dnn_dropout,
                         dnn_use_bn=dnn_use_bn)
    elif use_model == "DIFM":
        model = DIFM(linear_feature_columns=linear_feature_columns,
                     dnn_feature_columns=dnn_feature_columns,
                     task='binary',
                     dnn_dropout=dnn_dropout,
                     dnn_use_bn=dnn_use_bn)
    else:
        model = xDeepFM(linear_feature_columns=linear_feature_columns,
                        dnn_feature_columns=dnn_feature_columns,
                        task='binary',
                        dnn_dropout=dnn_dropout,
                        dnn_use_bn=dnn_use_bn)

    # 5.Train
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    filepath = "./checkpoint/{}-{}-weights-best.hdf5".format(topic, use_model)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath, monitor=monitor, verbose=1, save_best_only=True, mode='max')

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=3, min_delta=0.0001, verbose=1)

    lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=1e-8,
                                                        patience=3, verbose=1)

    history = model.fit(x=train_model_input, y=train[target].values,
                        callbacks=[checkpoint],
                        batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.1)

    # 6.Predict, use best model.
    model.load_weights(filepath=filepath)
    pred_ans = model.predict(x=test_model_input, batch_size=batch_size)
    print("\n[BEST] ===============================================================")
    print("[test] LogLoss: {} ".format(round(log_loss(test[target].values, pred_ans), 4)))
    print("[test] Accuracy: {} ".format(round(accuracy_score(test[target].values, pred_ans >= 0.5), 4)))
    print("[test] AUC: {} ".format(round(roc_auc_score(test[target].values, pred_ans), 4)))
    print("[test] classification_report: \n{} ".format(classification_report(test[target].values, pred_ans >= 0.5, digits=4)))

    # 7. export train model
    model.summary()
    tf.compat.v1.saved_model.save(
        obj=model,
        export_dir='./export_model/{}-{}/{}'.format(topic, use_model, export_version),
        signatures=None)
