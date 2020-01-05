import codecs
import json
import pandas as pd
import numpy as np
import os

from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from beard.similarity.pairs import *
from beard.similarity import JaccardSimilarity
from beard.similarity import PairTransformer
from beard.similarity import StringDistance
from beard.similarity import EstimatorTransformer
from beard.similarity import ElementMultiplication
from beard.utils import FuncTransformer
from beard.utils import Shaper
import pickle
import time

positive_sample = None
negative_sample = None
pub_dict = None
if os.path.exists("data/train2w/pos.csv"):
    with codecs.open("data/train/train_pub.json", "r", "utf-8") as f:
        pub_dict = json.load(f)
    positive_sample = pd.read_csv("data/train2w/pos.csv", encoding='utf-8')
    negative_sample = pd.read_csv("data/train2w/neg.csv", encoding='utf-8')
else:
    with codecs.open("data/train/train_pub.json", "r", "utf-8") as f:
        pub_dict = json.load(f)
    ad_pair = pd.read_csv("data/pair_data1.csv", encoding="utf-8")
    ad_pair = ad_pair.sample(frac=1).reset_index(drop=True)
    positive_sample = ad_pair[ad_pair['2'] == 1][:40000]
    negative_sample = ad_pair[ad_pair['2'] == 0][:40000]
    positive_sample.to_csv("data/train2w/pos.csv", index=False, encoding="utf-8")
    negative_sample.to_csv("data/train2w/neg.csv", index=False, encoding="utf-8")

sample_all = positive_sample.append(negative_sample)
sample_all = sample_all.values


def _build_distance_estimator(X, y, Xt, yt, verbose=0):
    """Build a vector reprensation of a pair of signatures."""
    transformer = FeatureUnion([
        ("author_name", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("full_name", FuncTransformer(func=get_authors)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="word",
                                           ngram_range=(1, 2),
                                           dtype=np.float32,
                                           decode_error="replace")),
            ]))),
            ("combiner", ConcatVec())
        ])),
        ("author_name_jad", Pipeline([
            ("pairs", FuncTransformer(func=get_authors)),
            ("combiner", MyJaccardSimilarity())
        ])),
        ("affiliation_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("affiliation", FuncTransformer(func=get_author_affiliations)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="word",
                                           dtype=np.float32,
                                           ngram_range=(1, 2),
                                           decode_error="replace")),
            ]))),
            ("combiner", ConcatVec())
        ])),
        ("affiliation_similarity_jad", Pipeline([
            ("pairs", FuncTransformer(func=get_author_affiliations)),
            ("combiner", MyJaccardSimilarity())
        ])),
        ("title_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("title", FuncTransformer(func=get_title)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="word",
                                           ngram_range=(1, 2),
                                           dtype=np.float32,
                                           decode_error="replace")),
            ]))),
            ("combiner", ConcatVec())
        ])),
        ("title_similarity_jad", Pipeline([
            ("pairs", FuncTransformer(func=get_title)),
            ("combiner", MyJaccardSimilarity())
        ])),
        ("venue_similarity_jad", Pipeline([
            ("pairs", FuncTransformer(func=get_venue)),
            ("combiner", MyJaccardSimilarity())
        ])),
        ("venue_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("venue", FuncTransformer(func=get_venue)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="word",
                                           # dtype=np.float32,
                                           ngram_range=(1, 2),
                                           decode_error="replace")),
            ]))),
            ("combiner", ConcatVec())
        ])),
        ("abstract_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("abstract", FuncTransformer(func=get_abstract)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="word",
                                           dtype=np.float32,
                                           ngram_range=(1, 2),
                                           decode_error="replace")),
            ]))),
            ("combiner", ConcatVec())
        ])),
        ("abstract_similarity_jad", Pipeline([
            ("pairs", FuncTransformer(func=get_abstract)),
            ("combiner", MyJaccardSimilarity())
        ])),
        ("keywords_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("keywords", FuncTransformer(func=get_keywords)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="word",
                                           # dtype=np.float32,
                                           ngram_range=(1, 2),
                                           decode_error="replace")),
            ]))),
            ("combiner", ConcatVec())
        ])),
        ("keywords_similarity_jad", Pipeline([
            ("pairs", FuncTransformer(func=get_keywords)),
            ("combiner", MyJaccardSimilarity())
        ])),
        ("author_org", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("keywords", FuncTransformer(func=get_author_org)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="word",
                                           # dtype=np.float32,
                                           ngram_range=(1, 2),
                                           decode_error="replace")),
            ]))),
            ("combiner", ConcatVec())
        ])),
        ("author_org_jad", Pipeline([
            ("pairs", FuncTransformer(func=get_author_org)),
            ("combiner", MyJaccardSimilarity())
        ])),
        ("year_diff", Pipeline([
            ("pairs", FuncTransformer(func=get_year, dtype=np.int)),
            ("combiner", AbsoluteDifference())
        ]))
    ])

    # clf = GradientBoostingClassifier(n_estimators=80,
    #                                  max_depth=10,
    #                                  # max_features=6,
    #                                  learning_rate=0.129,
    #                                  verbose=verbose)
    # clf = XGBClassifier(
    #     # learning_rate=0.1,
    #     n_estimators=1000,
    #     # max_depth=5,
    #     min_child_weight=1,
    #     gamma=0,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     objective='binary:logistic',
    #     nthread=4,
    #     scale_pos_weight=1,
    #     seed=27)
    # parameters = {'learning_rate': [0.01, 0.02], 'max_depth': [5]}
    # estimator = Pipeline([("transformer", transformer),
    #                       ("clf", clf)]).fit(X, y)
    # clf = GridSearchCV(estimator, param_grid=parameters, scoring='roc_auc')

    # 网格调参
    # parameters = {
    #     'clf__max_depth': [5, 10, 15, 20, 25],
    #     'clf__learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    #     'clf__n_estimators': [500, 1000, 2000, 3000, 5000]
    #     # 'min_child_weight': [0, 2, 5, 10, 20],
    #     # 'max_delta_step': [0, 0.2, 0.6, 1, 2],
    #     # 'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
    #     # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    #     # 'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
    #     # 'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
    #     # 'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
    #
    # }

    clf = XGBClassifier(max_depth=10,
                        learning_rate=0.01,
                        n_estimators=2000,
                        silent=True,
                        objective='binary:logistic',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=1440,
                        missing=None,
                        n_jobs=4)

    estimator = Pipeline([("transformer", transformer),
                          ("clf", clf)]).fit(X, y)

    # gsearch = GridSearchCV(estimator, param_grid=parameters, scoring='accuracy', cv=3)
    # gsearch.fit(X, y)
    #
    # print("Best score: %0.3f" % gsearch.best_score_)
    # print("Best parameters set:")
    # best_parameters = gsearch.best_estimator_.get_params()
    # for param_name in sorted(parameters.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))

    y_pred = estimator.predict(Xt)
    print("\tPrecision: %1.3f" % precision_score(yt, y_pred))
    print("\tRecall: %1.3f" % recall_score(yt, y_pred))
    print("\tF1: %1.3f\n" % f1_score(yt, y_pred))

    return estimator


def learn_model(pub_dict, sample=True, verbose=0):
    """Learn the distance model for pairs of signatures.
    """
    input_dataset = sample_all if sample else ad_pair
    train, test = train_test_split(input_dataset, train_size=0.7)
    X, y = train[:, :2], train[:, 2].astype(int)
    Xt, yt = test[:, :2], test[:, 2].astype(int)

    for i in range(len(X)):
        X[i][0] = pub_dict[X[i][0]]
        X[i][1] = pub_dict[X[i][1]]
    for i in range(len(Xt)):
        Xt[i][0] = pub_dict[Xt[i][0]]
        Xt[i][1] = pub_dict[Xt[i][1]]
    del pub_dict, input_dataset, train, test
    # Learn a distance estimator on paired signatures
    distance_estimator = _build_distance_estimator(
        X, y, Xt, yt, verbose=verbose)

    pickle.dump(distance_estimator,
                open("distance_model", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)


time_start = time.time()
learn_model(pub_dict, sample=True, verbose=0)
time_elapsed = time.time() - time_start
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
