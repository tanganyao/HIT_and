import codecs
import pandas as pd
import numpy as np
import json
import random
import pickle

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
from scipy import sparse


def disambiguate_by_cluster():
    with codecs.open("data/sna_data/sna_valid_pub.json", "r", "utf-8") as f:
        pub_valid_dict = json.load(f)

    with codecs.open("data/sna_data/sna_valid_author_raw.json", "r", "utf-8") as f:
        author_raw = json.load(f)

    clusters = {}
    n = 0
    for author in author_raw:
        papers = author_raw[author]
        print(author + str(n))
        n += 1
        n_samples = len(papers)
        if not n_samples:
            clusters[author] = []
            continue
        paper_dict = {}
        author_name = []
        affiliation = []
        title = []
        abstract = []
        venue = []
        keywords = []
        years = []
        coauther_orgs = []
        author_venue = []
        org_title = []
        org_venue = []
        title_abstract_keywords = []
        author_keywords = []
        author_title = []
        author_org_title = []
        for paper in papers:
            p = pub_valid_dict[paper]
            author_name.append(get_authors(p))
            affiliation.append(get_author_affiliations(p))
            title.append(get_title(p))
            abstract.append(get_abstract(p))
            venue.append(get_venue(p))
            keywords.append(get_keywords(p))
            years.append(get_year(p))
            coauther_orgs.append(get_author_org(p))
            author_venue.append(get_author_venue(p))
            org_title.append(get_org_title(p))
            org_venue.append(get_org_venue(p))
            title_abstract_keywords.append(get_title_abstract_keywords(p))
            author_keywords.append(get_author_keywords(p))
            author_title.append(get_author_title(p))
            author_org_title.append(get_author_org_title(p))
        tfidf1 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(coauther_orgs)
        tfidf2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
            fit_transform(coauther_orgs)
        tfidf3 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(author_venue)
        tfidf4 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
            fit_transform(author_venue)
        tfidf5 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(org_title)
        tfidf6 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
            fit_transform(org_title)
        tfidf7 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(org_venue)
        tfidf8 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
            fit_transform(org_venue)
        tfidf9 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(title_abstract_keywords)
        tfidf10 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
            fit_transform(title_abstract_keywords)
        tfidf18 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(author_keywords)
        tfidf19 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
            fit_transform(author_keywords)
        tfidf20 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(author_title)
        tfidf21 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
            fit_transform(author_title)
        tfidf22 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(author_org_title)
        tfidf23 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
            fit_transform(author_org_title)
        tfidf11 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(author_name)
        #         tfidf18 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
        #             fit_transform(author_name)
        tfidf12 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(affiliation)
        #         tfidf19 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
        #             fit_transform(affiliation)
        tfidf13 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(title)
        #         tfidf20 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
        #             fit_transform(title)
        #         tfidf14 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(abstract)
        tfidf15 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(venue)
        #         tfidf21 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
        #             fit_transform(venue)
        tfidf16 = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(keywords)
        #         tfidf22 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), dtype=np.float32, decode_error="replace"). \
        #             fit_transform(keywords)
        tfidf17 = np.array([[int(i) / 2019] for i in years])
        #         tfidf = sparse.hstack((tfidf1, tfidf2, tfidf3, tfidf4, tfidf5, tfidf6, tfidf7,
        #                                tfidf8, tfidf9, tfidf10, tfidf11, tfidf12, tfidf13,
        #                                tfidf15, tfidf16, tfidf17, tfidf18, tfidf19, tfidf20,
        #                                tfidf21, tfidf22))
        tfidf = sparse.hstack((tfidf1, tfidf2, tfidf3, tfidf4, tfidf5, tfidf6, tfidf7,
                               tfidf8, tfidf10, tfidf9, tfidf11, tfidf12, tfidf13,
                               tfidf15, tfidf16, tfidf17, tfidf18, tfidf19,
                               tfidf20, tfidf21, tfidf23))
        # sim_mertric = pairwise_distances(tfidf, metric='cosine')

        clf = DBSCAN(eps=0.5, metric='cosine', min_samples=2)
        s = clf.fit_predict(tfidf)
        print("--------------------")
        print(s)
        # 每个样本所属的簇
        for label, paper in zip(clf.labels_, papers):
            if label not in paper_dict:
                paper_dict[label] = [paper]
            else:
                paper_dict[label].append(paper)
        clusters[author] = list(paper_dict.values())
    # return clusters
    with codecs.open("cluster_output.json", "w", "utf-8") as wf:
        wf.write(json.dumps(clusters))


if __name__ == "__main__":
    disambiguate_by_cluster()
