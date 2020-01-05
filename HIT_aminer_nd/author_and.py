import pickle
import codecs
import json
import numpy as np
# These imports are used during unpickling.
from utils import *

'''
--- measure distance between the same name authors, and finally cluster the same signatures belong to the same real
--- world author.
--- estimator classifier was train from file distance.py

@author tay 
'''


def author_and():

    distance_estimator = pickle.load(open("distance_model", "rb"))

    with codecs.open("data/sna_data/sna_valid_pub.json", "r", "utf-8") as f:
        pub_valid_dict = json.load(f)

    with codecs.open("data/sna_data/sna_valid_author_raw.json", "r", "utf-8") as f:
        author_raw = json.load(f)

    Xt = np.empty((1, 2), dtype=np.object)
    name2cluster = {}
    count = 0
    for author, pub_list in author_raw.items():
        if not len(pub_list):
            name2cluster[author] = []
            continue
        name2cluster[author] = [[pub_list[0]]]
        for i, x in enumerate(pub_list[1:]):
            break_tag = 0
            for j, y in enumerate(name2cluster[author]):
                for k, z in enumerate(y):
                    Xt[0, 0], Xt[0, 1] = pub_valid_dict[x], pub_valid_dict[z]
                    predict = distance_estimator.predict_proba(Xt)[0][0]
                    if predict > 0.5:
                        name2cluster[author][j].append(x)
                        break_tag = 1
                        break
                if break_tag:
                    break
            if not break_tag:
                name2cluster[author].append([x])
        count += 1
        print(count)
    with codecs.open("cluster_output.json", "w", "utf-8") as wf:
        wf.write(json.dumps(name2cluster))


if __name__ == "__main__":
    author_and()


