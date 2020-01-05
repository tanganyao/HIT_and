import codecs
import json
import os


def concat_res(path="data/res"):
    paths = os.listdir(path)
    res = dict()
    for p in paths:
        with codecs.open(path+"/"+p, "r", "utf-8") as f:
            tmp = json.load(f)
            res = dict(res, **tmp)
            print(len(res.keys()))
    with codecs.open("concat_result.json", "w", "utf-8") as f:
        f.write(json.dumps(res))


if __name__ == "__main__":
    concat_res("data/res")