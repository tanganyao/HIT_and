import mpi4py.MPI as MPI
import codecs
import json
import os


def generate_rank_directory(way="min", output_folder='data/rank'):
    # instance for invoking MPI relatedfunctions
    comm = MPI.COMM_WORLD
    # the node rank in the whole community
    comm_rank = comm.Get_rank()
    # the size of the whole community, i.e.,the total number of working nodes in the MPI cluster
    comm_size = comm.Get_size()
    with codecs.open("data/sna_data/sna_valid_author_raw.json", "r", "utf-8") as f:
        signatures = json.load(f)
        signatures_sorted = sorted(signatures.items(), key=lambda x: len(x[1]), reverse=True)
    rank_sum = [0 for _ in range(comm_size)]
    index_rank = []
    for s in signatures_sorted:
        min_index = rank_sum.index(min(rank_sum))
        publication_num = len(s[1])
        if way == "poly":
            y = publication_num * publication_num + 1
        else:
            y = publication_num
        rank_sum[min_index] += y
        index_rank.append(min_index)
    print(rank_sum)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    sub_dir = '%s/%d' % (output_folder, comm_rank)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    result = {}
    for index, name2sign in enumerate(signatures_sorted):
        if index_rank[index] == comm_rank:
            result[name2sign[0]] = name2sign[1]
    with codecs.open(sub_dir+"/sna_valid_author_raw.json", "w", "utf-8") as f:
        f.write(json.dumps(result))


if __name__ == "__main__":
    output_folder = "data/rank"
    generate_rank_directory(output_folder)