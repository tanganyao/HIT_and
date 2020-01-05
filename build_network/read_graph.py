import argparse
import os
import re
import pandas as pd
from pyspark.sql import Row, SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *

home_dir = os.path.expanduser('.')
graph_dir = os.path.join(home_dir)


def read_graph(args):
    """
        读取作者图谱
    """
    spark = SparkSession \
        .builder \
        .appName("read_graph") \
        .getOrCreate()
    author_graph = spark.read.parquet(args.graph)
    pd.DataFrame(author_graph.collect()).to_csv("pubmeda_b.csv", index=False)
    print("test")


conf = SparkConf().setAppName('read_graph') \
    .setMaster('local[10]') \
    .set('executor.memory', '8g') \
    .set('driver.memory', '2g') \
    .set('spark.driver.maxResultSize', '0')


if __name__ == '__main__':
    sc = SparkContext(conf=conf)
    # input 3 file: 1）作者图
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, type=str)
    args = parser.parse_args()
    read_graph(args)
    sc.stop()