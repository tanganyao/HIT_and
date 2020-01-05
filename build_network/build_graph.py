import argparse
import os
import re
import pandas as pd
from pyspark.sql import Row, SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *
import gc

home_dir = os.path.expanduser('.')
save_dir = os.path.join(home_dir)


def build_graph(args):
    """
    构建作者图谱
    """
    spark = SparkSession \
        .builder \
        .appName("pubmed_oa_spark1") \
        .getOrCreate()
    affiliation = spark.read.parquet(args.affiliation)
    author = spark.read.parquet(args.author)

    if affiliation and author:
        author = author.groupBy("first_name", 'last_name', 'pmid', 'pmc').\
            agg(concat_ws(";", collect_set(col('affiliation_id'))).alias("aff_list"))
        author_fullname = author.withColumn("author_name", concat_ws(" ", col('last_name'), col('first_name')))
        groupid = author_fullname.groupBy("pmid").agg(concat_ws(";", collect_list(col('author_name'))).
                                                      alias("co_author"))
        co_authors = author_fullname.alias('a').join(groupid.alias('b'), col('a.pmid') == col('b.pmid'), how="left").\
            select('a.*', 'b.co_author')

        del author_fullname, author
        gc.collect()

        concat_pmid_co_author = co_authors.withColumn("pmid_co_author", concat_ws("@", col('pmid'), col('co_author')))
        author_agg_all = concat_pmid_co_author.groupBy("author_name").\
            agg(concat_ws("#", collect_set(col('pmid_co_author'))).alias("pmid_cu_list"))

        del concat_pmid_co_author
        gc.collect()

        def getpmid_agg(x):
            # pm_author = x.pmid_cu_list
            ca_result = []
            pm_result = []
            # pmid_cu_list = pm_author.split("#")
            pmid_cu_list = x.split("#")
            for pcu in pmid_cu_list:
                if len(pcu.split("@")) < 2:
                    continue
                au_list = pcu.split("@")[1].split(";")
                pm_id = pcu.split("@")[0]
                if not len(ca_result):
                    ca_result.append(au_list)
                    pm_result.append([pm_id])
                else:
                    flag = False
                    for i in range(len(ca_result)):
                        if len(list(set(au_list).intersection(set(ca_result[i])))) > 2:
                            pm_result[i].extend([pm_id])
                            ca_result[i].extend(list(set(au_list).difference(set(ca_result[i]))))
                            flag = True
                            break
                    if not flag:
                        ca_result.append(au_list)
                        pm_result.append([pm_id])
            result = ''
            for i in range(len(pm_result)):
                result += "#".join(pm_result[i])
                if i < len(pm_result)-1:
                    result += ";"
            del ca_result, pm_result, pmid_cu_list
            gc.collect()
            # return Row(author_name=x.author_name, pmid_all=result, pmid_cu_list=x.pmid_cu_list)
            return result

        # author_agg_rdd = author_agg_all.rdd
        # author_all_pmid_rdd = author_agg_rdd.\
        #     map(lambda x: getpmid_agg(x))
        # author_all_pmid = author_all_pmid_rdd.toDF()
        udf_get_pmid = udf(getpmid_agg, StringType())
        author_all_pmid = author_agg_all.withColumn('pmid_all', udf_get_pmid(col('pmid_cu_list')))
        author_all_pmid = author_all_pmid.drop("pmid_cu_list")
        author_pmids = author_all_pmid.withColumn("pmid_each", explode(split(col('pmid_all'), ";")))\
            .withColumn('flag', col('pmid_each'))\
            .withColumn("pmid", explode(split(col('pmid_each'), "#")))
        columns_to_drop = ['pmid_all', 'pmid_each']
        author_pmids = author_pmids.drop(*columns_to_drop)

        del author_agg_all, author_all_pmid
        gc.collect()

        def rename_cols(rename_df):
            for column in rename_df.columns:
                new_column = column+"1"
                rename_df = rename_df.withColumnRenamed(column, new_column)
            return rename_df
        co_authors = rename_cols(co_authors)

        author_pmid_join = author_pmids.alias('a').join(co_authors.alias('b'), col('a.pmid') == col('b.pmid1'), "left")\
            .select('a.*', 'b.aff_list1', 'b.co_author1')

        # 获取机构信息
        author_pmid_aff = author_pmid_join.withColumn("aff", explode(split(col("aff_list1"), ";")))
        del author_pmid_join
        gc.collect()
        author_pmid_aff = author_pmid_aff.drop("aff_list1")
        author_pmid_aff = author_pmid_aff.alias("a").\
            join(affiliation.alias("b"), (col("a.aff") == col("b.affiliation_id")) & (col('a.pmid') == col('b.pmid')),
                 "left").select("a.*", "b.affiliation")
        author_result_dump = author_pmid_aff.groupBy(col('author_name'), col('flag')).\
            agg(countDistinct("pmid").alias("publication_num"),
                concat_ws(";", collect_set(col('pmid'))).alias('pmids'),
                concat_ws(";", collect_set(col('affiliation'))).alias('aff_names'),
                concat_ws(";", collect_set(col('co_author1'))).alias('coauthors_')
                )

        del co_authors, author_pmids, author_pmid_aff
        gc.collect()

        def delete_author_name(coauthors, author_name):
            coauthors = list(set(coauthors.split(";")))
            coauthors.remove(author_name)
            return ";".join(coauthors)

        udf_dan = udf(delete_author_name, StringType())
        author_result_dump = author_result_dump.withColumn("co_authors", udf_dan(col('coauthors_'), col('author_name'))).\
            select("author_name", 'publication_num', 'pmids', 'aff_names', 'co_authors')
        author_result_dump.write.parquet(os.path.join(save_dir, 'pubmed_graph_1.parquet'),
                                    mode='overwrite')
        print("keke")
    else:
        print("输入有误，请检查")


conf = SparkConf().setAppName('pubmed_build_graph') \
    .setMaster('local[20]') \
    .set('spark.executor.memory', '8g') \
    .set('spark.driver.memory', '16g') \
    .set('spark.local.dir', 'E:/tmp') \
    .set('spark.driver.maxResultSize', '0') \
    .set("spark.sql.shuffle.partitions", '2000')


if __name__ == '__main__':
    sc = SparkContext(conf=conf)
    # input 3 file: 1）全文。2）作者信息。3）作者机构
    parser = argparse.ArgumentParser()
    # parser.add_argument("--items", required=True, type=str)
    parser.add_argument("--author", required=True, type=str)
    parser.add_argument("--affiliation", required=True, type=str)
    args = parser.parse_args()
    build_graph(args)
    sc.stop()