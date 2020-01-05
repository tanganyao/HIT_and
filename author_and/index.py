# coding:utf-8

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import codecs
import os
import json
import math
from util import get_affiliation

app = Flask(__name__)
# machine = ''
# context = ''


@app.route('/')
def index():
    # 获得paper前十返回
    with codecs.open("paper_sum_result.json", "r", "utf-8") as f1:
        author2jdi = json.load(f1)
        author2paper = dict()
        for author, infor in author2jdi.items():
            jdi_list = infor[1:]
            # print(jdi_list)
            sorted_jdi = sorted(jdi_list, key=lambda x: x['num'], reverse=True)[:3]
            if author not in author2paper:
                affiliation = infor[0][1]
                aff = get_affiliation(affiliation)
                author2paper[author] = [aff]
                author2paper[author].append(infor[0][-2])
                author2paper[author].extend(sorted_jdi)
    with codecs.open("jdi_sum_result.json", "r", "utf-8") as f2:
        author2jdi = json.load(f2)
        author2jdi_sum = dict()
        for author, infor in author2jdi.items():
            if author not in author2jdi_sum:
                affiliation = infor[0][1]
                aff = get_affiliation(affiliation)
                author2jdi_sum[author] = [aff]
                author2jdi_sum[author].append(infor[0][-2])
                author2jdi_sum[author].append(infor[0][-1])
                author2jdi_sum[author].append(infor[-1])
    return render_template('index.html', author2paper=author2paper, author2jdi_sum=author2jdi_sum)


@app.route('/doc')
def doc():
    return render_template('docs.html')


# @app.route('/authors/<keywords>/<int:p>')
@app.route('/authors', methods=['GET', 'POST'])
def authors(keywords=None, p=1):
    show_shouye_status = 0

    if request.method == 'POST':
        keywords = request.values.get("keyword")
        p = int(request.values.get("p"))
    elif request.method == 'GET':
        # r = request
        keywords = request.args.get("keywords")
        p = request.args.get("p")      # 页数
    # return keywords, p
    # page = request.args.get('page', 1, type=int)
    # pagination = Post.query.paginate(page, per_page=1, error_out=False)
    # posts = pagination.items
    author_list = []
    with codecs.open("author_info_output.json", "r", "utf-8") as f:
        author_domin_infor = json.loads(f.read())
        # keywords = keywords.strip()
        if keywords and keywords in author_domin_infor:
            author_list = author_domin_infor[keywords]
    # p = request.values.get("p")

    if p == '':
        p = 1
    else:
        p = int(p)
        if p > 1:
            show_shouye_status = 1
    number_per_page = 6
    total = int(math.ceil(len(author_list)/number_per_page))
    if number_per_page * p <= len(author_list):
        author_list_per_page = author_list[number_per_page * (p-1):number_per_page * p]
    else:
        author_list_per_page = author_list[number_per_page * (p - 1):len(author_list)]

    # print(author_list_per_page)
    # 只取第一个
    for i, author in enumerate(author_list_per_page):
        if ";" in author[1][1]:
            author_list_per_page[i][1][1] = author[1][1].split(";")[0]
        elif '.' in author[1][1]:
            author_list_per_page[i][1][1] = author[1][1].split(".")[0]
        elif '\n' in author[1][1]:
            author_list_per_page[i][1][1] = author[1][1].split('\n')[0]
        # break

    dic = get_page(total, p)
    datas = {
        'info_list': author_list_per_page,
        'p': int(p),
        'total': total,
        'show_shouye_status': show_shouye_status,
        'dic_list': dic
    }

    return render_template('authors.html', datas=datas, keywords=keywords)


def get_page(total, p):
    show_page = 5   # 显示的页码数
    pageoffset = 2  # 偏移量
    start = 1    # 分页条开始
    end = total  # 分页条结束

    if total > show_page:
        if p > pageoffset:
            start = p - pageoffset
            if total > p + pageoffset:
                end = p + pageoffset
            else:
                end = total
        else:
            start = 1
            if total > show_page:
                end = show_page
            else:
                end = total
        if p + pageoffset > total:
            start = start - (p + pageoffset - end)
    # 用于模版中循环
    dic = range(start, end + 1)
    return dic


@app.route('/coauthor')
def get_coauthor_kg():
    print("aaa")


@app.route('/author_detail', methods=['GET', 'POST'])
def get_author_detail():
    if request.method == 'POST':
        author_detail = request.values.get("author_detail")
    return render_template('author_detail.html', author_detail=author_detail)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run()
    # test()
