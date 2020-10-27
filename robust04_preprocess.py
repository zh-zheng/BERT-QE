import gzip
from bs4 import BeautifulSoup
import re
import os
import multiprocessing
from config import robust04_collection_path, robust04_output_path, robust04_docno_list

# This script is based on:
# https://gist.github.com/dervn/859717/15b69ef75a04489f3a517b3d4f70c7e97b39d2ec

useful_docno = set()
if robust04_output_path != "":
    with open(robust04_docno_list, 'r') as f:
        for line in f:
            docno = line.strip()
            useful_docno.add(docno)


def filter_tags(htmlstr):
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)
    re_br = re.compile('<br\s*?/?>')
    re_h = re.compile('</?\w+[^>]*>')
    re_comment = re.compile('<!--[^>]*-->')
    s = re_cdata.sub('', htmlstr)
    s = re_script.sub('', s)
    s = re_style.sub('', s)
    s = re_br.sub('\n', s)
    s = re_h.sub('', s)
    s = re_comment.sub('', s)
    blank_line = re.compile('\n+')
    s = blank_line.sub('\n', s)
    s = replaceCharEntity(s)
    return s


def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"', }

    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(htmlstr)
    while sz:
        key = sz.group('name')
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr


def preprocess(path):
    f_out = open(os.path.join(robust04_output_path, path.split("/")[-1].split(".gz")[0]), 'w')
    f = gzip.open(path, 'rt', encoding='utf8', errors='ignore')
    file = f.read()
    f.close()
    soup = BeautifulSoup(file, 'lxml')
    for doc in soup.find_all('doc'):
        docno = doc.find('docno').get_text().strip()
        if len(useful_docno) != 0 and docno not in useful_docno:
            continue
        title = ""
        try:
            title = doc.find('ti').get_text().strip()
        except AttributeError:
            try:
                title = doc.find('headline').get_text().strip()
            except AttributeError:
                pass
        title = re.sub("\n", " ", title)
        title = re.sub("\s+", " ", title)
        text = ""
        try:
            text = doc.find('text').get_text().strip()
            text = filter_tags(text)
            text = re.sub("\n", " ", text)
            text = re.sub("\s+", " ", text)
        except AttributeError:
            pass

        f_out.write(docno + "\t" + title.strip() + "\t" + text.strip() + "\n")
    f_out.close()


if __name__ == "__main__":
    main_path = robust04_collection_path
    folders = ['disk4/FR94', 'disk4/FT', 'disk5/FBIS', 'disk5/LATIMES']
    files = []
    for folder in folders:
        for f in os.listdir(os.path.join(main_path, folder)):
            if f.startswith("FT") or f.startswith("FR94") or f.startswith("FB") or f.startswith("LA"):
                files.append(os.path.join(main_path, folder, f))
    pool = multiprocessing.Pool(28)
    pool.map(preprocess, files)
    pool.close()
    pool.join()
