# -*- coding: utf-8 -*-
import gzip
import nltk
from os import listdir, makedirs
from os.path import join, exists
import re
from bs4 import BeautifulSoup
import subprocess
from config import gov2_collection_path, gov2_output_path, gov2_docno_list

# This script is based on:
# https://gist.github.com/dervn/859717/15b69ef75a04489f3a517b3d4f70c7e97b39d2ec

useful_docno = set()
if gov2_docno_list != "":
    with open(gov2_docno_list, 'r') as f:
        for line in f:
            docno = line.strip()
            useful_docno.add(docno)


def filter_tags(htmlstr):
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  # Script
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)  # style
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


def run(command, get_ouput=False):
    try:
        if get_ouput:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            output, err = process.communicate()
            return output
        else:
            subprocess.call(command)
    except subprocess.CalledProcessError as e:
        print(e)


class TrecReader:
    """
    Read trec web files and give document raw text
    """

    def __init__(self, file_path):
        self.f = gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')

    def __iter__(self):
        return self

    def __next__(self):
        """
        :return: the next document
        """
        line = self.f.readline()  # <DOC>
        if not line:
            raise StopIteration()

        line = self.f.readline()  # <DOCNO>
        docno = line.strip().split('>')[1].split('<')[0]

        while True:
            line = self.f.readline().strip()
            if line == "</DOCHDR>":
                break
        lines = []
        while True:
            line = self.f.readline().strip()
            if line == "</DOC>":
                break
            lines.append(line)
        html_text = ' '.join(lines)
        return docno, html_text


def text_clean(text):
    ltoken = nltk.word_tokenize(text)
    for i in range(len(ltoken)):
        token = ltoken[i]
        if not token.isalnum():
            continue
        ltoken[i] = token.lower()
    res = ' '.join(ltoken)
    res = ' '.join(res.split())
    return res


def main():
    main_path = gov2_collection_path
    folders = sorted([folder for folder in listdir(main_path) if folder.startswith('GX')])
    files = []

    for folder in folders:
        for f in sorted(listdir(join(main_path, folder))):
            files.append(join(main_path, folder, f))

    for file_path in files:

        output_path = join(gov2_output_path, file_path.split("/")[-2])
        if not exists(output_path):
            makedirs(output_path)
        fout_path = join(output_path, file_path.split("/")[-1].split(".gz")[0] + ".txt")
        with open(fout_path, 'w') as fout:
            trec_reader = TrecReader(file_path)

            for docno, html_text in trec_reader:
                # if useful_docno set is provided, we only consider those docnos.
                if len(useful_docno) != 0 and docno not in useful_docno:
                    continue
                if not html_text:
                    print("html_text empty: {}".format(docno))
                else:
                    soup = BeautifulSoup(html_text, 'html5lib')
                    text = soup.get_text()
                    text = text.replace('\n', ' ').replace('\t', ' ').strip()

                    if not text:
                        print("use filter_tags for {}".format(docno))
                        text = filter_tags(html_text)
                        text = text.replace('\n', ' ').replace('\t', ' ').strip()
                    text = re.sub("\r", "", text)
                    text = re.sub("\s+", " ", text)
                    fout.write(docno + '\t' + text + '\n')


if __name__ == '__main__':
    main()
