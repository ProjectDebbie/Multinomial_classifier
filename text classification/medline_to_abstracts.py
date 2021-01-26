#!/usr/bin/env python
# coding: utf-8


import re
import os


#define folders and prefix
refs = '...' # put the directory with the abtracts in MEDLINE format
folder = '...' # put the directory of the folder where you want to store the abstracts
folder_2 = '...' # put the directory of the folder where you want to store the final abstracts
prefix = '...' # choose your prefix here


# break medline file into segments

def convert_medline(refs):
    with open (refs) as data_file:
        block = ""
        found = False
        x = 0
        for line in data_file:
            if found:
                block += line
                if line.startswith("SO"):
                    x = x + 1
                    found = False
                    with open(folder + prefix + str(x) + ".txt", "w") as file:
                        file.write(block)
                        block = ""
            else:
                if line.startswith("PMID"):
                    found = True
                    block += line


# extract PMID, title and abstract

def clean_to_title_abstract(folder):
    abstracts = [os.path.join(folder, f) for f in os.listdir(folder)]
    for abstract in abstracts:
        with open(abstract) as f:
            content = f.read()
        try:
            text0 = re.search(r'PMID-(.*?)OWN -', content, re.DOTALL).group(1)
            file_name = str(folder_2 + text0[1:-1] + '.txt')
            date = re.search(r'DP  -(.*?)TI  -', content, re.DOTALL).group(1)
            text1 = re.search(r'TI  -(.*?).  -', content, re.DOTALL).group(1)
            with open(file_name, "a") as file:
                file.write(date)
                file.write(text0)
                file.write(text1)
            try:
                text2 = re.search(r'AB  -(.*?)CI  -', content, re.DOTALL).group(1)
                with open(file_name, "a") as file:
                    file.write(text2)
            except AttributeError:
                print(text0[:-1] + ': no abtract text on first try')
                
                try:
                    text3 = re.search(r'AB  -(.*?)FAU -', content, re.DOTALL).group(1)
                    with open(file_name, "a") as file:
                        file.write(text3)
                except AttributeError:
                    print(text0[:-1] + ': no abstract at all')
                    os.remove(file_name)
                    continue
        except:
            print('no PMID')
            continue


convert_medline(refs)
clean_to_title_abstract(folder)


# clean spaces, remove some punctatuation signs

def simplify_text_abstracts(input):
    abstracts = [os.path.join(input, f) for f in os.listdir(input)]
    for abstract in abstracts:
        L = []
        with open(abstract) as f:
            content = f.readlines()
            for line in content:
                line= line.replace("-", " ")
                line= line.replace("(", " ")
                line= line.replace(")", " ")
                line = line.strip()
                L.append(line)
                with open(abstract, 'w') as f:
                    for item in L:
                        f.write("%s\n" % item)

simplify_text_abstracts(folder_2)                       

