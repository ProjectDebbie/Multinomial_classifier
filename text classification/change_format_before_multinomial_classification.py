# master thesis project - Clarence Lépine (UPC Barcelona)
# code to change the format of the data before multinomial text classification

# import packages
import os
import pandas as pd
import glob


data = r'...'
# put the directory of your folder that contains abstracs from one category 

# to replace the date of the article by a label at the beginning of each text file
def add_label(data):
    abstracts = [os.path.join(data, f) for f in os.listdir(data)]
    for abstract in abstracts:
        with open(abstract) as f:
            lines = f.readlines()
            lines[0] = 'in vivo-in vitro\n'
            lines[1] = ';'
            with open(abstract, 'w') as f:
                f.writelines(lines)

# add_label(data)

input = r'...' # put your directory here

# text preprocessing
def preprocess(input):
   abstracts = [os.path.join(input, f) for f in os.listdir(input)]
   for abstract in abstracts:
       L = []
       with open(abstract) as f:
           content = f.readlines()
           for line in content:
               # line = line.replace(".", " ")
               line = line.replace(",", " ")
               line = line.replace(":", " ")
               line = line.lower()
               line = line.strip()
               L.append(line)
               with open(abstract, 'w') as f:
                   for item in L:
                       f.write("%s\n" % item)


# preprocess(input)


input = r'...' # put your directory here


# text preprocessing (put the text in one line after the label;)
def make_it_into_one_line(input):
    abstracts = [os.path.join(input, f) for f in os.listdir(input)]
    for abstract in abstracts:
        L = []
        with open(abstract) as f:
            content = f.readlines()
            for line in content:
                line = line.strip()
                line = line + ' '
                # line = line.replace(";", ",")
                # line = line.replace(", ", ",")
                line = line.replace("  ", " ")
                L.append(line)
                with open(abstract, 'w') as f:
                    for item in L:
                        f.write("%s" % item)

# make_it_into_one_line(input)


input = r'...' # put your directory here

# more text preprocessing (to add /n at the end of the file)
# abstracts = [os.path.join(input, f) for f in os.listdir(input)]
# for abstract in abstracts:
#     L = []
#     with open(abstract) as f:
#         content = f.readlines()
#         for line in content:
#             L.append(line)
#             with open(abstract, 'w') as f:
#                 for item in L:
#                     f.write("%s\n" % item)


# to convert all of the text files from one category into one

def make_one_file():
    read_files = glob.glob(r'...\*.txt') # put your directory here

    with open(r'... .txt', "wb") as outfile: # put your directory here
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())


# make_one_file()


input = r'...' # put your directory here
output = r'...' # put your final directory here

# to remove blank lines if needed -> lines with only one blank space here
def remove_blanklines():
    with open(input, "r") as f, open(output, "w") as outline:
        for i in f.readlines():
            if not i.strip():
                continue
            if i:
                outline.write(i)


# remove_blanklines()

# # to change the label of the files if needed
# content = []
# filename = r'...' # put your directory here
# with open(filename, 'r') as read_file:
#     content = read_file.readlines()
#
# with open(filename, 'w') as write_file:
#     for line in content:
#         write_file.write(line.replace('other ;', 'non biomaterials ;'))


# to create a dataframe for the multinomial text classification
def make_dataframe():
    with open(r'... .txt') as f: # put your directory here
        content = f.readlines()
        labels, texts = ([], [])
        for line in content:
            label, text = line.split(';', 1)
            labels.append(label)
            texts.append(text)

        df = pd.DataFrame()
        df['label'] = labels
        df['text'] = texts
        print(df)
        df.to_csv(r'... .csv', index=False, header=True, encoding='utf-8') # put your final directory here


# make_dataframe()

