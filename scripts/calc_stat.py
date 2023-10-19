import sys
from datasets import load_dataset


# calcuate the average length of question and answer in the dataset
filepath = sys.argv[1]

# load the dataset
dataset = load_dataset('json', data_files=filepath, split='train')

# get the length of each question and answer
title_length = []
question_length = []
answer_length = []
for item in dataset:
    title_length.append(len(item['title'].strip().split(' ')))
    question_length.append(len(item['question'].strip().split(' ')))
    answer_length.append(len(item['answer'].strip().split(' ')))


# calculate the average length
title_avg = sum(title_length) / len(title_length)
question_avg = sum(question_length) / len(question_length)
answer_avg = sum(answer_length) / len(answer_length)

# print the result
print('title_avg: ', title_avg)
print('question_avg: ', question_avg)
print('answer_avg: ', answer_avg)

# visualize the distribution of length
# save the figure to file
import matplotlib.pyplot as plt
# save the figure to file
plt.hist(title_length, bins=100)
plt.savefig('title_length.png')
plt.hist(question_length, bins=100)
plt.savefig('question_length.png')
plt.hist(answer_length, bins=100)
plt.savefig('answer_length.png')


plt.clf()
# draw them in one figure and add legend
# plt.hist(title_length, bins=100, alpha=0.5, label='title')
# i want the style of the figure to be the same as the one in the paper
# set style to be paper style
# plt.style.use('seaborn-paper')
# adjust the size of the figure to fit two column width such as ACL series
# plt.figure(figsize=(6, 3))
# 
# adjust font size to fit ACL series tempalte
# plt.rcParams.update({'font.size': 24})

# max_len = 1000
# plt.hist([x for x in question_length if x <= max_len], bins=100, alpha=0.5, label='Q', color='orange')
# plt.hist([x for x in answer_length if x <= max_len], bins=100, alpha=0.5, label='A', color='blue')
# # plt.xlim(0, 1500)
# plt.xlabel('Length')
# plt.ylabel('Count')
# plt.legend(loc='upper right')
# plt.savefig('length.png')


# plot histogram of length of question and answer using seaborn
import seaborn as sns
import matplotlib.ticker as ticker


sns.set()
sns.set_style('whitegrid')
sns.set_palette('Set2')
sns.set_context('paper')
sns.set(font_scale=1)
plt.figure(figsize=(6, 3))
max_len = 1000
sns.histplot([x for x in question_length if x <= max_len], bins=100, alpha=0.5, label='Q', color='orange')
sns.histplot([x for x in answer_length if x <= max_len], bins=100, alpha=0.5, label='A', color='blue')
plt.xlabel('Length')
plt.ylabel('Count')
plt.legend(loc='upper right')
plt.gca().xaxis.set_major_formatter(ticker.EngFormatter())
plt.gca().yaxis.set_major_formatter(ticker.EngFormatter())
plt.tight_layout()
plt.savefig('length.png')
