import copy
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk.corpus import stopwords, wordnet

# nltk.download() # 下载语料库
my_text = "Hello Mr. Adam, how are you? I hope everythin is going well. Today is a good day,see you later."

# 分词
print(word_tokenize(my_text))
# 分句
print(sent_tokenize(my_text))
# 词频统计
tokens = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please', 'maybe', 'not', 'take', 'him', 'to', 'dog', 'park',
          'stupid', 'my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him']
freq = nltk.FreqDist(tokens)
for k, v in freq.items():  # 输出每个词出现的次数
    print(f"{str(k)}:{str(v)}")
common = freq.most_common(3)  # 统计最常出现的若干个词
print(common)
freq.plot(20)  # 画图显示频率最高的若干个词(以词作为x轴次数做y轴，画折线图)

# 去停用词
clean_tokens = copy.deepcopy(tokens)
stwords = stopwords.words("english")  # 英文的停用词表
for token in tokens:
    if token in stwords:
        clean_tokens.remove(token)
print(clean_tokens)

# 词干提取
porter = PorterStemmer()
print(porter.stem("working"))
for token in clean_tokens:
    print(porter.stem(token))  # 提取词干

# 词性标注
text_1 = word_tokenize(my_text)
print(nltk.pos_tag(text_1))

# 使用wordnet获取单词的词集
syn = wordnet.synsets("car")  # 获得 car 这个单词的词集
print(syn[0].definition())  # 获得单词的释义
print(syn[0].examples())  # 例句
# 获取近义词
synonyms = list()  # 近义词列表
for syn in wordnet.synsets('pc'):
    for w in syn.lemmas():
        synonyms.append(w.name())
print(synonyms)

# 获取反义词
antonyms = list()  # 翻译成列表
for syn in wordnet.synsets('large'):
    for w in syn.lemmas():
        if w.antonyms():
            antonyms.append(w.antonyms()[0].name())
print(antonyms)
