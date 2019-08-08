import GlobalParameter
import jieba
import pandas as pd


# 去掉回车换行
def delete_r_n(line):
    return line.replace("\r", "").replace("\n", "").strip()


# 读取停用词
def get_stop_words(stop_words_dir):
    stop_words = []

    with open(stop_words_dir, "r", encoding=GlobalParameter.encoding) as f_reader:
        for line in f_reader:
            line = delete_r_n(line)
            stop_words.append(line)

    stop_words = set(stop_words)
    return stop_words


# jieba精确分词
def jieba_cut(content, stop_words):
    word_list = []

    if content != "" and content is not None:
        seg_list = jieba.cut(content)
        for word in seg_list:
            if word not in stop_words:
                word_list.append(word)

    return word_list


# jieba搜索引擎分词
def jieba_cut_for_search(content, stop_words):
    word_list = []
    if content != "" and content is not None:
        seg_list = jieba.cut_for_search(content)
        for word in seg_list:
            if word not in stop_words:
                word_list.append(word)

    return word_list


# 清除不存在词汇表中的词语
def clear_word_from_vocab(word_list, vocab):
    new_word_list = []

    for word in word_list:
        if word in vocab:
            new_word_list.append(word)

    return new_word_list


# 文本预处理
def preprocessing_text_pd(text_dir, after_process_text_dir, stop_words_dir):
    stop_words = get_stop_words(stop_words_dir)
    sentences = []
    df = pd.read_csv(text_dir)

    for index, row in df.iterrows():
        # print(index)
        title = delete_r_n(row['title'])
        word_list = jieba_cut(title, stop_words)
        df.loc[index, 'title'] = " ".join(word_list)
        sentences.append(word_list)
    df.to_csv(after_process_text_dir, encoding=GlobalParameter.encoding, index=False)

    return sentences


# 文本预处理第二种方式
def preprocessing_text(text_dir, after_process_text_dir, stop_words_dir):
    # count = 0
    stop_words = get_stop_words(stop_words_dir)
    sentences = []
    f_writer = open(after_process_text_dir, "w", encoding=GlobalParameter.encoding)

    with open(text_dir, "r", encoding=GlobalParameter.encoding) as f_reader:
        for line in f_reader:
            line_list = line.split(",")
            if len(line_list) == 2:
                line_list[1] = delete_r_n(line_list[1])
                word_list = jieba_cut(line_list[1], stop_words)
                sentences.append(word_list)
                f_writer.write(line_list[0] + "," + " ".join(word_list) + "\n")
                f_writer.flush()
                # count = count + 1
                # print(count)
            else:
                print(line)

    f_writer.close()

    return sentences


if __name__ == "__main__":  # 如果只是运行本脚本
    stop_words = get_stop_words(GlobalParameter.stop_word_dir)
    # content = "我就读于北京航空航天大学，现是一名大三学生。"
    # word_list = jieba_cut(content, stop_words)
    # print(word_list)
    # word_list = jieba_cut_for_search(content, stop_words)
    # print(word_list)
    sentences = preprocessing_text(GlobalParameter.test_set_dir, GlobalParameter.test_after_process_text_dir, GlobalParameter.stop_word_dir)
    print(sentences[:10])


