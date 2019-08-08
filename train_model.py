import GlobalParameter
import utils
from gensim.models import word2vec


# 训练word2vec
def train(sentences, model_out_put_path):
    print("开始训练")
    # min_count为需要计算词向量的最小词频，可以用于减小总词汇量
    model = word2vec.Word2Vec(sentences=sentences, size=GlobalParameter.train_size, window=GlobalParameter.train_window, min_count=20)
    model.save(model_out_put_path)
    print("训练完成")


if __name__ == "__main__":
    # 训练
    #stop_words = utils.get_stop_words(GlobalParameter.stop_word_dir)
    #sentences = utils.preprocessing_text(GlobalParameter.train_set_dir, GlobalParameter.train_after_process_text_dir, GlobalParameter.stop_word_dir)
    #train(sentences, GlobalParameter.model_output_path)
    # 查看训练结果
    model = word2vec.Word2Vec.load(GlobalParameter.model_output_path)
    vocab = list(model.wv.vocab.keys())
    for e in model.most_similar(positive=['地震'], topn=10):
        print(e[0], e[1])
    print(len(vocab))




