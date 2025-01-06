import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import BertTokenizer, BertModel
from paddlenlp.embeddings import TokenEmbedding
from paddlenlp.data import JiebaTokenizer
import numpy as np


def filter_by_text2vec(dataset, threshold=0.1):
    tokenizer = BertTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
    model = BertModel.from_pretrained('shibing624/text2vec-base-chinese')

    def element2vec(item):
        s = str(item["conversations"])
        with paddle.no_grad():
            inputs = tokenizer(s, return_tensors='pd')
            inputs["input_ids"]=inputs["input_ids"][:,:512]
            inputs["token_type_ids"]=inputs["token_type_ids"][:,:512]
            outputs = model(**inputs)
        vec = outputs[1]
        return vec

    vec_list = []
    for i in range(len(dataset)-1, -1, -1):
        new_vec = element2vec(dataset[i])
        for old_vec in vec_list:
            dis = 1-F.cosine_similarity(new_vec, old_vec).numpy()[0]
            print(dis)
            if dis < threshold:
                dataset.pop(i)
                break
        else:
            print(f"len of vec list {len(vec_list)}")
                
    return dataset


def filter_by_ngram(dataset, threshold=0.1):
    token_embedding = TokenEmbedding(embedding_name='w2v.baidu_encyclopedia.target.word-ngram.1-2.dim300')
    tokenizer = JiebaTokenizer(vocab=token_embedding.vocab)

    def element2vec(item):
        s = str(item["conversations"])
        words = tokenizer.cut(s)
        vec = token_embedding.search(words).mean(axis=0)
        vec = paddle.to_tensor(vec).reshape([1,-1])
        return vec

    vec_list = []
    for i in range(len(dataset)-1, -1, -1):
        new_vec = element2vec(dataset[i])
        for old_vec in vec_list:
            dis = 1-F.cosine_similarity(new_vec, old_vec).numpy()[0]
            print(dis)
            if dis < threshold:
                dataset.pop(i)
                break
        else:
            print(f"len of vec list {len(vec_list)}")
                
    return dataset
