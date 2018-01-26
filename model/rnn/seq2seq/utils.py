from konlpy.tag import Twitter

"""BotEngine에서 사용하기 위한 각종 서브 유틸들을 모았다."""

def query_disintegrator(query):
    tagger = Twitter()
    disintegrated_sentence = ['/'.join(tags) for tags in tagger.pos(query, norm=True, stem=True)]
    result = []
    for token in disintegrated_sentence:
        if 'Eomi' not in token and 'Josa' not in token and 'Number' not in token and 'KoreanParticle' not in token and 'Punctuation' not in token:
            result.append(token)
    return result

def query_preserver(query):
    tagger = Twitter()
    preserved_sentence = ['/'.join(tags) for tags in tagger.pos(query)]
    result = []
    for token in preserved_sentence:
        if 'Number' not in token and 'Punctuation' not in token:
            result.append(token)
    return result


def idx2token(idx, reverse_vocab):  # 이제 반대로 인덱스를 받으면 값을 주는 메소드다. (아마 generative하게 인덱스를 추론하고 가능성 높은 녀석을 디코딩하는 작업에 필요하겠지)
    return reverse_vocab[str(idx)]


def idx2sent(indices, reverse_vocab):  # 이제 숫자의 배열을 받으면 그녀석에 대한 워드 값들로 디코딩하고 " "(공백)으로 나눠서 붙일 것.
    return " ".join([idx2token(idx, reverse_vocab) for idx in indices])


def token2idx(token, vocab):
    return vocab[token]

# 그러나 sents 크기는 1일 것이다.
def sent2idx(sent, vocab):
    result = []
    for word in sent:
        try:
            result.append(vocab[word])
        except KeyError as ke:
            result.append(vocab["<UNK>"])
    return result


def tokenize(doc):
    '''
    need to import konlp as a dependency
    spliting a string into taged morpheme

    Args:
      sent: A string. A sentence.

    Returns: A list of words.    
    '''
    from konlpy.tag import Twitter
    tagger = Twitter()
    return ['/'.join(t) for t in tagger.pos(doc, norm=True, stem=True)]

