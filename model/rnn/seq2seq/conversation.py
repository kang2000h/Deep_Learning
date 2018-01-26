import tensorflow as tf

import sys

import BotEngine
import GrammarGenerator

def train_be():
    BotEngine.train()

def train_gg():
    GrammarGenerator.train()

def test_be():
    with tf.Session() as sess:
        with tf.variable_scope("be_model"):
            model = BotEngine.create_model(sess, True)
        while True:
            try :
                query = input("나 : ")
            except EOFError as eofe:
                print("\n이만 테스트를 마칩니다.")
                break
            answer = BotEngine.decode(sess, model, query)
            result = []
            for word in answer:
                if '/' in word:
                    result.append(word[:word.index('/')])
                #else :
                #    result.append(word)
            answer = " ".join(result)
            print("Com : ", answer)


def test_gg():
    with tf.Session() as sess:
        with tf.variable_scope("gg_model"):
            model = GrammarGenerator.create_model(sess, True)
        while True:
            query = input()
            answer = GrammarGenerator.decode(sess, model, query, True)

            result = []
            for word in answer:
                result.append(word[:word.index('/')])
            answer = " ".join(result)
            print("Com : ", answer)

def conversation():
    with tf.Session() as sess:
        with tf.variable_scope("be_model") as scope:
            model_be = BotEngine.create_model(sess, True)
            print("be 모델 생성 완료")
        with tf.variable_scope("gg_model"):
            scope.reuse_variables()
            model_gg = GrammarGenerator.create_model(sess, True)
            print("gg 모델 생성 완료, 이어서 대화 시작")
        while True:
            try:
                query = input()
                print("나 : ", query)
            except EOFError as eofe:
                print("이만 테스트를 마칩니다.")
                break
            f_answer = BotEngine.decode(sess, model_be, query)
            print("be 모델 판단")
            i_answer = GrammarGenerator.decode(sess, model_gg, f_answer, False)
            print("gg 모델 판단")
            result = []
            for word in i_answer:
                if '/' in word:
                    result.append(word[:word.index('/')])
            answer = " ".join(result)
            print("Com : ", answer)


def main():
    optionLen = len(sys.argv)
    if optionLen <= 1 or optionLen > 2 :
        print("Usage : python %prog %option")
        print("-h 혹은 --help를 추가하여 사용법을 확인하세요.")
        exit()

    if sys.argv[1] == '-h' or sys.argv[1] =='--help':
        print("-tb or ", "--train_be", " : BE 훈련")
        print("-tg or ", "--train_gg", " : GG 훈련")
        print("-ttbe or ", "--test_be", " : BE 테스트")
        print("-ttge or ", "--test_gg", " : GG 테스트")
        print("-cv or ", "--conversation", " : 대화 테스트")

    elif sys.argv[1] == '-tb' or sys.argv[1] =='--train_be':
        print("[System] BotEngine 훈련 시작")
        train_be()
    elif sys.argv[1] == '-tg' or sys.argv[1] == '--train_gg':
        print("[System] GrammarGenerator 훈련 시작")
        train_gg()
    elif sys.argv[1] == '-ttbe' or sys.argv[1] == '--test_be':
        print("[System] BotEngine 테스트 시작")
        test_be()
    elif sys.argv[1] == '-ttge' or sys.argv[1] == '--test_gg':
        print("[System] GrammarGenerator 테스트 시작")
        test_gg()
    elif sys.argv[1] == '-cv' or sys.argv[1] == '--conversation':
        conversation()
    else :
        print("-h 혹은 --help를 추가하여 사용법을 확인하세요.")

    '''
    options, args = parser.parse_args()
    if options.train_be:
        
    '''

if __name__ == "__main__":
    main()