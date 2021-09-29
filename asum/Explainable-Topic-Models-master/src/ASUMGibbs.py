import time
import numpy as np
import scipy.special
import re
import codecs as cs
import utils


class ASUMGibbs:
    """
    Aspect and Sentiment Unification Model for Online Review Analysis
    Yohan Jo, Alice Oh, 2011
    """
    def __init__(self, num_topics, senti_words_path, doc_file_path, alpha=0.1, beta=0.01, gamma=0.1, high_beta=0.7):
        """
        Constructor method
        Constructor method
        :param num_topics: the number of topics
        :param doc_file_path: BOW document file path
        :param vocas: vocabulary list
        :param alpha: alpha value in ASUM
        :param beta: beta value in ASUM
        :param gamma: gamma value in ASUM
        :param high_beta: beta value in ASUM for sentiment words
        :return: void
        :return: void
        """
        #self.docs = self.read_bow(doc_file_path)  # s
        #self.words = vocas

        self.docs,self.words_to_idx,self.ratings = self.read_corpus(doc_file_path)

        self.senti_words = self.read_model_prior(senti_words_path)

        self.K = num_topics                             #opic数量
        self.D = len(self.docs)                         #文档数量
        self.W = len(self.words_to_idx)   #word num            单词数量
        self.S = len(self.senti_words)  #sentiment num  几种情感

        # Hyper-parameters
        self.alpha = alpha

        #self.senti_words = [[pos_word],[neg_word]]

        self.beta = np.zeros((self.S, self.W)) + beta
        for senti_idx, one_senti_words in enumerate(self.senti_words):
            for one_senti_word in one_senti_words:
                if one_senti_word in self.words_to_idx.keys():
                    if senti_idx == 0: #pos的
                        self.beta[0, self.words_to_idx[one_senti_word]] = 1
                        self.beta[1, self.words_to_idx[one_senti_word]] = 0
                    else:   #neg
                        self.beta[1, self.words_to_idx[one_senti_word]] = 1
                        self.beta[0, self.words_to_idx[one_senti_word]] = 0
        self.gamma = gamma

        self.DST = np.zeros((self.D, self.S, self.K), dtype=np.int64)
        self.STW = np.zeros((self.S, self.K, self.W), dtype=np.int64)

        # Random initialization of topics
        self.doc_topics = list()

        for di in range(self.D):
            doc = self.docs[di]

            topics = np.random.randint(self.S * self.K, size=len(doc))  #每个句子都有一个主题和情感
            self.doc_topics.append(topics)

            #句子的主题和情感
            for senti_topic, sentence in zip(topics, doc):
                senti = senti_topic // self.K
                topic = senti_topic % self.K

                self.DST[di, senti, topic] += 1

                target_mat = self.STW[senti, topic, :]      #取出senti 和 topic关联的那一行单词
                for word_idx, word_cnt in sentence:
                    target_mat[word_idx] += word_cnt



    def read_corpus(self, corpus_path):
        """Joint Sentiment-Topic Model using collapsed Gibbs sampling.

        In the corpus, every document takes one line, the first word of a document
        is 'di',where i is the index of the document,following are the words in the
        document sperated by ' '.


        ASUM读取数据集，这部分改写成自己的
        one_doc.append(zip(word_ids, word_counts))
        docs.append(one_doc)
        """


        """
        #读取所有的csv文件名
        csv_files = os.listdir('./data')
        #读取所有csv文件
        for file in csv_files:    
        """

        #每次只运行一个类型的
        raw_reviews, self.docs_ratings = utils.read_csv_data(corpus_path)

        cleaned_docs = utils.preprocessing(raw_reviews)
        #对文本进行预处理，得到处理后的文件
        print("document size : ",len(cleaned_docs))
        #处理后的文件 每个doc有
        #
        self.vocab = set()

        #添加字典
        for doc in cleaned_docs: #每个文档
            for st in doc:       #每个句子
                for word in st:
                    self.vocab.add(word) #将单词加入到字典中


        # word2id
        word_idx = 0
        self.words_to_idx = dict()
        self.idx_to_words = dict()
        for word in self.vocab:
            self.words_to_idx[word] = word_idx
            self.idx_to_words[word_idx] = word
            word_idx += 1

        #做one_doc.append(zip(word_ids, word_counts))
        #docs.append(one_doc)
        self.docs = []
        for doc in cleaned_docs:
            one_doc = []
            for st in doc:
                word_ids = []
                word_counts = []
                for word in st:
                    word_id = self.words_to_idx[word]
                    if word_id in word_ids:
                        index = word_ids.index(word_id)
                        word_counts[index]+=1
                    else:
                        word_ids.append(word_id)
                        word_counts.append(1)
                one_doc.append(zip(word_ids, word_counts))
            self.docs.append(one_doc)
        return self.docs,self.words_to_idx,self.docs_ratings



    def read_model_prior(self, prior_path):
        """Joint Sentiment-Topic Model using collapsed Gibbs sampling.

        Sentiment lexicon or other methods.

        The format of the prior imformation are as follow :
        [word]  [neu prior prob.] [pos prior prob.] [neg prior prob.]
        ...

        读取情感字典
        使用的是MPQA
        The format of the prior imformation are as follow :
        [word]  [neu prior prob.] [pos prior prob.] [neg prior prob.]

        """
        self.senti_words = []
        pos_words = []
        neg_words = []
        model_prior = cs.open(prior_path, 'r')
        for word_prior in model_prior.readlines():
            word_prior = word_prior.strip().split()
            index = 1
            maxm = -1.0
            for i in range(2,len(word_prior)): #第3个位置开始，因为第一个是word，第二个是中性概率
                word_prior[i] = float(word_prior[i])
                if word_prior[i] > maxm:       #找到三种情感中最大的
                    maxm = word_prior[i]
                    index = i
                if index-2 == 0:
                    pos_words.append(word_prior[0])
                else:
                    neg_words.append(word_prior[0])

        self.senti_words.append(pos_words)
        self.senti_words.append(neg_words)
        return self.senti_words   #self.prior形式  ： {word: [0.05,0.9, 1]}  0.05是pos概率 0.9是neg概率， 1是说明这个单词是neg的,
                            #ASUM 所以根据情感字典 pos=0,neg =1




    def run(self, max_iter=2000, do_optimize=False, do_print_log=False):
        """
        Run Collapsed Gibbs sampling for ASUM
        :param max_iter: Maximum number of gibbs sampling iteration
        :param do_optimize: Do run optimize hyper-parameters
        :param do_print_log: Do print loglikelihood and run time
        :return: void
        """
        if do_optimize and do_print_log:
            prev = time.perf_counter()
            for iteration in range(max_iter):
                print(iteration, time.clock() - prev, self.loglikelihood())
                prev = time.perf_counter()
                self._gibbs_sampling()
                if 99 == iteration % 100:
                    self._optimize()
        elif do_optimize and not do_print_log:
            for iteration in range(max_iter):
                self._gibbs_sampling()
                if 99 == iteration % 100:
                    self._optimize()
        elif not do_optimize and do_print_log:
            prev = time.perf_counter()
            for iteration in range(max_iter):
                print(iteration, time.clock() - prev, self.loglikelihood())
                prev = time.perf_counter()
                self._gibbs_sampling()
        else:
            prev = time.perf_counter()
            for iteration in range(max_iter):
                print(iteration, time.perf_counter() - prev)
                prev = time.perf_counter()
                self._gibbs_sampling()


    def _gibbs_sampling(self):
        """
        Run Gibbs Sampling
        :return: void
        """
        for di in range(self.D):
            doc = self.docs[di]
            cur_doc_senti_topics = self.doc_topics[di]

            for sentence_idx, sentence in enumerate(doc):
                # Old one
                old_senti_topic = cur_doc_senti_topics[sentence_idx]
                senti = old_senti_topic // self.K
                topic = old_senti_topic % self.K

                self.DST[di, senti, topic] -= 1
                target_mat = self.STW[senti, topic, :]
                for word_idx, word_cnt in sentence:
                    target_mat[word_idx] -= word_cnt

                first_term = np.sum(self.DST[di, :, :], axis=1, keepdims=True) + self.gamma
                second_term_part = self.DST[di, :, :] + self.alpha
                second_term = second_term_part / (np.sum(second_term_part, axis=1, keepdims=True))
                third_term_part = np.sum(self.STW, axis=2) + np.sum(self.beta, axis=1, keepdims=True)
                forth_term = 1
                words_in_doc = 0
                for word_idx, word_cnt in sentence:
                    words_in_doc += word_cnt
                    forth_term_part = self.STW[:, :, word_idx] + self.beta[:, word_idx]
                    temp_prod = 1
                    for cnt_idx in range(word_cnt):
                        temp_prod *= (forth_term_part + cnt_idx)
                    forth_term *= temp_prod
                third_term = scipy.special.gamma(third_term_part) / (scipy.special.gamma(third_term_part + words_in_doc))

                # Sampling
                prob = first_term * second_term * third_term * forth_term
                prob = prob.flatten()

                # New one
                new_senti_topic = self._sampling_from_dist(prob)
                senti = new_senti_topic // self.K
                topic = new_senti_topic % self.K

                cur_doc_senti_topics[sentence_idx] = new_senti_topic

                self.DST[di, senti, topic] += 1
                target_mat = self.STW[senti, topic, :]
                for word_idx, word_cnt in sentence:
                    target_mat[word_idx] += word_cnt

    @staticmethod
    def _sampling_from_dist(prob):
        """
        Multinomial sampling with probability vector
        :param prob: probability vector
        :return: a new sample (In this class, it is new topic index)
        """
        thr = prob.sum() * np.random.rand()
        new_topic = 0
        tmp = prob[new_topic]
        while tmp < thr:
            new_topic += 1
            tmp += prob[new_topic]
        return new_topic

    def loglikelihood(self):
        """
        Compute log likelihood function
        :return: log likelihood function
        """
        return self._topic_loglikelihood() + self._document_loglikelihood()

    def _topic_loglikelihood(self):
        """
        Compute log likelihood by topics
        :return: log likelihood by topics
        """
        raise NotImplementedError

    def _document_loglikelihood(self):
        """
        Compute log likelihood by documents
        :return: log likelihood by documents
        """
        raise NotImplementedError

    def _optimize(self):
        """
        Optimize hyperparameters
        :return: void
        """
        self._alphaoptimize()
        self._betaoptimize()
        self._gammaoptimize()

    def _alphaoptimize(self, conv_threshold=0.001):
        """
        Optimize alpha vector
        :return: void
        """
        raise NotImplementedError

    def _betaoptimize(self, conv_threshold=0.001):
        """
        Optimize beta value
        :return: void
        """
        raise NotImplementedError

    def _gammaoptimize(self, conv_threshold=0.001):
        """
        Optimize gamma value
        :return: void
        """
        raise NotImplementedError

    def export_result(self, output_file_name, rank_idx=100):
        """
        Export Algorithm Result to File
        :param output_file_name: output file name
        :param rank_idx:
        :return: the number of printed words in a topic in output file
        """
        # Raw data
        np.save("%s_DST.npy" % output_file_name, self.DST)
        np.save("%s_STW.npy" % output_file_name, self.STW)

        # Ranked words in topics
        with open("%s_Topic_Ranked.csv" % output_file_name, "w") as ranked_topic_word_file:
            for senti_idx in range(self.S):
                for topic_idx in range(self.K):
                    topic_vec = self.STW[senti_idx, topic_idx, :]
                    sorted_words = sorted(enumerate(topic_vec), key=lambda x: x[1], reverse=True)
                    print('senti/topic {}/{},{}'.format(senti_idx, topic_idx,
                                                        ",".join([self.idx_to_words[x[0]] for x in sorted_words[:rank_idx]])),
                          file=ranked_topic_word_file)


    def doc_sentiment_acc(self):

        #self.DST[di, senti, topic]
        #计算每个文档的Sentiment的概率
        sentiments = []
        for di in range(self.D):
            if np.sum(self.DST[di,0,:]) > np.sum(self.DST[di,1,:]):
                sentiments.append(0)
            else:
                sentiments.append(1)

        pos_cnt = 0
        neg_cnt = 0
        true_pos = 0
        true_neg  = 0
        for i in range(self.D):
            r = self.docs_ratings[i][0]
            if r > 3:
                pos_cnt+=1
            if r < 3:
                neg_cnt +=1
            if r > 3 and sentiments[i]==0:
                true_pos += 1
            if r< 3 and sentiments[i]==1:
                true_neg += 1

        print("pos acc:{}".format(true_pos/pos_cnt))
        print("neg acc:{}".format(true_neg / neg_cnt))
        print("acc:{}".format((true_pos+true_neg)/(pos_cnt+neg_cnt)))