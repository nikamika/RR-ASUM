""" JST RR Model using collapsed Gibbs sampling. """

import codecs as cs
import random
import os
import sys
import numpy as np
import numbers
import heapq

import pyximport
import utils
pyximport.install()
import _JST_RR

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


class JST_RR(object):

    def __init__(self, topics=1, sentilab=2, iteration=300,
                 sigma = 2, ratings=None, random_state=123456789,
                 refresh=50):

        #默认的rating等级
        if ratings is None:
            ratings = [1, 2, 3, 4, 5]

        self.topics = topics       #多少个topic
        self.sentilab = sentilab   #多少个sentiment,默认2个
        self.iter = iteration      #Gibbs采样的迭代次数
        self.alpha = 1.0 / (sentilab + .0) * (topics + .0)   #alpha = 3.0 /(S*K)
        self.beta = 0.001
        self.gamma = 1.0 / (sentilab + .0)                   #gamma = 3.0 / S
        self.sigma = sigma         #rating关联单词的权重

        self.ratings = ratings    #评分的等级 默认1,2,3,4,5
        self.rating_num = len(ratings)   #有几种rating


        self.random_state = random_state
        self.refresh = refresh

        #if self.alpha <= 0 or beta <= 0:
            #raise ValueError("alpha,beta and gamma must be greater than zero")

        rng = self.check_random_state(random_state)
        self._rands = rng.rand(1024**2 // 8)

    def check_random_state(self, seed):
        if seed is None:
            # i.e., use existing RandomState
            return np.random.mtrand._rand
        if isinstance(seed, (numbers.Integral, np.integer)):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError("{} cannot be used as a random seed.".format(seed))

    def read_corpus(self, corpus_path):
        """Joint Sentiment-Topic Model using collapsed Gibbs sampling.

        In the corpus, every document takes one line, the first word of a document
        is 'di',where i is the index of the document,following are the words in the
        document sperated by ' '.


        读取数据集，这部分改写成自己的
        """


        self.vocab = set()
        self.docs_ratings = []
        self.docs = []
        self.doc_num = 0
        self.doc_size = 0



        """
        #读取所有的csv文件名
        csv_files = os.listdir('./data')
        #读取所有csv文件
        for file in csv_files:    
        """

        #每次只运行一个类型的
        raw_reviews, ratings = utils.read_csv_data(corpus_path)
        self.docs.extend(raw_reviews)
        self.docs_ratings.extend(ratings)

        #对文本进行预处理，得到处理后的文件
        print("document size : ",len(self.docs))
        self.docs = utils.preprocessing(self.docs)


        for doc in self.docs:
            self.doc_num += 1  #记录总的doc数量
            for word in doc:
                self.vocab.add(word)   #将单词加入到字典中
                self.doc_size += 1     #记录总的单词数量

        return self.docs,self.docs_ratings

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
        self.prior = {}
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
            self.prior[word_prior[0]] = word_prior[2:] #先记录三种概率
            self.prior[word_prior[0]].append(index-2)  #最后一位是记录最大概率的index

        return self.prior   #self.prior形式  ： {word: [0.05,0.9, 1]}  0.05是pos概率 0.9是neg概率， 1是说明这个单词是neg的,
                            #由于JST_RR只分pos和neg两类，所以根据情感字典 pos=1,neg =0

    def analyzecorpus(self):

        """
        记录word2id : word到id的映射关系
           id2word : id到word的映射关系
        :return:
        """
        #get dict {word:id ...} and {id:dict ...}

        self.vocabsize = len(self.vocab)
        if "d1950" in self.vocab:
            print ("you are wrong !!!")
        print ("vocabsize : ",self.vocabsize)
        print ("total doc words : ",self.doc_size)
        self.word2id = {}
        self.id2word = {}
        index = 0
        for item in self.vocab:
            self.word2id[item] = index
            self.id2word[index] = item
            index += 1

    def init_model_parameters(self):

        """
        初始化参数
        nd :  文档d中有多少单词
        ndl : 文档d中sentiment l有多少单词
        ndlz : 文档d中sentiment l关联的topic z有多少个单词
        nlzw : 单词w关联sentiment l和 topic z的次数
        nlz : 关联sentiment l和topic z的次数
        mdl : 文档d中sentiment l 的评分数量
        md  : 文档d中的评分数量
        mlr : sentiment l关联的rating=r的数量
        ml  : sentiment l关联的rating总数量

        最后估计的分布参数
        pi_dl
        theta_dlz
        phi_lzw
        mu_lr

        alpha_lz
        alphasum_l
        beta_lzw
        betasum_lz
        gamma_dl
        gammasum_d
        delta_lr
        :return:
        """
        #model counts
        self.nd = np.zeros((self.doc_num, ), dtype=np.int32)
        self.ndl = np.zeros((self.doc_num, self.sentilab), dtype=np.int32)
        self.ndlz = np.zeros((self.doc_num, self.sentilab, self.topics), dtype=np.int32)
        self.nlzw = np.zeros((self.sentilab, self.topics, self.vocabsize), dtype=np.int32)
        self.nlz = np.zeros((self.sentilab, self.topics), dtype=np.int32)
        #新增的对rating的 counts
        self.mdl = np.zeros((self.doc_num,self.sentilab),dtype=np.int32)
        self.md = np.zeros((self.doc_num, ), dtype=np.int32)
        self.mlr =  np.zeros((self.sentilab,self.rating_num),dtype=np.int32)
        self.ml = np.zeros((self.sentilab,),dtype=np.int32)

        #model parameters
        self.pi_dl = np.zeros((self.doc_num, self.sentilab), dtype=np.float)
        self.theta_dlz = np.zeros((self.doc_num, self.sentilab, self.topics), dtype=np.float)
        self.phi_lzw = np.zeros((self.sentilab, self.topics, self.vocabsize), dtype=np.float)
        self.mu_lr = np.zeros((self.sentilab,self.rating_num),dtype=np.float)

        #init hyperparameters with prior imformation
        self.alpha_lz = np.full((self.sentilab, self.topics), fill_value=self.alpha)
        self.alphasum_l = np.full((self.sentilab, ), fill_value=self.alpha*self.topics)


        if(self.beta <= 0):
            self.beta = 0.01
        # self.beta_lzw = np.full((self.sentilab, self.topics, self.vocabsize), fill_value=self.beta)
        # self.betasum_lz = np.full((self.sentilab, self.topics), fill_value=self.beta*self.vocabsize)

        # 初始化beta，情感l,topic z和单词w
        self.beta_lzw = np.full((self.sentilab, self.topics, self.vocabsize), fill_value=self.beta)
        # 情感l和topic z的总和
        self.betasum_lz = np.zeros((self.sentilab, self.topics), dtype=np.float)     #

        # #word prior 每个单词对于每个sentiment 都有一个先验概率，根据情感字典，添加先验概率
        # l=0是pos的词 l=1是neg
        self.add_lw = np.full((self.sentilab, self.vocabsize), fill_value=self.beta)

        #self.add_lw
        self.add_prior()
        for l in range(self.sentilab):
            for z in range(self.topics):
                for r in range(self.vocabsize):
                    #print(self.add_lw)
                    #print(self.betasum_lz)
                    self.beta_lzw[l][z][r] = self.add_lw[l][r]   #如果是在情感字典中的则直接赋值1或0，其余字默认为0.01
                    self.betasum_lz[l][z] += self.beta_lzw[l][z][r]
        print(self.betasum_lz)

        if self.gamma <= 0:
            self.gamma = 1.0
        # self.gamma_dl = np.full((self.doc_num, self.sentilab), fill_value=self.gamma)
        # self.gammasum_d = np.full((self.doc_num, ), fill_value=self.gamma*self.sentilab)

        self.gamma_dl = np.full((self.doc_num, self.sentilab), fill_value=0.0)
        self.gammasum_d = np.full((self.doc_num, ), fill_value=.0)
        for d in range(self.doc_num):
            # self.gamma_dl[d][1] = 1.8
            self.gamma_dl[d][0] = self.gamma      #gamma默认值值为3.0 / S
            self.gamma_dl[d][1] = self.gamma
        for d in range(self.doc_num):
            for l in range(self.sentilab):
                self.gammasum_d[d] += self.gamma_dl[d][l]


        self.delta_lr = np.zeros((self.sentilab, self.rating_num),dtype=np.float)
        self.deltasum_l = np.zeros((self.sentilab, ),dtype=np.float)
        for i in range(self.sentilab):
            for j in range(self.rating_num):
                if i==0: #pos的词
                    self.delta_lr[i][j] = self.ratings[j]*10.0
                    self.deltasum_l[i] += self.delta_lr[i][j]
                else: #i==1 neg
                    self.delta_lr[i][j] = 10.0*(6.0 - self.ratings[j])
                    self.deltasum_l[i] += self.delta_lr[i][j]



    def add_prior(self):

        """
        利用情感字典对单词的情感分布进行修改
        :return:
        """
        #beta add prior imformation
        for word in self.prior:
            if word in self.vocab:
                label = self.prior[word][-1] #获取到这个单词的情感最大的标签
                for l in range(self.sentilab):
                    #JST_RR和JST这一部分有一些许不一样
                    #self.add_lw[l][self.word2id[word]] *= self.prior[word][l]
                    if l == label:
                        self.add_lw[l][self.word2id[word]] = 1.0 #JST_RR中对再情感字典中的词，固定其情感
                    else:
                        self.add_lw[l][self.word2id[word]] = 0.0

    def init_estimate(self):
        print ("Estimate initializing ...")
        self.ZS = []   #记录topic顺序
        self.LS = []   #记录情感顺序
        self.WS = []   #记录单词顺序
        self.DS = []   #记录文档顺序
        self.IS = []   #记录是否在情感字典中
        self.DR = []    #记录每个文档中的rating  d * m种rating
        self.DRL = []  #记录文档中每个rating的情感分类    d * m

        cnt = 1
        prior_word_cnt = 0

        #遍历每个文档
        for m, doc in enumerate(self.docs):
            #遍历每个单词
            for t, word in enumerate(doc):
                cnt += 1    #记录单词数量+1

                if word in self.prior:    #如果单词是在情感字典中
                    senti = self.prior[word][-1]  #他的情感直接是明确的
                    self.IS.append(int(1))    #标记为是固定情感的
                    prior_word_cnt += 1     #记录情感单词数量+1
                else:
                    senti = random.choice(range(0,self.sentilab))
                    ##senti = (cnt) % self.sentilab  #如果不在情感字典中，则随机一个情感
                    self.IS.append(int(0))    #标记为不是固定情感的
                # topi = int(np.random.uniform(0,self.topics))
                topi = (cnt) % self.topics  #随机一个topic
                self.DS.append(int(m))      #记录这个单词的所属的doc id
                self.WS.append(int(self.word2id[word]))  #记录单词的id
                self.LS.append(int(senti))  #记录单词的sentiment
                self.ZS.append(int(topi))   #记录单词的topic

                self.nd[m] += 1             #记录文档m的单词数量+1
                self.ndl[m][senti] += 1     #记录文档m中sentiment l的数量+1
                self.ndlz[m][senti][topi] += 1   #记录文档m seitment l topic z的数量+1
                self.nlzw[senti][topi][self.word2id[word]] += 1  #记录word w关联sentiment l topic z的数量+1
                self.nlz[senti][topi] += 1    #记录sentiment l和topic 关联的数量

        r_cnt = 1
        for m, review_ratings in enumerate(self.docs_ratings):

            R = [] #记录一个文档中的rating  如果一个rating就只有一个 R = [1] ，如果多个则会 R=[1,2,3]
            RL = []  #记录每个rating的sentiment
            for rating in review_ratings:

                R.append(rating)
                r_cnt += 1
                senti = (r_cnt) % self.sentilab #给一个rating随机一个sentiment
                RL.append(senti)

                self.mdl[m][senti] += 1
                self.md[m] += 1
                self.mlr[senti][rating-1] += 1  #rating 1的位置在0
                self.ml[senti] += 1

            self.DR.append(R)
            self.DRL.append(RL)

        self.DS = np.array(self.DS, dtype = np.int32)
        self.WS = np.array(self.WS, dtype = np.int32)
        self.LS = np.array(self.LS, dtype = np.int32)
        self.ZS = np.array(self.ZS, dtype = np.int32)
        self.IS = np.array(self.IS, dtype = np.int32)
        self.DR = np.array(self.DR, dtype = np.int32)
        self.DRL = np.array(self.DRL, dtype = np.int32)

        print ("DS number and cnt are : ", len(self.DS),' ',cnt - 1)
        print ("affected words : ",prior_word_cnt)

        print ("total word # is : ", cnt - 1)

    def estimate(self):
        random_state = self.check_random_state(self.random_state)
        rands = self._rands.copy()
        # print rands[:100]
        self.init_estimate()
        print ("set topics : ",self.topics)
        print ("set gamma : ",self.gamma_dl[0][0], self.gamma_dl[0][1])
        print ("The {} iteration of sampling ...".format(self.iter))
        ll = ll_pre = 0.0

        for it in range(self.iter):
            """
            Recalculate the log likelihood every 50 times. If the likelihood is not improved, it will exit in advance
            """
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                print ("Iteration {} :".format(it))
                ll += self.loglikelihood()
                print ("<{}> log likelihood: {:.0f}".format(it, ll/(it/self.refresh + 1)))
                if ll/(it/self.refresh + 1) - 10 <= ll_pre and it > 0:
                    break
                ll_pre = ll/(it/self.refresh + 1)
            self._sampling(rands)

    def loglikelihood(self):

        """Calculate complete log likelihood, log p(w,z,l)
        Formula used is log p(w,z,l) = log p(w|z,l) + log p(z|l,d) + log p(l|d)
        """
        nd, ndl, ndlz, nlzw, nlz = self.nd, self.ndl, self.ndlz, self.nlzw, self.nlz
        return _JST_RR._loglikelihood(nd, ndl, ndlz, nlzw, nlz, self.alpha, self.beta, self.gamma)


    def _sampling(self, rands):
        _JST_RR._sample_topics(self.nd, self.ndl, self.ndlz, self.nlzw, self.nlz, self.mdl, self.md, self.mlr, self.ml, self.alpha_lz,
                               self.alphasum_l, self.beta_lzw, self.betasum_lz, self.gamma_dl, self.gammasum_d, self.sigma, self.delta_lr,
                               self.deltasum_l, self.DS, self.WS, self.LS, self.ZS, self.IS, self.DR, self.DRL, rands)

    def cal_pi_ld(self):
        for d in range(self.doc_num):
            for l in range(self.sentilab):
                self.pi_dl[d][l] = (self.ndl[d][l] + self.sigma*self.mdl[d][l] + self.gamma_dl[d][l]) / \
                                   (self.nd[d] + self.sigma*self.md[d] +self.gammasum_d[d])

    def cal_theta_dlz(self):
        for d in range(self.doc_num):
            for l in range(self.sentilab):
                for z in range(self.topics):
                    self.theta_dlz[d][l][z] = (self.ndlz[d][l][z] + self.alpha_lz[l][z]) \
                    / (self.ndl[d][l] + self.alphasum_l[l])

    def cal_phi_lzw(self):
        for l in range(self.sentilab):
            for z in range(self.topics):
                for w in range(self.vocabsize):
                    self.phi_lzw[l][z][w] = (self.nlzw[l][z][w] + self.beta_lzw[l][z][w]) \
                    / (self.nlz[l][z] + self.betasum_lz[l][z])

    def cal_mu_lr(self):
        for l in range(self.sentilab):
            for r in range(self.rating_num):
                self.mu_lr[l][r] = (self.mlr[l][r] + self.delta_lr[l][r]) / (self.ml[l] + self.deltasum_l[l])

def main():
    test = JST_RR(topics = 5, sentilab = 2, iteration = 300,
                    sigma = 2, ratings = [1,2,3,4,5], random_state = 123456789,
                    refresh = 50)

    #读取数据集
    clothes_csv_path = './data/Clothes.csv'
    sports_csv_path = './data/Sports.csv'
    Videogame_csv_path = './data/Videogame.csv'

    #以videogame为测试例子
    test.read_corpus(Videogame_csv_path)
    #读取情感字典
    test.read_model_prior('./preprocessing/mpqa.constraint')


    # print test.docs[1][:10]
    # print len(test.vocab),' ',test.prior['happi']
    # test.read_model_prior(r'.\constraint\filter_lexicon.constraint')
    test.analyzecorpus()
    test.init_model_parameters()
    test.estimate()

    test.cal_pi_ld()
    t1 = np.min(test.pi_dl[:,0])  #0是pos
    t2 = np.min(test.pi_dl[:,1])  #1是neg
    print ("PI_min is : ", min(t1, t2))
    cnt_pos = cnt_neg = 0
    doc_pos = 0
    doc_neg = 0

    #进行测试 这里将rating 小于3的作为neg的reviews,大于3的作为pos的reviews 等于3的先不做计算
    for d in range(test.doc_num):
        # if(d < 10):
        #     print test.pi_dl[d]
        """
        if test.pi_dl[d][0] > test.pi_dl[d][1]:
            doc_pos += 1
        if(test.pi_dl[d][0] <= test.pi_dl[d][1]):
            doc_neg += 1
        """
        if(test.docs_ratings[d][0] <= 3):
            doc_neg += 1
        if (test.docs_ratings[d][0] >= 4):
            doc_pos += 1

        if(test.docs_ratings[d][0] <= 3 and test.pi_dl[d][0] < test.pi_dl[d][1]):  #如果rating小于3并且 预测的情绪也是消极的
            cnt_neg += 1
        elif(test.docs_ratings[d][0] >= 4 and test.pi_dl[d][0] > test.pi_dl[d][1]):
            cnt_pos += 1

    print ("doc_neg : ",doc_neg,' ',"doc_pos : ",doc_pos)
    print("cnt_neg : ", cnt_neg, ' ', "cnt_pos : ", cnt_pos)
    # for i in xrange(test.doc_num):
    #     if(i < 1000 and test.ndl[i][1] < test.ndl[i][2] ):
    #         cnt_neg += 1
    #     elif(i >= 1000 and test.ndl[i][1] > test.ndl[i][2]):
    #         cnt_pos += 1

    print ("pos accurancy is : {:.2f}%".format((cnt_pos + .0) / doc_pos * 100))
    print ("neg accurancy is : {:.2f}%".format((cnt_neg + .0) / doc_neg * 100))
    print ("total accurancy is : {:.2f}%".format((cnt_pos + cnt_neg + .0) / test.doc_num * 100))

    # pp = (cnt_pos + .0) / doc_pos * 100
    # pn = (cnt_neg + .0) / doc_neg * 100
    # print "pos accurancy is : {:.2f}%".format(pp)
    # print "neg accurancy is : {:.2f}%".format(pn)
    # print "avrage accurancy is : {:.2f}%".format((pp + pn + .0) / 2)
    # print "total accurancy is : {:.2f}%".format((cnt_pos + cnt_neg + .0) / test.doc_num * 100)

    # print test.DS[:30]
    # print test.LS[:30]
    # print test.ZS[:30]
    test.cal_phi_lzw()


   # for l in range(test.sentilab):
   #     print ("sentiment ",l," :")
   #     for z in range(test.topics):
   #         res = heapq.nlargest(20, range(len(test.phi_lzw[l][z])), test.phi_lzw[l][z].take)
   #         print ("\ntopic ",z," : ")
   #         for item in res:
    #            print (test.id2word[item],' ')
  #      print("\n*************************************")


    print(test.pi_dl)

if __name__ == '__main__':
    main()
