#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec

from libc.stdlib cimport malloc, free

cdef extern from "gamma.c":
    cdef double jst_lgamma(double x) nogil


cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive")
    return jst_lgamma(x)



cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)
       C语言实现二分法
    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length - 1
    while imin <= imax:
        imid = imin + ((imax - imin) / 2)
        if value < arr[imid]:
            imax = imid - 1
        else:
            imin = imid + 1
    if imin <= length - 1 :
        return imin
    else:
        return length - 1

def _sample_topics(int[:] nd, int[:, :] ndl, int[:, :, :] ndlz, int[:, :, :] nlzw, 
                int[:, :] nlz,  int[:,:] mdl, int[:] md, int[:,:] mlr, int[:] ml, double[:, :] alpha_lz, double[:] alphasum_l,
                double[:, :, :] beta_lzw, double[:, :] betasum_lz, double[:, :] gamma_dl, 
                double[:] gammasum_d, double sigma, double[:,:] delta_lr, double[:] deltasum_l,
                int[:] DS, int[:] WS,  int[:] LS, int[:] ZS, int[:] IS, int[:,:] DR, int[:,:] DRL, int[:] SS,
                double[:] rands) :


    """
    nd : 每个文档中的数量
    ndl : 每个文档中每个sentiment的数量
    ndlz : 每个文档中sentiment l 和 topic z结合的数量
    nlzw : 每个单词w和sentiment l 和 topic z组合的数量
    nlz : 所有文档中 sentiment l 和 topic z组合的数量
    alpha_lz : 每个sentiment l 和 topic z组合对应的alpha
    alphasum_l :每个sentiment l对应的alpha总和
    beta_lzw : 每个sentiment l，topic z 和单词w组合对应的beta
    betasum_lz : 每个sentiment l,topic z组合的beta总和
    gamma_dl : 每个文档d中每个sentiment l的gamma
    gammasum_d : 每个文档中gamma总和

    为了rating分布新加入的变量：
    DR : 记录所有文档的rating  如 DR[d] = [1,5,3]，文档d的三个评分分别为1 5 3  一直不用更新

    mdl : 每个文档中sentiment l中有的 rating数量
    md : 每个文档d中的rating个数
    mlr : rating等于r时 sentiment l 的数量
    ml : sentiment l 的rating在整个数据集的数量
    sigma : 计算md和mdl的先验系数
    delta_lr : sentiment l 和rating r的联系数量的先验参数delta
    deltasum_l : sentiment l的 所有rating r的联系数量

    实现中把所有文档和单词的矩阵映射 展开为一维列表,加快算法的速度
    DS : 记录每个位置上对应的document id
    WS : 记录每个位置上对应的word id
    LS : 记录每个位置上对应的sentiment id
    ZS : 记录每个位置上对应的topid id
    IS : 记录单词是否是在情感字典中，如果是则不会对其进行更新sentiment 只更新其topic
    DR : 记录每个文档d中的rating的具体数值
    DRL : 记录每个文档中每个rating关联到的sentiment
    SS : 记录单词所属于的句子
    """
    cdef int i, j, k, w, z, d, l, z_new, l_new, res , update_flag, rating_sentiment, rating, rating_l_new, previous_S
    cdef int s, w_i, now_s
    cdef double r, dist_cum, r_dist_cum
    cdef int N = DS.shape[0]   #所有单词的数量
    cdef int n_rand = rands.shape[0]
    cdef int n_senti = nlz.shape[0]   #n种sentiment
    cdef int n_topics = nlz.shape[1]  #n种topic
    cdef double eta_sum = 0

    cdef int[:] RL
    cdef int[:] R

    cdef int n_ratings = 5 #先默认有5个档位的rating
    """记录不同sentiment l 和topic z的概率分布"""
    cdef double* dist_sum = <double*> malloc( n_topics * n_senti * sizeof(double))
    cdef double* r_dist_sum = <double*> malloc(n_ratings * n_senti * sizeof(double))

    if dist_sum is NULL:
        raise MemoryError("Could not allocate memory during sampling.")

    if r_dist_sum is NULL:
        raise MemoryError("Could not allocate memory during sampling.")

    previous_S = -1


    with nogil:
        for i in xrange(N):

            now_s = SS[i]
            '''先固定一个句子的情感'''
            #如果和之前的句子是不一样的,对句子重新采样情感
            if now_s != previous_S:

                #如果是第一次到这个句子
                previous_S = now_s

                '''先对同一个句子的count改变'''
                for w_i in xrange(i, N):
                    '''是同一个句子的'''
                    if SS[w_i] == previous_S:
                        w = WS[w_i]  # 单词w
                        d = DS[w_i]  # 对应的document id
                        z = ZS[w_i]  # 对应的topic z
                        l = LS[w_i]  # 对应的sentiment l
                        s = SS[w_i]  # 对应的sentiment id

                        dec(nd[d])  # document d的单词数量-1
                        dec(ndl[d, l])  # document d中sentiment l的单词数量-1
                        dec(ndlz[d, l, z])  # document d中sentiment l和topic z的数量-1
                        dec(nlzw[l, z, w])  # 单词w和sentiment l, topic z的组合数量-1
                        dec(nlz[l, z])  # sentiment l和topic z的组合数量-1
                    else:
                        break
                dist_cum = 0  # 相当于一个累积的概率分布 P

                """对每一种sentiment l和sentiment z计算"""
                '''对于每一种sentiment'''
                for j in xrange(n_senti):
                    '''对于每一种topic'''
                    for k in xrange(n_topics):
                        # eta is a double so cdivision yields a double
                        # 采样过程中和JST不同
                        dist_cum += (nlzw[j, k, w] + beta_lzw[j, k, w]) / (nlz[j, k] + betasum_lz[j, k]) \
                                    * (ndlz[d, j, k] + alpha_lz[j, k]) / (ndl[d, j] + alphasum_l[j]) \
                                    * (ndl[d, j] + sigma * mdl[d, j] + gamma_dl[d, j]) / (
                                                nd[d] + sigma * md[d] + gammasum_d[d])
                        index = j * n_topics + k
                        dist_sum[index] = dist_cum

                r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
                res = searchsorted(dist_sum, n_senti * n_topics, r)
                l_new = res / n_topics  # 随机采样一个new sentiment
                z_new = res % n_topics  # 随机采样一个new topic

                '''将新的topic和sentiment赋值给句子中的每个单词'''
                for w_i in xrange(i, N):
                    '''是同一个句子的'''
                    if SS[w_i] == previous_S:
                        w = WS[w_i]  # 单词w
                        d = DS[w_i]  # 对应的document id
                        s = SS[w_i]  # 对应的sentiment id

                        ZS[w_i] = z_new
                        LS[w_i] = l_new

                        inc(nd[d])
                        inc(ndl[d, l_new])
                        inc(ndlz[d, l_new, z_new])
                        inc(nlzw[l_new, z_new, w])
                        inc(nlz[l_new, z_new])
                    else:
                        break


            """接下来是对rating部分进行采样，要在每个文档结束后对文档的中的rating分布进行采样更新 
            但因为之前将所有document展开为一维，所以要判断是否到下一个document需要对比document id"""

            update_flag = 0
            if i+1 < N:
                """此时说明这个文档结束了"""
                if DS[i+1] != d:
                    update_flag = 1
            else:
                """说明是最后一个文档了，因此也更新"""
                update_flag = 1

            if update_flag==1:

                R = DR[d] #一个文档的评分数据
                RL = DRL[d]  #一个文档的评分关联的sentiment

                """记录采样不同rating下sentiment l的概率分布"""

                r_dist_cum = 0
                for j in xrange(R.shape[0]):

                    """对于每个rating"""
                    rating = R[j] - 1           #rating的数值,rating 1-5 对应位置是0-4 所以-1
                    rating_sentiment = RL[j] #rating关联的sentiment
                    '''对这些位置减一'''
                    dec(mdl[d, rating_sentiment])  # 文档d中sentiment l 数量-1
                    dec(md[d])  # rating r 关联的所有sentiment 数量-1
                    dec(mlr[rating_sentiment, rating])  # 文档d中rating r关联sentiment l的数量-1
                    dec(ml[rating_sentiment])  # 关联sentiment l 的rating的数量

                    """对于每种sentiment l"""
                    for k in xrange(n_senti):

                        r_dist_cum += (ndl[d, rating_sentiment] + sigma * mdl[d,rating_sentiment] + gamma_dl[d, rating_sentiment]) \
                                      / (nd[d] + sigma * md[d]+ gammasum_d[d])  \
                        * (mlr[rating_sentiment, rating] + delta_lr[rating_sentiment, rating]) / (ml[rating_sentiment] + deltasum_l[rating_sentiment])

                        index = j * n_ratings + k
                        r_dist_sum[index] = r_dist_cum

                r = rands[i % n_rand] * r_dist_cum  # dist_cum == dist_sum[-1]
                res = searchsorted(r_dist_sum, n_ratings * n_senti, r)
                rating_l_new = res / n_ratings  # 随机采样一个new sentiment

                #DR是固定的 不会更新
                DRL[d][j] = rating_l_new   #rating关联的sentiment更新

                inc(mdl[d, rating_l_new])  # rating r 关联的sentiment l 数量+1
                inc(md[d])  # rating r 关联的所有sentiment 数量+1
                inc(mlr[rating_l_new, rating])  # 文档d中rating r关联sentiment l的数量+1
                inc(ml[rating_l_new])  # 关联sentiment l 的rating的数量+1


        free(dist_sum)
        free(r_dist_sum)



cpdef double _loglikelihood(int[:] nd, int[:, :] ndl, int[:, :, :] ndlz, int[:, :, :] nlzw, 
                int[:, :] nlz, double alpha, double beta, double ga):
    cdef int z, d, l, w
    cdef int D = nd.shape[0]
    cdef int n_topics = nlz.shape[1]
    cdef int n_senti = ndl.shape[1]
    cdef int vocab_size = nlzw.shape[2]

    cdef double ll = 0

    with nogil:
        #calculate log p(w|z,l)
        lgamma_beta = lgamma(beta)
        lgamma_alpha = lgamma(alpha)
        lgamma_gamma = lgamma(ga)

        ll = n_senti*n_topics*lgamma(beta*vocab_size) - n_senti*n_topics*vocab_size*lgamma_beta
        for l in xrange(n_senti):
            for z in xrange(n_topics):
                ll -= lgamma(beta*vocab_size + nlz[l, z])
                for w in xrange(vocab_size):
                    ll += lgamma(beta + nlzw[l, z, w])

        #calculate log p(z|l,d)
        ll += n_senti*D*lgamma(alpha*n_topics) - n_senti*D*n_topics*lgamma(alpha)
        for l in xrange(n_senti):
            for d in xrange(D):
                ll -= lgamma(alpha*n_topics + ndl[d, l])
                for z in xrange(n_topics):
                    ll += lgamma(alpha + ndlz[l, d, z])

        #calculate log p(l|d)
        ll += D*lgamma(n_senti*ga) - D*n_senti*lgamma_gamma
        for d in xrange(D):
            ll -= lgamma(n_senti*ga + nd[d])
            for l in xrange(n_senti):
                ll += lgamma(ga + ndl[d, l])

        return ll