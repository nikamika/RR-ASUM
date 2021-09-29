import sys
import ASUMGibbs


def main():
    document_file_path = '../data/Clothes.csv'
    output_file_name = '../result'
    ASUM_Model = ASUMGibbs.ASUMGibbs(num_topics = 5, senti_words_path= '../preprocessing/mpqa.constraint', doc_file_path=document_file_path)
    ASUM_Model.run(max_iter=2000)
    ASUM_Model.export_result(output_file_name)

    #计算文档情感准确率
    ASUM_Model.doc_sentiment_acc()

if __name__ == '__main__':
    main()
