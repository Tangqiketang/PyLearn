"""

“朴素”贝叶斯，朴素就是特征独立

应用：文档分类
    先用tfidf获取每篇文章中的重要词，获取到特征。

======================================================================================
文档分类  P（科技|文档）  文档1： 词1  词2  词3   ==》 P（文档类别1 | 文档特征值F1,F2,F3..）
        P（娱乐|文档）   文档2： 词1  词2  词3   ==》 P（文档类别2 | 文档特征值F1,F2,F3..）



P(科技类|词1,词2,词3...) = P(F词1,词2,词3...|科技类) × P(科技) / P(F词1,词2,词3...)
P（C|W）=P（W|C）×P（C）/P（W）
    P(w|C):给定类别下，文档中同时出现这些词的概率。其中某个词的概率 P（F1|C）= F1在C类所有文档中的次数/C类文档下所有词出现的次数和
    P(C):类别数/总文档数量
    P（W)： 被预测文档中每个词出现的概率

求出这些特征属于哪个类别的概率最高，P（科技类|词1,词2,词3...） P（娱乐类|词1,词2,词3...） P（汇总类|词1,词2,词3...）

增加啦普拉斯平滑系数,否则会有概率为0的问题：
P（F1|C） = （Ni+a）/（N+am）=（Ni+1）/(N+1×特征值个数),a系数默认1 from sklearn.naive_bayes import MultinomialNB（alpha=1.0）

"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_iris,fetch_20newsgroups,load_boston
from sklearn.model_selection import train_test_split
###标准化
from sklearn.preprocessing import StandardScaler
###特征提取,可以有text/image/
from sklearn.feature_extraction.text import TfidfVectorizer
##分类报告  召唤率
from sklearn.metrics import classification_report
import pandas as pd

def naiveBayes():
    """贝叶斯文本分类 通过新闻描述来预测是什么类型 """
    news = fetch_20newsgroups(subset="all")
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25)
    ##特征抽取
    tf = TfidfVectorizer()
    ##以训练集中词的列表进行每篇文章重要性统计
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    ### 进行朴素bayes预测,构建一个模型
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train,y_train)
    ##得出准确率
    print("准确率:",mlt.score(x_test,y_test))

    y_predict = mlt.predict(x_test)
    print("每个类别的精确率和召回率",classification_report(y_test,y_predict))
    return None

if __name__ == '__main__':
    naiveBayes()


