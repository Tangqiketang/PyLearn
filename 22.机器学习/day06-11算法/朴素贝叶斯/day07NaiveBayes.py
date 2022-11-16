"""

“朴素”贝叶斯，朴素就是特征独立

应用：文档分类
    先用tfidf获取每篇文章中的重要词，获取到特征。

======================================================================================
文档分类  P（科技|文档）  文档1： 词1  词2  词3   ==》 P（文档类别1 | 文档特征值F1,F2,F3..）
        P（娱乐|文档）   文档2： 词1  词2  词3   ==》 P（文档类别2 | 文档特征值F1,F2,F3..）


P(C|F1,F2,F3...) = P(F1,F2,F3|C) × P(C) / P(F1,F2,F3...)
注解：P(C|F1,F2,F3...) 表示 F1,F2,F3的条件下，是C类型文档的概率。
    P(科技类|词1,词2,词3...) = P(F词1,词2,词3...|科技类) × P(科技) / P(F词1,词2,词3...)

P（C|W）=P（W|C）×P（C）/P（W）
    P(w|C):给定类别下，文档中出现词的概率。其中某个词的概率 P（F1|C）= F1在C类所有文档中的次数/C类文档下所有词出现的次数和
    P(C):类别数/总文档数量
    P（W)： 被预测文档中每个词出现的概率
"""

