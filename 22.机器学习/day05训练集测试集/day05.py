from sklearn.datasets import load_iris,fetch_20newsgroups
from sklearn.model_selection import train_test_split

#加载数据集 。
#   load数据集比较小直接下载好的。
#   fetch用于分类的大数据集会默认保存在home/scikit_learn
# load_*和fetch返回的数据类型datasets.base.Bunch（字典类型）





li = load_iris()
#print(li)
#print(li.data)
#print(li.target)
#print(li.DESCR)
#print(li.feature_names)
#print(li.target_names)
"""划分训练集和测试集  经验值75% 25%"""
#x_train,x_test,y_train,y_test=train_test_split(li.data,li.target,test_size=0.25)
#print(x_train)


"""获取20篇新闻"""
news = fetch_20newsgroups(subset='all')
print(news.data)
