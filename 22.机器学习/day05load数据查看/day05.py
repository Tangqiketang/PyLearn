from sklearn.datasets import load_iris,fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""转化器 实现特征工程
    fit_transform = fit + transform
    后续多组数据用transform，否则数据标准不一样

估计机：
    用于分类的估计器：
        sklearn.neighbors
        sklearn.naive_bayes
        sklearn.linear_model.logisticRegression
        sklearn.tree 决策树与随机森林
    用于回归的估计器：
        sklearn.linear_model.LinearRegression
        sklearn.linear_model.Ridge    岭回归
"""


#加载数据集 。
#   load数据集比较小直接下载好的。
#   fetch用于分类的大数据集会默认保存在home/scikit_learn_data/
# load_*和fetch返回的数据类型datasets.base.Bunch（字典类型）

def descData():
    data = load_iris()
    print("data:",data)
    print("data.data:",data.data)
    print("data.target:",data.target)
    print("data.DESCR:",data.DESCR)
    print("data.feature_names:",data.feature_names)
    print("data.target_names:",data.target_names)
    """划分训练集和测试集  经验值75% 25%"""
    #x_train,x_test,y_train,y_test=train_test_split(li.data,li.target,test_size=0.25)
    #print(x_train)

def desDataNews():
    """获取20篇新闻"""
    news = fetch_20newsgroups(subset='all')
    print(news.data)



if __name__ == '__main__':
    descData()