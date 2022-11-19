"""
K近邻算法：
    优点：无需估计器，无需训练。
    缺点：懒惰算法，计算内存消耗大
         必须指定K值，K值指定不当分分类精度不高
    场景：小数据量，几千几万样本


1.有些特征数据相差很大的问题？ 标准化！！

============================================================================================
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto')
1.n_neighbors：int,可选（默认= 5），k_neighbors查询默认使用的邻居数
2.algorithm：{‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}，可选用于计算最近邻居的算法：‘ball_tree’将会使用 BallTree，
 ‘kd_tree’将使用 KDTree。‘auto’将尝试根据传递给fit方法的值来决定最合适的算法。 (不同实现方式影响效率)
"""

from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def knncls():
    """
    K-近邻预测用户签到位置。
    row_id,x,y,accuracy,time,place_id
    https://www.kaggle.com/c/facebook-v-predicting-check-ins
    :return:None
    """
    # 读取数据
    data = pd.read_csv("./data/train.csv")
    print(data.head(10))
    # 1、缩小数据,查询数据晒讯
    data = data.query("x > 1.0 &  x < 1.25 & y > 2.5 & y < 2.75")
    # 处理时间的数据
    time_value = pd.to_datetime(data['time'], unit='s')
    print(time_value)

    # 把日期格式转换成 字典格式
    time_value = pd.DatetimeIndex(time_value)

    # 构造一些特征
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    # 把时间戳特征删除
    data = data.drop(['time'], axis=1)

    print(data)

    # 把签到数量少于n个目标位置删除.分组之后把次数大于3的拿出来，索引重新生成
    # place_count = data.groupby('place_id').count()
    # tf = place_count[place_count.row_id > 3].reset_index()
    # data = data[data['place_id'].isin(tf.place_id)]
    # 取出数据当中的特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)
    # 进行数据的分割训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 特征工程（标准化）
    std = StandardScaler()
    # 对测试集和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    # 进行算法流程 # 超参数
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(x_train, y_train)
    # # 得出预测结果
    y_predict = knn.predict(x_test)
    #
    print("预测的目标签到位置为：", y_predict)
    # # 得出准确率
    print("预测的准确率:", knn.score(x_test, y_test))
    return None


if __name__ == '__main__':
    knncls()