"""
决策树：模型主要是构造一个树形结构，根节点的特征是最能够减少信息熵的
       即第一个分支的信息增益最大
-（p1logp1+p2logp2+.....）
优点：
    简单的理解和解释，树木可视化。
    需要很少的数据准备，其他技术通常需要数据归一化，
缺点：
    决策树学习者可以创建不能很好地推广数据的过于复杂的树，
            这被称为过拟合。
    决策树可能不稳定，因为数据的小变化可能会导致完全不同的树被生成

改进：
    1.减枝cart算法,样本数量很少的分支裁剪掉
    2.随机森林
==================================================================================================
集成学习方法：
    多个分类器/模型，独立学习预测。这些预测最后结合成单预测，这样就优于单一的分类器。
    其中多个决策树组成的就叫做随机森林
随机森林：
    包含多个决策树的分类器，并且其输出类别是由多个分类器的结果总数最多的而定。类似投票。
    每次随机从样品中取出N个样品生成一颗树(同时得把样本放回池子)
优点：
    1.在当前所有算法中，具有极好的准确率
    2.能够有效地运行在大数据集上
    3.能够处理具有高维特征的输入样本，而且不需要降维
    4.能够评估各个特征在分类问题上的重要性
    5.对于缺省值问题也能够获得很好得结果


"""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def decision():
    """
    决策树对泰坦尼克号进行预测生死
    :return: None
    """
    # 获取数据
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    # 处理数据，找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']
    print(x)
    # 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)
    # 分割数据集到训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 进行处理（特征工程）特征-》类别-》one_hot编码
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(dict.get_feature_names())
    x_test = dict.transform(x_test.to_dict(orient="records"))
    # print(x_train)
    # 用决策树进行预测
    # dec = DecisionTreeClassifier()
    # dec.fit(x_train, y_train)
    # # 预测准确率
    # print("预测的准确率：", dec.score(x_test, y_test))
    # # 导出决策树的结构
    # export_graphviz(dec, out_file="./tree.dot", feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

    # 随机森林进行预测 （超参数调优）。
    """
    class  sklearn.ensemble.RandomForestClassifier(n_estimators=10,  criterion=’gini’,
     max_depth=None, bootstrap=True,  random_state=None)
    随机森林分类器
    n_estimators：integer，optional（default = 10） 森林里的树木数量
    criteria：string，可选（default =“gini”）分割特征的测量方法
    max_depth：integer或None，可选（默认 = 无）树的最大深度
    bootstrap：boolean，optional（default = True）是否在构建树时使用放回抽样
    """
    rf = RandomForestClassifier()
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    # 网格搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)
    print("准确率：", gc.score(x_test, y_test))
    print("查看选择的参数模型：", gc.best_params_)
    return None

if __name__ == '__main__':
    decision()