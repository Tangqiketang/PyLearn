"""
算法 ： 10逻辑回归
策略：  对数似然损失。 梯度下降会有多个最低点的问题。可以用多个初始化权重或调整学习率
优化：  梯度下降

根据样本数量大小，哪个类别少，判定概率值是指的这个类别！！！


线性回归的式子作为逻辑回归的输入。

二分类问题：
    广告点击率/是否为垃圾邮件/是否患病/是否金融诈骗/虚假帐号

1. 准备好 特征 和目标值
2. 建立模型 随机初始化准备一个权重w,一个偏置项b
    y_predict = x w + b
3. 求损失函数，loss对数似然函数。 h表示概率值，-log(h)从无穷介于0
               =  -log(h)  if y=1。表示目标值为1类时，概率为h时的损失
    cost(h，y) =
               = -log(1-h) if y=0。 表示目标值为0类时，概率为h的损失

    单独一个样品损失为
    -ylog(h)-(1-y)log(1-h)，y表示

"""

import pandas as pd
import  numpy as np
##import sklearn.model_selection as sm
###数据切分训练和测试集
from sklearn.model_selection import train_test_split
###标准化
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.metrics import classification_report

def logistic():
    """癌症预测 二分类"""
    ##构造列标签名字.样本字段描述
    mycolumn = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'Class']
    #### https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
    data = pd.read_csv("./logisticdata/breast-cancer-wisconsin.data",names=mycolumn)
    print(data)

    ##缺失值处理,丢弃无用的数据
    data = data.replace(to_replace="?",value=np.nan)
    data = data.dropna()
    ##进行数据的分割. 注意数据中mycolumn[0]列是序列号没用，1-9列是特征，10列是目标值
    x_train,x_test,y_train,y_test = train_test_split(data[mycolumn[1:10]],
                                                     data[mycolumn[10]],
                                                     test_size=0.25)
    ###对测试集和训练集进行相同的标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    ###逻辑回归预测 默认penalty正则化l2， C正则化力度，防止过拟合
    lg = LogisticRegression(penalty="l2",C=1.0)
    lg.fit(x_train,y_train)
    print("打印权重:",lg.coef_)
    print("准确率:",lg.score(x_test,y_test))

    ###癌症看的不是准确率，看召回率=预测为癌症的数量/所有真实癌症的数量.高召回率意味着更少的漏检
    ## https://zhuanlan.zhihu.com/p/369936908?ivk_sa=1024320u
    y_predict = lg.predict(x_test)
    print("召回率:",classification_report(y_test,y_predict,labels=[2,4],target_names=["良性","恶性"]))

    return None


if __name__ == '__main__':
    logistic()