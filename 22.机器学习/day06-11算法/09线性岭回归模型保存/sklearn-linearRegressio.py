"""
标准化：实现跨量冈

求解方式：
    正规方程LinearRegression：逆矩阵求解，
        缺点：1.特征数量较小（小于1W）才适用
             2.只适用线性模型，不适用于逻辑回归等模型
        优点：一次运算/不用学习率
    梯度下降SGDRegression：
        缺点：1.要多次迭代2.选择学习率
        优点：
            适用于各种类型的模型


线性回归-过拟合-正则化-岭回归

岭回归：
    为了解决过拟合的问题，带有正则化L2

"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor,  Ridge, LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
import joblib
import pandas as pd
import numpy as np


def mylinearAndSaveModel(type):
    """
    线性回归预测房子价格 并保存模型
    type=1正规方程求权重
         2随机梯度下降球权重，只是和1的求解方式不同
         3岭回归对权重做正则化
    最终评判损失还是最小二乘法
    :return:
    """
    lb = load_boston()
    print(lb.feature_names)
    x_train,x_test,y_train,y_test = train_test_split(lb.data,lb.target,test_size=0.25)
    print(y_train,y_test)
    ##测试集和目标值做标准化处理
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    ##目标值需要另外的标准化，列数量都不一样
    std_y = StandardScaler()
    y_train = std_y.fit_transform(np.array(y_train).reshape(-1,1))
    y_test = std_y.transform(np.array(y_test).reshape(-1,1))
    if type==1:
        ##正规方程
        model = LinearRegression()
    elif type==2:
        ##使用梯度下降进行预测
        model = SGDRegressor()
    else:
        ##岭回归 alpha正则化力度
        # param={"alpha":[0.0001,0.001,0.01,0.1,1,2,3]}
        # model_core = Ridge(alpha=1.0)
        # model = GridSearchCV(model_core,param,cv=4)
        # model.fit(x_train, y_train)
        # print("type=3时，网格搜索得到最佳alpha超参数:",model.best_params_)
        # print("type=3时，得到最佳估计器:",model.best_score_)
        # return
        model = Ridge(alpha=3)

    model.fit(x_train,y_train)
    print("权重参数coef:",model.coef_)
    joblib.dump(model,"./sklearnmodel/test.pkl")

    y_model_predict = std_y.inverse_transform(model.predict(x_test))
    print("梯度下降测试集里面每个房子的预测价格：", y_model_predict)
    print("均方误差:",mean_squared_error(std_y.inverse_transform(y_test),y_model_predict))


def predict():
    """
    使用之前保存的模型进行预测
     :return:
    """
    lb = load_boston()
    print(lb.feature_names)
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    print(y_train, y_test)
    ##测试集和目标值做标准化处理
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    ##目标值需要另外的标准化，列数量都不一样
    std_y = StandardScaler()
    y_train = std_y.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test = std_y.transform(np.array(y_test).reshape(-1, 1))

    model = joblib.load("./sklearnmodel/test.pkl")
    y_predict = std_y.inverse_transform(model.predict(x_test))
    print("保存的模型预测的结果:",y_predict)
    return None

if __name__ == '__main__':
    mylinearAndSaveModel(3)
    #predict()