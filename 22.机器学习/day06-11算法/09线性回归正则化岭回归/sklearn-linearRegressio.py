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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
import joblib
import pandas as pd
import numpy as np


def mylinearAndSaveModel(type):
    """
    线性回归预测房子价格 并保存模型
    type=1梯度下降 2正规方程 3岭回归
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
        ##使用梯度下降进行预测
        model = SGDRegressor()
    elif type==2:
        ##正规方程
        model = LinearRegression()
    else:
        ##岭回归 alpha正则化力度
        model = Ridge(alpha=1.0)
    model.fit(x_train,y_train)
    print("权重参数coef:",model.coef_)
    joblib.dump(model,"./sklearnmodel/test.pkl")

    y_model_predict = std_y.inverse_transform(model.predict(x_test))
    print("梯度下降测试集里面每个房子的预测价格：", y_model_predict)
    print("均方误差:",mean_squared_error(std_y.inverse_transform(y_test),y_model_predict))


def mylinear():
    """
    线性回归直接预测房子价格
    :return: None
    """
    # 获取数据
    lb = load_boston()
    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    print(y_train, y_test)
    # 进行标准化处理(?) 目标值处理？
    # 特征值和目标值是都必须进行标准化处理, 实例化两个标准化API
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    # 目标值
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train)
    y_test = std_y.transform(y_test)
    # 预测房价结果.从标准化的预测值变回之前的值
    model = joblib.load("./tmp/test.pkl")
    y_predict = std_y.inverse_transform(model.predict(x_test))
    print("保存的模型预测的结果：", y_predict)
    # estimator预测
    # 正规方程求解方式预测结果
    # lr = LinearRegression()
    #
    # lr.fit(x_train, y_train)
    #
    # print(lr.coef_)
    # 保存训练好的模型
    # joblib.dump(lr, "./tmp/test.pkl")

    # # 预测测试集的房子价格
    # y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    #
    # print("正规方程测试集里面每个房子的预测价格：", y_lr_predict)
    #
    # print("正规方程的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))
    #
    # # 梯度下降去进行房价预测
    # sgd = SGDRegressor()
    #
    # sgd.fit(x_train, y_train)
    #
    # print(sgd.coef_)
    #
    # # 预测测试集的房子价格
    # y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    #
    # print("梯度下降测试集里面每个房子的预测价格：", y_sgd_predict)
    #
    # print("梯度下降的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))
    #
    # # 岭回归去进行房价预测
    # rd = Ridge(alpha=1.0)
    #
    # rd.fit(x_train, y_train)
    #
    # print(rd.coef_)
    #
    # # 预测测试集的房子价格
    # y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    #
    # print("梯度下降测试集里面每个房子的预测价格：", y_rd_predict)
    #
    # print("梯度下降的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))

    return None

if __name__ == '__main__':
    mylinearAndSaveModel(3)