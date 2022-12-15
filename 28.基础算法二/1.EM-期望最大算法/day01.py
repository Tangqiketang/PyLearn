
"""
1.最大似然估计
    已知样本服从分布的模型，求模型的参数隐变量
2.EM expectation-Maximization
3.GMM 高斯混合模型

流程：
两个硬币 对应两个目标值。
两个概率对于 模型的隐变量。

1. 假设模型的隐变量
2. 求每个样本在每个目标值下的期望= 使用某个模型的概率×值
3. 累加，并求出隐变量。
4.将隐变量重新带入，迭代指导变量收敛


"""
