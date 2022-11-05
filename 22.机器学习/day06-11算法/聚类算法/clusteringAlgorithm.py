"""
假设要分为3类
1.随机选择三个样本点，作为三个类的中心点
2.计算其他数据距离这三个点的距离，把每个数据归类为距离最短的点
3.分成了3大类后，计算每个类的平均值，和初始三个样本点是否相同。
如果相同则停止;如果不同，把这三个平均值作为新的三个样品点，进行迭代

"""


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# In[2]:


# 读取四张表的数据
prior = pd.read_csv("./data/instacart/order_products__prior.csv")


# In[3]:


products = pd.read_csv("./data/instacart/products.csv")


# In[4]:


orders = pd.read_csv("./data/instacart/orders.csv")


# In[5]:


aisles = pd.read_csv("./data/instacart/aisles.csv")


# In[7]:


# 合并四张表到一张表  （用户-物品类别）
_mg = pd.merge(prior, products, on=['product_id', 'product_id'])
_mg = pd.merge(_mg, orders, on=['order_id', 'order_id'])
mt = pd.merge(_mg, aisles, on=['aisle_id', 'aisle_id'])


# In[8]:


mt.head(10)


# In[9]:


# 交叉表（特殊的分组工具）
cross = pd.crosstab(mt['user_id'], mt['aisle'])


# In[12]:


cross.head(10)


# In[13]:


# 进行主成分分析
pca = PCA(n_components=0.9)


# In[14]:


data = pca.fit_transform(cross)


# In[21]:


# 把样本数量减少
x = data[:500]
x.shape


# In[22]:


# 假设用户一共分为四个类别
km = KMeans(n_clusters=4)


# In[23]:


km.fit(x)


# In[24]:


predict = km.predict(x)


# In[25]:


predict


# In[27]:


# 显示聚类的结果
plt.figure(figsize=(10,10))


# In[28]:


# 建立四个颜色的列表
colored = ['orange', 'green', 'blue', 'purple']
colr = [colored[i] for i in predict]
plt.scatter(x[:, 1], x[:, 20], color=colr)


# In[29]:


plt.xlabel("1")
plt.ylabel("20")


# In[30]:


plt.show()


# In[32]:


# 评判聚类效果，轮廓系数
silhouette_score(x, predict)


# In[ ]:



