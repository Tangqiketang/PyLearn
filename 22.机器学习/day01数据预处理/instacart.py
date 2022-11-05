import pandas as pd
from sklearn.decomposition import PCA

"""
pd.read_csv("./data/instacart/orders.csv")
用户购买商品信息 降维

prior:   product_id,order_id    
product: product_id,aisle_id   
orders:  order_id,user_id
aisles:  aisle_id,aisle
"""
prior=pd.read_csv("./data/order_products_prior.csv")
product = pd.read_csv("./data/instacart/products.csv")
orders = pd.read_csv("./data/instacart/orders.csv")
aisles = pd.read_csv("./data/instacart/aisles.csv")
"""合并四张表,其实类似mysql join表"""
_mg = pd.merge(prior,product,on=["product_id","product_id"])
_mg = pd.merge(_mg,orders,on=["order_id","order_id"])
mt = pd.merge(_mg,aisles,on=["aisle_id","aisle_id"])

mt.head(10)
"""交叉表"""
cross = pd.crosstab(mt['user_id'],mt['aisle'])
"""PCA 保留90%的信息 """
pca = PCA(n_components=0.9)
data = pca.fit_transform(cross)
data
data.shape

