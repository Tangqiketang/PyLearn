# 导包
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine

# 数据库类型+数据库驱动名称://用户名:口令@机器地址:端口号/数据库名
engine = create_engine('mysql+pymysql://root:Gp1cpam4Q+U=@139.196.169.82:3396/data-manager?charset=utf8')
df = pd.read_sql('select * from attribute_code',engine) # 从数据库中导入数据表

print(df)