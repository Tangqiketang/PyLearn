from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import numpy as np

"""
   包含了字典类型/文本类型转换成sparse矩阵，中文切词/tfidf重要性，
   归一化/标准化/
   缺失值处理
   pca自动筛选主成分
"""




def dictvec():
    """
    字典类型的数据抽取，转成数值类型矩阵 sparse为了节约内存
    :return: None
    """
    # 实例化
    dict = DictVectorizer(sparse=False)
    # 调用fit_transform
    data = dict.fit_transform([{'city': '北京','temperature': 100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature': 30}])
    # 获取特征名称
    print("DictVect.getFeatureName对应了下面的Inverse后的矩阵的每列字段:",dict.get_feature_names())
    print("DictVect.getInverseTransform:",dict.inverse_transform(data))
    print(data)
    return None


def countvec():
    """
    对文本进行特征值化. 文本分类情感分析。每个关键字作为一个特征。注意这里中间有空格！需要用jieba
    :return: None
    """
    cv = CountVectorizer()
    data = cv.fit_transform(["人生 苦短，我 喜欢 python", "人生漫长，不用 python"])
    print("CountVect.getFeatureName:",cv.get_feature_names())
    print("CountVect.data.toarray:",data.toarray())
    return None

def cutword():
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")
    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)
    # 把列表转换成字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)
    return c1, c2, c3

def hanzivec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cutword()
    ###返回词频矩阵
    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print("hanzi.CountVect.getFeatureName:",cv.get_feature_names())
    print("hanzi.CountVect.data.toarray:",data.toarray())
    return None



def tfidfvec():
    """
    中文特征值化.  tf idf
    term frequency 词的频率出现的次数,在其他文本中不常出现说明区分度高
    idf 逆文档频率 inverse document frequency log(总文档数量/该词出现的文档数量)
    :return: None
    """
    c1, c2, c3 = cutword()
    tf = TfidfVectorizer()
    data = tf.fit_transform([c1, c2, c3])
    print("TFidf.getFeatureName:",tf.get_feature_names())
    print("TFidf.data.toarray:",data.toarray())
    return None

"数值类型" \
    "标准化-常用：" \
    "归一化-场景有限：解决特征同等重要的问题(对异常点处理不太好，和最大值最小值关系太大)，默认映射到[0-1]可以设置" \
    "缺少值：" \
    "正则化： 解决过拟合"
"类别型：one hot编码"
"时间型：时间切分"

def mm():
    """min max 缩放
    归一化处理-有场景限制不常用 默认0-1，设置到[2-3]之间，
    :return: NOne
    """
    mm = MinMaxScaler(feature_range=(2, 3))
    data = mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])
    print("MinMaxScaler.data:",data)
    return None

def stand():
    """
    标准化缩放，对归一化的一种改进。处理之后聚集在标准正太分布(均值为0,方差为1)均值0附近
    fit_transform(numpy array)
    mean_ 原始数据中每列特征平均值
    std_ 每列特征的方差
    fit_transform多个数据，后续用transform，否则fit之后标准会变
    :return:
    """
    std = StandardScaler()
    data = std.fit_transform([[ 1., -1., 3.],[ 2., 4., 2.],[ 4., 6., -1.]])
    print("StandardScaler.data:",data)
    return None

def im():
    """
    第一种缺失值处理，用平均值
    missing_values='NaN', strategy='mean',ax
    :return:NOne

    ##第二种方式缺失值处理,丢弃无用的数据
    data = data.replace(to_replace="?",value=np.nan)
    data = data.dropna()

    """
    # NaN, nan
    im = SimpleImputer(missing_values=np.nan, strategy="mean")
    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
    print("SimpleImputer.data:",data)
    return None

##数据降维：
"1特征选择：选择区分度高的特征," \
"   主要方法：" \
"       1.FIlter:VarianceThreshold 方差门槛选择方差大的" \
"       2.Embedded嵌入式：正则化（防止过拟合）,决策树" \
"       3.Wrapper包裹式 不常用" \
"2.主成分分析："


def var():
    """
    把没有明显特征的删除
    特征选择-删除低方差的特征。threshold=1.0表示方差小于1的删除掉
    :return: None
    """
    var = VarianceThreshold(threshold=1.0)
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print("VarianceThreshold.data:",data)
    return None

def pca():
    """
    场景：
        特征数量上百时
        一个图片几万个特征
    主成分分析进行特征降维：
        特征会减少，数据也会改变。用低维度表示高维度
        消减回归或聚类分析中
    n_components:
        小数 0-1  90-95%经验值
        整数  减少到的特征数量--一般不用
    :return: None
    """
    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])
    print("PCA.data:",data)
    return None

if __name__ == "__main__":
   dictvec()
   countvec()
   hanzivec()
   tfidfvec()
   mm()
   stand()
   im()
   var()
   pca()