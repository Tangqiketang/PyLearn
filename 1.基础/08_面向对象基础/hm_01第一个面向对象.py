def test_import():
    print("import触发第一个面向对象py的方法")

## python3.x 默认统一以object为基类
class A(object):
    #公有类属性。每个对象创建时，会复制类属性过去，值取决于类属性的当前的值
    static_attr = 1
    count =1

    def __init__(self,name,age):
        self.name = name #默认公有对象属性
        self._age = age #单个下划线，建议受保护的对象属性
        # 私有对象属性。两个下划线表示私有。会name Mangling被加上类前缀变成_A__password
        self.__password = None
        ## self指的是对象,self.count是对象属性，并不是类属性（只是名字相同而已）,所以并没有修改类的属性
        self.count = self.count + 1

    #通过get/set设置私有属性
    def set_password(self,password):
        self.__password = password
    def get_password(self):
        return self.__password

    ##静态方法。
    @staticmethod
    def staticmethod():
        print('执行staticmethod')
        pass
    #类方法。cls就是类的引用，和self一样
    @classmethod
    def myclassmethod(cls):
        pass
    #实例公有方法。访问实例属性
    def foo(self):
        print(self._age)
    #私有方法
    def __privatemethod(self):
        pass
    #内建方法，魔法方法。如print(obj1 + obj2)时,# 自动调用此方法。
    def __add__(self,other):
        self._age += other._age

a = A("wma",12)
b = A("wmb",13)
##优先从对象属性查找，找不到就从类中找
a.count = 99
print("类属性A.count=",A.count)
print("对象属性a.count",a.count)
print("对象属性b.count",b.count)
A.myclassmethod()
A.staticmethod()


class Cat(object):
    #重写new方法
    def __new__(cls, *args, **kwargs):
        print("__new__创建对象，分配空姐")
        ###return super().__new__(cls)
        return object.__new__(cls)

    ## 重写构造方法.默认内置无参
    def __init__(self,myname):
        self.myname=myname
        #私有属性，前面加__
        self.__sex = "boy"
        print("初始化开始")

    # 默认方法
    def __str__(self):
        return "mystring:"+self.myname
    def __del__(self):
        print("del++++++++++")

tom = Cat("WmName1")
print(tom)
# 就算不写关键字 del tom， 也会回收。
del tom

##############################################单例模式
class Singleton(object):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
