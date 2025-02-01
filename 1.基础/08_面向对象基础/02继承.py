## python3.x 默认统一以object为基类
class A(object):
    #公有类属性。不同对象共用类属性
    static_attr = 1
    count =1

    def __init__(self,name,age):
        self.name = name #默认公有对象属性
        self._age = age #单个下划线，建议受保护的对象属性
        # 私有对象属性。两个下划线表示私有。会name Mangling被加上类前缀变成_A__password
        self.__password = None

    #通过get/set设置私有属性
    def set_password(self,password):
        self.__password = password
    def get_password(self):
        return self.__password

    ##静态方法。
    @staticmethod
    def staticmethod():
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
print(a.static_attr)
b = A("wmb",13)
##优先从对象属性查找，找不到就从类中找
b.count = 99
print(b.count)
A.myclassmethod()
A.staticmethod()

class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def eat(self):
        print("animal eat")
    def drink(self):
        print("animal drink")

class Player:
    def play(self):
        print("player play")
##################################多继承
class Dog(Animal,Player):    #重写
    def eat(self):
        print("dog eat")
    def bark(self):
        print("dog bark")
dog1 = Dog("wangcai",12)
dog1.eat()
dog1.play()

#############################
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def play_with_dog(self,dog):
        print("person play_with_dog %s" % dog.name)

dogWW = Animal("ww",12)
dogMM =Dog("mm",12)
person = Person("xiaoming",12)
person.play_with_dog(dogWW)
person.play_with_dog(dogMM)