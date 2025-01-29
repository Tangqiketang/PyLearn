## python3.x 默认统一以object为基类
class A(object):
    #定义类属性
    count = 0
    def __init__(self):
        A.count += 1
    ## 实例方法。访问实例属性
    def foo(self):
        print(self.count)
    ##类方法。cls就是类的引用，和self一样
    @classmethod
    def myclassmethod(cls):
        pass
    ##静态方法。
    @staticmethod
    def staticmethod():
        pass


a = A()
print(a.count)
b = A()
print(b.count)
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