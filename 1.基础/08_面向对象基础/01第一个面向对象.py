class Cat:
    ## 构造方法.默认内置无参
    def __init__(self,myname):
        self.myname=myname
        #私有属性，前面加__
        self.__sex = "boy"

    def eat(self):
        print("%s吃东西" % self.myname)



    ##私有方法,方法前加__
    def __drink(self):
        print("喝水")

    # 默认方法
    def __str__(self):
        return "mystring:"+self.myname
    def __del__(self):
        print("del++++++++++")



tom = Cat("WmName1")
tom.eat()
# 给对象（不是类）添加新的属性.不推荐
tom.myname = "WmTom2"
tom.eat()
print(tom)
# 就算不写关键字 del tom， 也会回收。
del tom

