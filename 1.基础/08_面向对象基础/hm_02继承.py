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