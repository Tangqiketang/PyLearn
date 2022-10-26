# 单变量
counter = 100; print(counter)
# 同时赋值多个
a = b = c = 1
a, b, c = 1, 2, 3
print(a, b, c)
#################数字
num = 1
#################字符串
s1 = '0123456'
s2 = "444"
print("字符串拼接:",s1+s2)
print("从第2到第4位，左闭右开:", s1[2:4])
print("从第1个到第4个每隔2位取一个:", s1[1:4:2])
print("s1为 %s s2为 %s" % (s1, s2))
############### 元组 数组()
tuple1 = (1,"aa","bb")
print("tuple1:",tuple)
print(tuple1[0:2])
############## 列表list[]
list1 = [100, "文字", 20.3]
print("list1获取第0位:", list1[0])
print("list1获取第0位到第二位", list1[0:2])

list2 = list(("a", "b", "c"))
print("把元组转成列表:",list2)
list3=list("abc")
print("把字符串转成列表:",list3)
#################字典 map {}
dict1={"key1":"wm1","key2":"yy"}
print(dict1["key2"])
#################################数据类型转换
##强转成int,str(),list(),tuple()
numx1 = int(s2)
numx2 = int(s2)
print(numx1+numx2)