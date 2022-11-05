#########if elif else
name = "wm"
if name == 'wm':
    print("wm---yye")
elif name == "yy":
    print("yyyy")
else:
    print("xxxxx")


##########for循环
for x in "abcd":
    print(x)
###利用角标
list1=["a","b","c","d"]
for i in range(len(list1)):
    print(list1[i])

###打印之后返回的是每个print的返回值9个None的列表
xx = [print(x) for x in range(10)]
print("xx",xx)


