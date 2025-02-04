print("111111111111111111111111111111")

def say_hello():
    print("执行测试导入py的say hello方法")


def main():
    ##在本文件中执行时，name是__main__。 如果被其他文件导入则变成文件名字。
    print(__name__)
    say_hello()

if __name__ == '__main__':
    main()