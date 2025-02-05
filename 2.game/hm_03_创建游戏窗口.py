import pygame

pygame.init()

# 创建游戏的窗口 480 * 700
screen = pygame.display.set_mode((480, 700))

##暂停10秒钟
pygame.time.wait(10000)

## 在wsl2中，moba中查找 ps -aux|grep python;  kill -9 pid 解决无法关闭窗口的问题


pygame.quit()
