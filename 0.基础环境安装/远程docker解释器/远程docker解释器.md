# 1.基础镜像 python:3.11-slim

# 2.容器中安装ssh
    ## 更新并安装 openssh-server
    apt update && apt install -y openssh-server
 
    ## 创建 SSH 启动目录
    mkdir -p /var/run/sshd
 
    ## 设置 root 密码（SSH 要求）
    passwd root
    ## 输入密码，比如：docker123
 
    ## 允许 root 登录
    sed -i 's/#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
    sed -i 's/#*PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
 
    ## 启动 SSH 服务
    /usr/sbin/sshd

