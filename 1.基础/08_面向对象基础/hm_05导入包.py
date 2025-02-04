#引入整个包文件。包文件中有__init__会把所有包导入
import mypackage
#
mypackage.SendMessageUtil.send_message("wm发送消息")
mypackage.ReceiveMessageUtil.receive_message("wm接收消息")