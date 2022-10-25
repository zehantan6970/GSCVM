import numpy as np
import logging
# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Log等级总开关  此时是INFO

# 第二步，创建一个handler，用于写入日志文件
logfile = "/home/light/gree/align/Hypothesis/registration_evaluate/scannet_eval/scene0002_00/scene0002_00_evaluation/scene0002_00_fragments/scene0002_00-evaluation/lowOverlap/gt.log"
# open的打开模式这里可以进行参考
fh = logging.FileHandler(logfile, mode='w')
# 输出到file的log等级的开关
fh.setLevel(logging.DEBUG)

# 第三步，再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
# 输出到console的log等级的开关
ch.setLevel(logging.WARNING)

# 第四步，定义handler的输出格式（时间，文件，行数，错误级别，错误提示）
formatter = logging.Formatter("")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 第五步，将logger添加到handler里面
logger.addHandler(fh)
logger.addHandler(ch)

# 日志级别
# logger.debug('debug级别，一般用来打印一些调试信息，级别最低')
# logger.info('info级别，一般用来打印一些正常的操作信息')
# logger.warning('waring级别，一般用来打印警告信息')
# logger.error('error级别，一般用来打印一些错误信息')
# logger.critical('critical级别，一般用来打印一些致命的错误信息，等级最高')

#
# DEBUG：详细的信息,通常只出现在诊断问题上
# INFO：确认一切按预期运行
# WARNING（默认）：一个迹象表明,一些意想不到的事情发生了,或表明一些问题在不久的将来(例如。磁盘空间低”)。这个软件还能按预期工作。
# ERROR：更严重的问题,软件没能执行一些功能
# CRITICAL：一个严重的错误,这表明程序本身可能无法继续运行
if __name__=="__main__":
    gtTxtPath="/home/light/gree/align/Hypothesis/registration_evaluate/scannet_eval/scene0002_00/scene0002_00_evaluation/scene0002_00_fragments/scene0002_00-evaluation/lowOverlap/lowgt.log"
    # ----------------------------------------------------------------------------------
    # 从gt.log中选出一部分pairs
    # ----------------------------------------------------------------------------------
    num=50
    with open(gtTxtPath,mode="r") as f:
        lines=f.readlines()
        linesLength=len(lines)
        # 对于gt.info把5改为7,并且把59，60两行注释去掉
        colNum=5
        jumpDistance=int(linesLength/num/colNum)
        for i in range(0,linesLength,jumpDistance*colNum):
            logger.info(lines[i].strip())
            logger.info(lines[i+1].strip())
            logger.info(lines[i+2].strip())
            logger.info(lines[i+3].strip())
            logger.info(lines[i+4].strip())
            # logger.info(lines[i+5].strip())
            # logger.info(lines[i+6].strip())
