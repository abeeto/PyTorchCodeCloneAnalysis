from  codebase.config.configReader  import *
from codebase.logger.logger import *
import pprint

config = ConfigReader("configFiles/config2.yaml")
cfg = config.getConfig()

pp = pprint.PrettyPrinter()
pp.pprint(cfg)

logCfg = cfg["logging"]

objLog = CustomLogger(logCfg)
logger = objLog.getLogger()

logger.debug("debug msg")
logger.info("info msg")
objLog.disableLogging()

logger.debug("after disabling debug msg")
logger.info("after disabling info msg")

objLog.reEnableLogging()
logger.debug("after enabling debug msg")
logger.info("after enabling info msg")


print("*"*6,"hello","*"*6)
