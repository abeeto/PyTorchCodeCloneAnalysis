import os
import inspect
from utils import parseArguments, initSavePath, saveCode, initGamesLogger, attachSignalsHandler
from Results import Results

# init current file (script) folder
baseFolder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory

# parse arguments
args = parseArguments()

# init save path
save_path, train_path, folderName = initSavePath(args.results_dir)

# save source code
code_path = saveCode(save_path, baseFolder)

# init Results object
results = Results(save_path, folderName)

# init games logger
gamesLogger = initGamesLogger('Games', save_path)

# log PID
gamesLogger.info('PID:[{}]'.format(os.getpid()))

# handle SIGTERM
attachSignalsHandler(results, gamesLogger)
