import json
import logging 
import os 

# Load json parameters into models 
class Hyperparameters():
    """Load params file into dict"""
    
    params = {}

    def __init__(self, path):
        with open(path, 'r') as f:
            self.params.update(json.load(f))

    def update(self, path):
        "Load new config"
        with open(path, 'r') as f:
            self.params.update(json.load(f))

    def save(self, path):
        with open(path, 'r+') as f:
            json.dump(self.params, f, indent=4) 
    
    def get(self, name):
        """Returns param value. Returns None if not found"""
        try: 
            return self.params[name]
        except:
            return None

    def get_string(self):
        s = '|'
        for key in self.params:
            s += f" {key}: {self.params[key]} |"
        return s

class BaseManager():
    """ Base manager class for runnning a training, testing, validation"""
    loggers = {}
    base_save_dir = ''
    model_checkpoints   = 'checkpoints/'
    images              = 'images/'
    reports             = 'reports/'

    h_params = None

    def __init__(self, base_dir, param_dir, debug=False):
        # Perform sanity checks 
        if not isinstance(base_dir, str):
            raise Exception("Base dir must be a string")
        if len(base_dir) == 0:
            raise Exception(f"Base dir cannot be {base_dir}")

        if not isinstance(param_dir, str):
            raise Exception("Params file must be a string")
        if len(param_dir) == 0:
            raise Exception(f"Params file cannot be {param_dir}")
        if not os.path.isfile(param_dir):
            raise Exception(f"Params file not found: {param_dir}")

        if base_dir[-1] != '/':
            base_dir += '/' 

        if not os.path.isdir('./output/' + base_dir):
            os.makedirs('./output/' + base_dir)

        self.base_save_dir = './output/' + base_dir
        print(f"Directory: {self.base_save_dir}")

        self.h_params = Hyperparameters(param_dir)

        # Create folders

        if not os.path.isdir(self.base_save_dir + self.model_checkpoints):
            os.makedirs(self.base_save_dir + self.model_checkpoints)

        if not os.path.isdir(self.base_save_dir + self.images):
            os.makedirs(self.base_save_dir + self.images)

        if not os.path.isdir(self.base_save_dir + self.reports):
            os.makedirs(self.base_save_dir + self.reports)

        self.create_logger(name='system', debug=debug)
        self.create_logger(name='hyparam', debug=debug)
        
        # Log initial hyper 
        self.get_logger('hyparam').info("Loaded hyperparameters")
        self.get_logger('hyparam').info(self.h_params.get_string())

    def get_params(self) -> Hyperparameters:
        """Get a hyperparameter"""
        return self.h_params
    
    def get_cp_dir(self) -> str:
        """Get checkpoint dir"""
        return self.base_save_dir + self.model_checkpoints

    def get_img_dir(self) -> str:
        """Get checkpoint dir"""
        return self.base_save_dir + self.images

    def get_rt_dir(self) -> str:
        """Get reporting dir"""
        return self.base_save_dir + self.reports
    
    def get_logger(self, name) -> logging:
        """Return a logger"""
        try:
            return self.loggers[name]
        except:
            raise Exception(f"Logger not found {name}")

    def create_logger(self, name, debug):
        """ Creates a log"""

        logger = logging.getLogger(name)
        fh = logging.FileHandler(f"{self.get_rt_dir()}{name}.log")
        if debug:
            logger.setLevel(logging.DEBUG)
            fh.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            fh.setLevel(logging.INFO)
    
        fm = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fm)
    
        logger.addHandler(fh)
        logger.info('Created')

        self.loggers[name] = logger 

class TrainManager(BaseManager):
    """ Train manager """ 
    
    data_dir = ''

    def __init__(self, base_dir, param_dir, debug=False):
        super().__init__(base_dir=base_dir, param_dir=param_dir, debug=debug)

        # Create a special log for training
        self.create_logger(name='train', debug=debug)  

        data_dir = self.h_params.get('train_dir')

        if not isinstance(data_dir, str):
            raise Exception("data_dir must be a string")
        if len(data_dir) == 0:
            raise Exception(f"data_dir cannot be {data_dir}")
        if not os.path.isdir(data_dir):
            raise Exception(f"data_dir not found: {data_dir}")

        if data_dir[-1] != '/':
            data_dir += '/' 

        self.data_dir = data_dir
        self.get_logger('train').info(f"Data directory: {self.data_dir}")

    def get_data_dir(self) -> str: 
        return self.data_dir


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def average(self):
        return self.avg
