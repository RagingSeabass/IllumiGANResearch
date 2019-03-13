import json
import logging 
import os 

# Load json parameters into models 
class Parameters():
    """Load params file into dict"""
    
    params = {}

    def __init__(self, path):
        with open(path, 'r') as f:
            self.params = json.load(f)

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
            raise Exception(f"Error: Parameter is not found '{name}'")

    def get_string(self):
        s = 'Loaded \n'
        for key in self.params:
            s += f"| {key} | {self.params[key]} |\n"
        return s

class BaseManager():
    """ Base manager class for runnning a training, testing, validation"""
    loggers = {}

    model_checkpoints   = 'checkpoints/'
    images              = 'images/'
    reports             = 'reports/'

    is_train = False

    def __init__(self, base_dir, options_f_dir, hyperparams_f_dir):
        # Perform sanity checks 
        if not isinstance(base_dir, str):
            raise Exception("Base dir must be a string")
        if len(base_dir) == 0:
            raise Exception(f"Base dir cannot be {base_dir}")
        
        if base_dir[-1] != '/':
            base_dir += '/' 

        if not os.path.isdir('./output/' + base_dir):
            os.makedirs('./output/' + base_dir)

        self.check_param_file(options_f_dir)
        self.check_param_file(hyperparams_f_dir)

        self.base_save_dir = './output/' + base_dir
        print(f"Directory: {self.base_save_dir}")

        self.hyper_params = Parameters(hyperparams_f_dir)
        self.options = Parameters(options_f_dir)

        # Do some sanity checks 

        if self.options.get('max_dataset_size') < self.hyper_params.get('batch_size') and self.options.get('max_dataset_size') > 0:
            raise Exception("Batch size must be smaller than dataset size")

        # Create folders
        if not os.path.isdir(self.base_save_dir + self.model_checkpoints):
            os.makedirs(self.base_save_dir + self.model_checkpoints)

        if not os.path.isdir(self.base_save_dir + self.images):
            os.makedirs(self.base_save_dir + self.images)

        if not os.path.isdir(self.base_save_dir + self.reports):
            os.makedirs(self.base_save_dir + self.reports)

        self.create_logger(name='system', debug=self.options.get("debug"))
        self.create_logger(name='hyparam', debug=self.options.get("debug"))
        
        # Log initial settings 
        #self.get_logger('hyparam').info("Loaded hyperparameters")
        self.get_logger('hyparam').info(self.hyper_params.get_string())

        #self.get_logger('system').info("Loaded options")
        self.get_logger('system').info(self.options.get_string())

    def check_param_file(self, file_path):
        """Checks that the provided file path is ok"""
        if not isinstance(file_path, str):
            raise Exception("Params file must be a string")
        if len(file_path) == 0:
            raise Exception(f"Params file cannot be {file_path}")
        if not os.path.isfile(file_path):
            raise Exception(f"Params file not found: {file_path}")

    def get_hyperparams(self) -> Parameters:
        """Get the hyperparameters"""
        return self.hyper_params
    
    def get_options(self) -> Parameters:
        """Get the options"""
        return self.options

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

    def __init__(self, base_dir, options_f_dir, hyperparams_f_dir):
        super().__init__(base_dir=base_dir, options_f_dir=options_f_dir, hyperparams_f_dir=hyperparams_f_dir)

        self.is_train = True

        # Create a special log for training
        self.create_logger(name='train', debug=self.options.get("debug"))  

        data_dir = self.options.get('train_dir')

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
