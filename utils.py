import json

# Load json parameters into models 
class Params():
    
    def __init__(self):
        with open(self, path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def update(self, path):
        "Load new config"
        with open(self, path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, path):
        with open(self, path) as f:
            json.dump(self.__dict__, f, indent=4) 

    @property
    def dict(self):
        "Gives dict-like access to Params instance by params.x"
        return self.__dict__

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
