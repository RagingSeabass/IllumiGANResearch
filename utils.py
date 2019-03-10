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


