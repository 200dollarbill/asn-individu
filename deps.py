import pickle

class handler():
    def __init__(self, time, value, filename):
        self.time = time
        self.value = value
        self.filename = filename

    @staticmethod
    def save(index, pleth, filename):
        if len(index) and len(pleth) != 0:
            dataobj = handler(index, pleth, filename)
            with open(f"{filename}.dat", 'wb') as file:
                pickle.dump(dataobj, file)
    
    @staticmethod
    def load(filename):
        with open(f"{filename}.dat", 'rb') as file:
            dataobj = pickle.load(file)
            return dataobj