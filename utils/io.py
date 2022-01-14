import pickle
from bson.binary import Binary


def model_serialize(model):
    return Binary(pickle.dumps(model, protocol=2))

def model_deserialize(model):
    return pickle.loads(model)
