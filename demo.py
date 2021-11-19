import h5py
import numpy as np
from app import retrieval

from utils.retrieval_model import Retrieval
from utils.retrieval_model import get_arg

retrieval_model = Retrieval(get_arg())

print("Query: ")