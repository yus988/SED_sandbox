import numpy as numpy
import os
from PIL import Image
import glob
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import torch
from torch import utils
from facenet_pytorch import MTCNN, InceptionResnetV1

USE_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
VALID_DIR = "data/jaffedabse/"


def validation(model):
    persons = []
    faces = []
    mtcnn = MTCNN()
    for f in glob.glob(f'{VALID_DIR}/*.tiff'):
        p = os.path.basename(f)
        p = p[:2]
        img = Image.open(f)
        img = img.convert('RGB')
        img_cropped = mtcnn(img)
        persons.append(p)
        faces.append(img_cropped)
    model.to(USE_DEVICE)
    model.eval()
    