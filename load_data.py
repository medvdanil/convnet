import numpy as np
from skimage.color import gray2rgb, rgb2gray
from skimage import img_as_float
import cv2
from os import path
from collections import namedtuple
from matplotlib import pyplot as plt
from skimage.draw import line_aa, circle_perimeter_aa
import json

dataset = namedtuple('dataset', ('x', 'y', 'yp'))
widths = [
    1.0,
    0.5,
    0.7,
    0.5,
    0.7,
    ]
pairs = np.array([
       [ 0,  1, 1],
       [ 1,  2, 2],
     # [ 2,  6, 0],
       [ 3,  4, 2],
     # [ 3,  6, 0],
       [ 4,  5, 1],
       [ 6,  8, 0],
       [ 8,  9, 0],
     # [13,  8, 0],
       [10, 11, 3],
       [11, 12, 4],
     # [12,  8, 0],
       [13, 14, 4],
       [14, 15, 3]])

extend_k = 0.1

def imread(filename):
    img = cv2.imread(filename)
    if img is None:
        raise OSError('%s is not an image' % filename)
    return img[:, :, ::-1] # brg2rgb

def subrect(img, roi):
    x, y, w, h = roi
    x += w/2
    y += h/2
    w = int(w + 0.5)
    h = int(h + 0.5)        
    return img_as_float(cv2.getRectSubPix(img, (w, h), (x, y)))


def imshow(image):
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(image.shape) == 2:
        ax.imshow(image, plt.cm.Greys_r)
    else:
        ax.imshow(image)
    ax.axis('off')
    plt.show()
    
    
def batch_show(batch):
    shape = batch.shape[2:]
    batch = np.reshape(batch, (-1, *shape))
    n = int(np.sqrt(len(batch))) + 1
    img = np.zeros((n * shape[0], n * shape[1], shape[2]), dtype=batch.dtype)
    for k in range(len(batch)):
        i, j = k // n, k % n
        img[i * shape[0]:(i+1) * shape[0], j * shape[1]:(j+1) * shape[1]] = batch[k]
    imshow(img)   
    

def joint_poselet(img, p1, p2, t):
    vec = (np.array(p2) - np.array(p1)) * (0.5 + extend_k)
    c = (np.array(p2) + np.array(p1)) / 2.
    x0, y0 = img.shape[1::-1]
    x1, y1 = 0, 0
    for r in range(4):
        p = c + vec * (widths[t] if r % 2 == 1 else 1.0)
        x0, y0 = np.maximum(np.minimum(p, (x0, y0)), (0, 0))
        x1, y1 = np.minimum(np.maximum(p + 1, (x1, y1)), img.shape[1::-1])
        vec = np.array((-vec[1], vec[0]))
    x0, x1, y0, y1 = np.round((x0, x1, y0, y1)).astype(int)
    return img[y0:y1, x0:x1]

    
def extract_poselets(img, pts, shape):
    res = []
    #new_shape = np.array(img.shape)
    #ext = (new_shape[:2] * extend_k).astype(int)
    #new_shape[:2] += ext * 2
    #img_pad = np.zeros(new_shape, dtype=img.dtype)
    #img_pad[ext[0]:-ext[0], ext[1]:-ext[1]] = img
    for v1, v2, t in pairs:
        res.append(cv2.resize(joint_poselet(img, pts[v1], pts[v2], t), shape))
    #batch_show(np.array([res]))
    return res        
    
    
def load_data_atr(datadir, shape, n_attr, n_classes):
    f = open(path.join(datadir, 'labels.txt'), 'r')
    rows = f.readlines()
    f.close()
    
    f = open(path.join(datadir, 'annotation.json'), 'r')
    joints_dict = json.load(f)
    f.close()
    n = len(rows)
    x, y = list(), list()
    for i, row in enumerate(rows):
        fields = [field.strip() for field in row.split()]
        assert(len(fields) == 1 + 4 + n_attr)
        #try:
        img = imread(path.join(datadir, fields[0]))
        roi = list(map(float, fields[1:5]))
        if roi[0] != roi[0]:
            continue
        if False:
            subimg = cv2.resize(subrect(img, roi), shape)
            x.append([subimg])
        elif False:
            subimg1 = cv2.resize(subrect(img, (roi[0], roi[1], roi[2], roi[3] / 2)), shape)
            subimg2 = cv2.resize(subrect(img, (roi[0], roi[1] + roi[3] / 2, roi[2], roi[3] / 2)), shape)
            x.append([subimg1, subimg2])
        elif True:
            joints = joints_dict[fields[0]][0]
            x.append(extract_poselets(img, joints, shape))
            
        y.append(list(map(float, fields[5:])))
    y = np.array(y, dtype=int) + 1
    yp = np.zeros((len(y), n_attr, n_classes), dtype=np.float32)
    yp[np.arange(len(y)), y] = 1
    xy = dataset(np.array(x), y.astype(np.float32), yp)
    return xy

def draw_joints(img, pts):
    for i in range(len(pts)):
        clr = np.array([i & 1, (i >> 1) & 1, (i >> 2) & 1], dtype=float) * (1.0 - 0.5 * (i >> 3))
        p = pts[i][1], pts[i][0]
        if p[0] + 3 >= img.shape[0] or p[0] - 3 < 0 or p[1] + 3 >= img.shape[1] or p[1] - 3 < 0:
            continue
        r, c, v = circle_perimeter_aa(int(p[0] + 0.5), int(p[1] + 0.5), 2)
        for i in range(len(r)):
            img[r[i], c[i]] = clr * v[i] + img[r[i], c[i]] * (1. - v[i])
    return img
    
class AttributesDataset:
    def __init__(self, datadir, shape, n_attr):
        self.train = load_data_atr(path.join(datadir, 'train'), shape, n_attr, n_classes)
        self.test = load_data_atr(path.join(datadir, 'test'), shape, n_attr, n_classes)
        self.order = np.arange(len(self.train.x))
        self.offset = len(self.train.x)
    
    def next_batch(self, batch_size):
        assert(batch_size <= len(self.train.x))
        if self.offset + batch_size <= len(self.train.x):
            self.offset += batch_size
            return self.train.x[self.order[self.offset - batch_size:self.offset]], \
                    self.train.y[self.order[self.offset - batch_size:self.offset]]
        p = self.order[self.offset:]
        self.order = np.random.permutation(len(self.train.x))
        self.offset += batch_size - len(self.train.x)
        p = np.hstack((p, self.order[:self.offset]))
        return self.train.x[p], self.train.y[p]
