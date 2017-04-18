import numpy as np
from skimage.color import gray2rgb, rgb2gray
from skimage import img_as_float
import cv2
from os import path
from collections import namedtuple
from matplotlib import pyplot as plt
from skimage.draw import line_aa, circle_perimeter_aa
import json
import pickle
from skimage.transform import AffineTransform

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

limb_idxes = [2, 3, 7, 8, 4, 5, 10, 11]
torso_idxes = [6, 9]
head_size_k = 1.9
extend_k = 0.1

def huge_dump(obj, filename):
    #joblib.dump(obj, filename, compress=9)
    f = open(filename, 'wb')
    pickle.dump(obj, f, protocol=4)
    f.close()
    print("data dumped to %s", filename)
    
    
def huge_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
    
def imread(filename):
    img = cv2.imread(filename)
    if img is None:
        raise OSError('%s is not an image' % filename)
    return img[:, :, ::-1] # brg2rgb


def subrect(img, xywh):
    x, y, w, h = xywh
    x += w/2
    y += h/2
    w = int(w + 0.5)
    h = int(h + 0.5)        
    return cv2.getRectSubPix(img, (w, h), (x, y))


def subimg_roi(img, roi):
    x = np.mean(roi[2:4])
    y = np.mean(roi[0:2])
    w = roi[3] - roi[2]
    h = roi[1] - roi[0]
    w = int(w + 0.5)
    h = int(h + 0.5)        
    return cv2.getRectSubPix(img, (w, h), (x, y))

    
def imshow(image):
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(image.shape) == 2:
        ax.imshow(image, plt.cm.Greys_r)
    else:
        ax.imshow(image)
    ax.axis('off')
    plt.show()
    
def dist2(p1, p2):
    return np.sum((np.array(p1) - p2) ** 2)

def dist(p1, p2=(0, 0)):
    return np.sqrt(np.sum((np.array(p1) - p2) ** 2))
    
def batch_show(batch):
    shape = batch.shape[2:]
    batch = np.reshape(batch, (-1, *shape))
    n = int(np.ceil(np.sqrt(len(batch))))
    img = np.zeros((n * shape[0], n * shape[1], shape[2]), dtype=batch.dtype)
    for k in range(len(batch)):
        i, j = k // n, k % n
        img[i * shape[0]:(i+1) * shape[0], j * shape[1]:(j+1) * shape[1]] = batch[k]
    imshow(img)   
    

def joint_poselet_old(img, p1, p2, t):
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


def calc_stats(joints_dict):
    mean_len = 0.0
    rel_lengths = np.zeros((len(joints_dict), len(pairs)))
    i = 0
    for _, (pts,) in joints_dict.items():
        rel_lengths[i] = [dist2(pts[v1], pts[v2]) for v1, v2, t in pairs]
        i += 1
    rel_lengths = np.sqrt(rel_lengths)
    rel_lengths = rel_lengths / np.mean(rel_lengths, axis=1)[:, None]
    return rel_lengths.mean(axis=0), np.sqrt(rel_lengths.var(axis=0))
    
    
def extract_poselets_old(img, pts, shape, stats=None):
    res = []
    dists = np.zeros(len(pairs), img.dtype)
    for i, (v1, v2, t) in enumerate(pairs):
        dists[i] = dist2(pts[v1], pts[v2])
    dists = np.sqrt(dists)
    dists /= dists.mean()
    mask = [stats is None or abs(dists[i] - stats[0][i]) < stats[1][i] \
            for i, (v1, v2, t) in enumerate(pairs)]
    #if np.count_nonzero(mask) != len(mask):
        #return None
    for i, (v1, v2, t) in enumerate(pairs):
        if mask[i]:
            res.append(cv2.resize(joint_poselet_old(img, pts[v1], pts[v2], t), shape))
        else:
            res.append(np.zeros(shape + img.shape[2:], dtype=img.dtype))
    #batch_show(np.array([res]))
    return res


def norm_limb(img, figure, shape, p1, p2):
    vec = (np.array(p2) - np.array(p1)) * (1 + extend_k)
    vlen = dist(vec)
    c = (np.array(p2) + np.array(p1)) / 2.
    p0 = c - vec / 2
    ang = np.arctan2(vec[1], vec[0])
    frame_size = (int(vlen + 0.5),) * 2
    #print('shape', np.array(shape[1::-1]) / np.array(frame_size), p1, p2)
    t = AffineTransform(translation=-c[1::-1]) + \
        AffineTransform(rotation=ang) + \
        AffineTransform(scale=np.array(shape[1::-1]) / np.array(frame_size)) + \
        AffineTransform(translation=np.array(shape[1::-1]) / 2)
    return cv2.warpAffine(img, t.params[:2], shape[1::-1], borderMode=cv2.BORDER_REFLECT)
        
        

def get_limbs(img, figure, shape):
    res = []    
    for v in figure[limb_idxes]:
        p1 = v[:2]
        p2 = v[2:4]
        if p1[0] < 0:
            res.append(np.zeros(shape + img.shape[2:], dtype=img.dtype))
            continue
        res.append(norm_limb(img, figure, shape, p1, p2))
    return res


def get_torso(img, figure, shape):
    idxes = []
    for i in torso_idxes:
        if figure[i][0] >= 0:
            idxes.append(i)
    if len(idxes) == 0:
        return np.zeros(shape + img.shape[2:], dtype=img.dtype)
    p1 = figure[idxes[0], :2]
    p2 = figure[idxes, 2:4].mean(axis=0)
    return norm_limb(img, figure, shape, p1, p2)


def extract_poselets(img, figure, shape):
    if figure is None:
        return [np.zeros(shape + img.shape[2:], dtype=img.dtype)] * 10
    head_roi = get_head_roi(figure)
    if head_roi[1] >= 0:
        res = [cv2.resize(subimg_roi(img, head_roi), shape)]
    else:
        res = [np.zeros(shape + img.shape[2:], dtype=img.dtype)]
    res += get_limbs(img, figure, shape)
    res.append(get_torso(img, figure, shape))
    #batch_show(np.array([res]))
    return res  
    
    
def symm_vert(x):
    not_list = False
    if type(x) != list:
        x = [x]
        not_list = True
    res = []
    for img in x:
        if img is None:
            res.append(None)
        else:
            res.append(img[:, ::-1])
    return res[0] if not_list else res
    

class AttributesDataset:
    def __init__(self, datadir, shape, n_attr, poselet_id=None):
        self.train = load_data_atr(path.join(datadir, 'train'), shape, n_attr, poselet_id=poselet_id)
        self.test = load_data_atr(path.join(datadir, 'test'), shape, n_attr, poselet_id=poselet_id, is_test=True)


class EmptyDataset:
    def __init__(self):
        self.train = None
        self.test = None


class Dataset:
    def __init__(self, x, y, names, px=None):
        self.x = x
        self.y = y
        self.px = px
        self.names = names


def merge_poselets_ds(ds):
    name_to_i = list()
    names = set()
    for d in ds:
        names = names.union(set(d.names))
        name_to_i.append(dict(zip(d.names, range(len(d.names)))))
    names = sorted(list(names))
    respx = np.zeros((len(names), len(ds)) + ds[0].px.shape[1:], dtype=ds[0].px.dtype)
    resy = -np.ones((len(names), ) + ds[0].y.shape[1:], dtype=ds[0].y.dtype)
    print("merging poselets:", respx.shape)
    for i, name in enumerate(names):
        for di in range(len(ds)):
            if name in name_to_i[di]:
                respx[i, di] = ds[di].px[name_to_i[di][name]]
                assert(resy[i][0] == -1 or np.all(resy[i] == ds[di].y[name_to_i[di][name]]))
                resy[i] = ds[di].y[name_to_i[di][name]]
    assert(np.all(resy != -1))
    return Dataset(None, resy, names, px=respx.reshape((len(respx), -1)))


def merge_poselets(n_poselets, precalc_fc_pattern):
    d_train = list()
    d_test = list()
    ds = EmptyDataset()
    for i in range(0, n_poselets):
        d = huge_load(precalc_fc_pattern % i)
        d_train.append(d.train)
        d_test.append(d.test)
    ds.train = merge_poselets_ds(d_train)
    ds.test = merge_poselets_ds(d_test)
    return ds         


def mean_edge(figure):
    s = 0
    n = 0
    for v in figure:              
        if v[0] != -1:          
            s += dist(v[:2], v[2:4])
            n+=1
    if n == 0:
        return -1
    return s / n

def get_head_roi(figure):
    head_pts = np.reshape(figure, (-1, 2))
    head_pts = head_pts[-9:]
    head_pts = head_pts[head_pts[:, 0] >= 0]
    if len(head_pts) == 0:
        return (1e9, -1, 1e9, -1)
    center = head_pts.mean(axis=0)
    shape = np.array((mean_edge(figure) * head_size_k,) * 2)
    return center[0] - shape[0] / 2, center[0] + shape[0] / 2, center[1] - shape[1] / 2, center[1] + shape[1] / 2

def get_rois(ann):            
    rois = np.array([[1e9, -1, 1e9, -1]] * len(ann)) #t:b, l:r
    for n in range(len(ann)):
        rois[n] = get_head_roi(ann[n])
        for i in range(len(ann[n])):
            if ann[n][i][0] < 0:
                continue
            r = np.zeros(2)
            c = np.zeros(2)
            r[0], c[0], r[1], c[1] = ann[n][i]
            rois[n][0] = min(rois[n][0], r[0], r[1])
            rois[n][1] = max(rois[n][1], r[0], r[1])
            rois[n][2] = min(rois[n][2], c[0], c[1]) 
            rois[n][3] = max(rois[n][3], c[0], c[1])
    return rois


def int_square(pos1, shape1, pos2, shape2):
    c1 = max(pos1[0], pos2[0]), max(pos1[1], pos2[1])
    c2 = min(pos1[0] + shape1[0], pos2[0] + shape2[0]), \
         min(pos1[1] + shape1[1], pos2[1] + shape2[1]) 
    return (c2[0] - c1[0]) * (c2[1] - c1[1])


def isec_norm(roi1, roi2):
    isec1 = max(roi1[0], roi2[0]), max(roi1[2], roi2[2])
    isec2 = max(min(roi1[1], roi2[1]), isec1[0]), max(min(roi1[3], roi2[3]), isec1[1])
    un1 = min(roi1[0], roi2[0]), min(roi1[2], roi2[2])
    un2 = max(roi1[1], roi2[1]), max(roi1[3], roi2[3])
    return np.prod(np.subtract(isec2, isec1)) / np.prod(np.subtract(un2, un1))


def select_figure(figures, label_xywh):
    if len(figures) == 0:
        return None
    label_roi = [label_xywh[1], label_xywh[1] + label_xywh[3]]
    label_roi += [label_xywh[0], label_xywh[0] + label_xywh[2]]
    maxi = 0
    maxv = 0
    rois = get_rois(figures)
    for i, roi in enumerate(rois):
        v = isec_norm(roi, label_roi)
        #print(i, roi, v)
        if v > maxv:
            maxv = v
            maxi = i
    #print("maxvv", maxv)
    return figures[maxi]
    

def load_data_atr(datadir, shape, n_attr, poselet_id=None, is_test=False):
    f = open(path.join(datadir, 'labels.txt'), 'r')
    rows = f.readlines()
    f.close()
    
    #f = open(path.join(datadir, 'annotation.json'), 'r')
    #joints_dict = json.load(f)
    #f.close()
    joints_dict = np.load(path.join(datadir, 'annotation.npz'))
    #stats = calc_stats(joints_dict)
    n = len(rows)
    x, y, names = list(), list(), list()
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
            subimg = cv2.resize(subrect(img, roi), shape)
            joints = select_figure(joints_dict[fields[0]], roi)
            x.append([subimg] + extract_poselets(img, joints, shape))
            #print(fields[0])
            #batch_show(np.array([x[-1]], dtype=np.uint8))
        if x[-1] is None:
            x.pop()
            print('pop %s' % fields[0])
        else:
            if poselet_id is not None:
                x[-1] = x[-1][poselet_id:poselet_id + 1]
                if np.all(x[-1][0] == 0):
                    x.pop()
                    print('pop_%d %s' % (poselet_id, fields[0]))
                    continue
            y.append(list(map(float, fields[5:])))
            names.append(fields[0])
            if not is_test:
                x.append(symm_vert(x[-1]))
                y.append(y[-1])
                names.append(fields[0] + ".s")
        
    y = (np.array(y, dtype=img.dtype) + 1) * 0.5
    from sys import getsizeof
    print(getsizeof(x))
    xy = Dataset(np.array(x, dtype=x[0][0].dtype), y, names)
    print(getsizeof(xy.x), xy.x.shape, xy.x.dtype)
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


def cvt(img):
    if img.dtype == np.uint8:
        return img.astype(img.dtype) / 255


class BatchGenerator:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.order = np.arange(len(self.data))
        self.offset = len(self.data)
    
    def next_batch(self, batch_size):
        assert(batch_size <= len(self.data))
        if self.offset + batch_size <= len(self.data):
            self.offset += batch_size
            return  self.data[self.order[self.offset - batch_size:self.offset]], \
                    self.labels[self.order[self.offset - batch_size:self.offset]]
        p = self.order[self.offset:]
        self.order = np.random.permutation(len(self.data))
        self.offset += batch_size - len(self.data)
        p = np.hstack((p, self.order[:self.offset]))
        return self.data[p], self.labels[p]
