import os
import random
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms
from scipy.misc import imread, imresize

def makeColorLabelDicts( label_csv) :
    nbrLabel = len(label_csv['Name'])
    c2l = dict()
    l2c = dict()
    for i in range(nbrLabel) :
        ri,gi,bi,labeli = label_csv['R'][i], label_csv['G'][i], label_csv['B'][i], label_csv['Name'][i]
        c2l[(ri,gi,bi)] = (i,labeli)
        l2c[i] = (ri,gi,bi)
    return c2l, l2c

def encodeColor2Label(seg, c2il) :
    h,w,_ = seg.shape
    sout = np.zeros( (h,w) ).astype('int')
    seg = seg.astype('int')
    '''
    for i in range(h) :
        for j in range(w) :
            k = tuple(seg[i,j])
            d = c2il[k]
            sout[i,j] = d[0]
    '''
    sout = labelIdxEncode( seg, c2il)

    return sout 

def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_set = set( labelmap.flatten() )
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),dtype=np.uint8)
    for label in labelmap_set:
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * np.tile(colors[label][0],(labelmap.shape[0], labelmap.shape[1], 1))
    
    return labelmap_rgb

def labelIdxEncode(labels, colormap):
    colormap_set = set( tuple( k ) for k in colormap.keys() )
    colormap_labelidx = np.zeros((labels.shape[0], labels.shape[1]),dtype=np.uint8)
    
    for color in colormap_set:
        if color == (0,0,0):
            continue
        off0 = (labels[:,:,0] == color[0])
        off1 = (labels[:,:,1] == color[1])
        off2 = (labels[:,:,2] == color[2])
        off = off0 * off1 * off2
        #print('off : ',off.shape)
        lblidx = colormap[color][0]
        tl = np.tile( lblidx, (labels.shape[0], labels.shape[1]) )
        #print('tl :', tl.shape)
        add = off[:,:] * tl
        #print('add :',add.shape)
        add = np.array(add, dtype=np.uint8)
        colormap_labelidx += add
    
    return colormap_labelidx

class Dataset(torch.utils.data.Dataset):
    def __init__(self, txt, opt, max_sample=-1, is_train=1,data='ADE20K'):
        self.root_img = opt['root_img']
        self.root_seg = opt['root_seg']
        self.imgSize = opt['imgSize']
        self.segSize = opt['segSize']
        self.is_train = is_train
        self.data = data
        
        if self.data == 'CamVid' :
            self.label_csv = pd.read_csv(os.path.join(self.root_img,'../labeled.csv'), sep=' ')
            self.color2ilabel, self.ilabel2color =  makeColorLabelDicts( self.label_csv)

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.list_sample = [x.rstrip() for x in open(txt, 'r')]

        if self.is_train:
            random.shuffle(self.list_sample)
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def _scale_and_crop(self, img, seg, cropSize, is_train):
        h, w = img.shape[0], img.shape[1]

        if is_train:
            # random scale
            scale = random.random() + 0.5     # 0.5-1.5
            scale = max(scale, 1. * cropSize / (min(h, w) - 1))
        else:
            # scale to crop size
            scale = 1. * cropSize / (min(h, w) - 1)

        img_scale = imresize(img, scale, interp='bilinear')
        seg_scale = imresize(seg, scale, interp='nearest')

        h_s, w_s = img_scale.shape[0], img_scale.shape[1]
        if is_train:
            # random crop
            x1 = random.randint(0, w_s - cropSize)
            y1 = random.randint(0, h_s - cropSize)
        else:
            # center crop
            x1 = (w_s - cropSize) // 2
            y1 = (h_s - cropSize) // 2

        img_crop = img_scale[y1: y1 + cropSize, x1: x1 + cropSize, :]
        seg_crop = seg_scale[y1: y1 + cropSize, x1: x1 + cropSize]
        return img_crop, seg_crop

    def _flip(self, img, seg):
        img_flip = img[:, ::-1, :]
        seg_flip = seg[:, ::-1]
        return img_flip, seg_flip

    def __getitem__(self, index):
        img_basename = img_basename_annot = self.list_sample[index]
        if 'CamVid' in self.data  :
            basename, ext = img_basename.split('.')
            img_basename_annot = basename + '_L.' + ext
        path_img = os.path.join(self.root_img, img_basename)
        path_seg = os.path.join(self.root_seg,
                                img_basename_annot.replace('.jpg', '.png'))

        assert os.path.exists(path_img), '[{}] does not exist'.format(path_img)
        assert os.path.exists(path_seg), '[{}] does not exist'.format(path_seg)

        # load image and label
        try:
            img = imread(path_img, mode='RGB')
            seg = imread(path_seg)

            assert(img.ndim == 3)
            
            if 'CamVid' in self.data :
                seg = encodeColor2Label(seg, self.color2ilabel)
            else :
                assert(seg.ndim == 2)

            assert(img.shape[0] == seg.shape[0])
            assert(img.shape[1] == seg.shape[1])

            # random scale, crop, flip
            if self.imgSize > 0:
                img, seg = self._scale_and_crop(img, seg,
                                                self.imgSize, self.is_train)
                if random.choice([-1, 1]) > 0:
                    img, seg = self._flip(img, seg)

            # image to float
            img = img.astype(np.float32) / 255.
            img = img.transpose((2, 0, 1))

            if self.segSize > 0:
                seg = imresize(seg, (self.segSize, self.segSize),
                               interp='nearest')

            # label to int from -1 to 149
            seg = seg.astype(np.int) - 1

            # to torch tensor
            image = torch.from_numpy(img)
            segmentation = torch.from_numpy(seg)
        except Exception as e:
            print('Failed loading image/segmentation [{}]: {}'.format(path_img, e))
            # dummy data
            image = torch.zeros(3, self.imgSize, self.imgSize)
            segmentation = -1 * torch.ones(self.segSize, self.segSize).long()
            return image, segmentation, img_basename

        # substracted by mean and divided by std
        image = self.img_transform(image)

        output = dict()
        output['image'] = image
        output['label'] = segmentation
        output['info'] = img_basename
        
        return output 

    def __len__(self):
        return len(self.list_sample)


def load_dataset_ADE20K(args,img_dim=512) :
    dataset_train = Dataset(args['list_train'], args, is_train=1)
    
    return dataset_train

def load_dataset(args,img_dim=384,data='CamVid') :
    dataset_train = Dataset(args['list_train'], args, is_train=1,data=data)
    
    return dataset_train