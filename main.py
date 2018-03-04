import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms 

from skimage import io, transform
from scipy.io import loadmat
from scipy.misc import imresize, imsave

import numpy as np

import math
from PIL import Image


from models import ResNet34RefineNet1
from dataset import load_dataset_ADE20K

use_cuda = True


def setting(args) :
	data = args.data

	size = args.imgSize
	batch_size = args.batch
	lr = args.lr
	epoch = args.epoch
	
	if 'ADE20K' in data :
		dataset = load_dataset_ADE20K(args,img_dim=size) 
	
	# Data loader
	data_loader = torch.utils.data.DataLoader(dataset=dataset,
    	                                      batch_size=batch_size, 
        	                                  shuffle=True)

	# Model :
	img_dim = size
	img_depth = 3
	#img_depths=[int(img_dim/8),int(img_dim/4),int(img_dim/2),int(img_dim)]
	conv_dim = args.conv_dim
	global use_cuda
	batch_size = 8
	semantic_labels_nbr = 150
	refinenet = ResNet34RefineNet1(img_dim_in=img_dim, img_depth_in=img_depth,conv_dim=conv_dim,use_cuda=use_cuda,semantic_labels_nbr=semantic_labels_nbr,use_batch_norm=args.use_batch_norm)
	frompath = True
	print(refinenet)
		
	# LOADING :
	path = 'SceneParsing--img{}-lr{}-conv{}'.format(img_dim,lr,conv_dim)
	if args.use_batch_norm :
		path += '-batch_norm'
		
	if not os.path.exists( './data/{}/'.format(path) ) :
		os.mkdir('./data/{}/'.format(path))
	if not os.path.exists( './data/{}/reconst_images/'.format(path) ) :
			os.mkdir('./data/{}/reconst_images/'.format(path))
	
	
	SAVE_PATH = './data/{}'.format(path) 

	if frompath :
		try :
			refinenet.load_state_dict( torch.load( os.path.join(SAVE_PATH,'weights')) )
			print('NET LOADING : OK.')
		except Exception as e :
			print('EXCEPTION : NET LOADING : {}'.format(e) )

	# OPTIMIZER :
	optimizer = torch.optim.Adam( refinenet.parameters(), lr=lr)
	
	if args.train :
		train_model(refinenet,data_loader, optimizer, SAVE_PATH,path,args,nbr_epoch=args.epoch,batch_size=args.batch,offset=args.offset)
	

def visualize_reconst_label(reconst,semantic_labels_nbr=150) :
	batch_size = reconst.size()[0]
	nbr_labels = reconst.size()[1]
	imgs = []
	for i in range(batch_size) :
		img =  reconst[i].float()
		img = (img-img.mean())/img.std()
		labels = img * 255.0
		imgs.append(labels.unsqueeze(0) )
	imgs = torch.cat(imgs, dim=0)
	return imgs

def visualize_reconst(reconst,semantic_labels_nbr=150) :
	batch_size = reconst.size()[0]
	nbr_labels = reconst.size()[1]
	dim = reconst.size()[2]
	imgs = []
	for i in range(batch_size) :
		img =  reconst[i]
		img = img.max(0)[1].float().view((1,dim,dim))
		img = (img-img.mean())/img.std()
		labels = img * 255.0
		imgs.append(labels.unsqueeze(0) )
	imgs = torch.cat(imgs, dim=2)
	return imgs

def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_set = set( labelmap.flatten() )
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),dtype=np.uint8)
    for label in labelmap_set:
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * np.tile(colors[label],(labelmap.shape[0], labelmap.shape[1], 1))
    
    return labelmap_rgb

def visualize(batch_data, pred, args,path,epoch=0):
    colors = loadmat('./color150.mat')['colors']
    imgs = batch_data['image']
    segs = batch_data['label']
    infos = batch_data['info']

    for j in range(len(infos)):
        img = imgs[j].clone()
        for t,m,s in zip(img, [0.229, 0.224, 0.225], [0.485, 0.456, 0.406]) :
        	t.mul_(m).add_(s)
        img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        img = imresize(img, (args.imgSize, args.imgSize),interp='bilinear')

        # segmentation
        lab = segs[j].numpy()
        lab_color = colorEncode(lab, colors)
        lab_color = imresize(lab_color, (args.imgSize, args.imgSize),interp='nearest')

        # prediction
        pred_ = np.argmax(pred.data.cpu()[j].numpy(), axis=0)
        pred_color = colorEncode(pred_, colors)
        pred_color = imresize(pred_color, (args.imgSize, args.imgSize),interp='nearest')

        # aggregate images and save
        im_vis = np.concatenate((img, lab_color, pred_color),axis=1).astype(np.uint8)
        imsave(os.path.join( path,'{}-{}'.format(epoch,infos[j].replace('/', '_').replace('.jpg', '.png')) ), im_vis)


def train_model(refinenet,data_loader, optimizer, SAVE_PATH,path,args,nbr_epoch=100,batch_size=32, offset=0, stacking=False) :
	global use_cuda
	
	img_depth=refinenet.img_depth
	pred_depth = refinenet.semantic_labels_nbr
	pred_dim = args.segSize
	img_dim = refinenet.img_dim_in

	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Debug :
	sample = next(data_iter)
	fixed_sample = sample
	fixed_x = sample['image']
	fixed_x = fixed_x.view( (-1, img_depth, img_dim, img_dim) )
	fixed_seg = sample['label'].view( (-1, 1, pred_dim, pred_dim) )
	fixed_seg_norm = visualize_reconst_label(fixed_seg)
	
	torchvision.utils.save_image(fixed_x.cpu(), './data/{}/real_images.png'.format(path))
	torchvision.utils.save_image(fixed_seg_norm, './data/{}/real_seg.png'.format(path))
	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim)).float()
	if use_cuda :
		fixed_x = fixed_x.cuda()

	best_loss = None
	best_model_wts = refinenet.state_dict()
	
	for epoch in range(nbr_epoch):	
		epoch_loss = 0.0
		
		for i, sample in enumerate(data_loader):
			images = sample['image'].float()
			labels = sample['label']
			
			# Save the reconstructed images
			if i % 100 == 0 :
				reconst_images_or = refinenet(fixed_x)
				reconst_images = reconst_images_or.cpu().data
				reconst_images = visualize_reconst(reconst_images)
				reconst_images = reconst_images.view(-1, 1, pred_dim, pred_dim)
				orimg = fixed_seg_norm.view(-1, 1, pred_dim, pred_dim)
				ri = torch.cat( [orimg, reconst_images], dim=2)
				torchvision.utils.save_image(ri,'./data/{}/reconst_images/{}.png'.format(path,(epoch+offset+1) ) )
				visualize(fixed_sample, reconst_images_or,args,path=SAVE_PATH,epoch=epoch+offset+1)
				model_wts = refinenet.state_dict()
				torch.save( model_wts, os.path.join(SAVE_PATH,'temp.weights') )
				print('Model saved at : {}'.format(os.path.join(SAVE_PATH,'temp.weights')) )

			#images = Variable( (images.view(-1, img_depth,img_dim, img_dim) ), volatile=False )#.float()
			images = Variable( images, volatile=False )#.float()
			labels = Variable( labels, volatile=False )#.float()
			#labels = Variable( (labels.view(-1, pred_depth,pred_dim, pred_dim) ) )#.float()
			
			if use_cuda :
				images = images.cuda() 
				labels = labels.cuda() 

			pred = refinenet(images)
			
			# Compute :
			#reconstruction loss :
			reconst_loss = nn.NLLLoss2d(ignore_index=-1)( pred, labels)
			
			# TOTAL LOSS :
			total_loss = reconst_loss
			
			# Backprop + Optimize :
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()

			epoch_loss += total_loss.cpu().data[0]

			if i % 10 == 0:
			    print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
			           "Reconst Loss: %.4f " 
			           %(epoch+1, nbr_epoch, i+1, iter_per_epoch, total_loss.data[0], 
			             reconst_loss.data[0]) )
			    if best_loss is not None :
			    	print("Epoch Loss : %.4f / Best : %.4f".format(epoch_loss, best_loss))


		if best_loss is None :
			#first validation : let us set the initialization but not save it :
			best_loss = epoch_loss
			best_model_wts = refinenet.state_dict()
			# save best model weights :
			torch.save( best_model_wts, os.path.join(SAVE_PATH,'weights') )
			print('Model saved at : {}'.format(os.path.join(SAVE_PATH,'weights')) )
		elif epoch_loss < best_loss:
			best_loss = epoch_loss
			best_model_wts = refinenet.state_dict()
			# save best model weights :
			torch.save( best_model_wts, os.path.join(SAVE_PATH,'weights') )
			print('Model saved at : {}'.format(os.path.join(SAVE_PATH,'weights')) )





if __name__ == '__main__' :
	import argparse
	parser = argparse.ArgumentParser(description='RefineNet')
	parser.add_argument('--train',action='store_true',default=False)
	parser.add_argument('--evaluate',action='store_true',default=False)
	parser.add_argument('--offset', type=int, default=0)
	parser.add_argument('--batch', type=int, default=32)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--conv_dim', type=int, default=64)
	parser.add_argument('--data', type=str, default='ADE20K')
	parser.add_argument('--use_batch_norm', action='store_true', default=False)
	parser.add_argument('--list_train', type=str, default='./SceneParsing/ADE20K_object150_train.txt')
	parser.add_argument('--list_val', type=str, default='./SceneParsing/ADE20K_object150_val.txt')
	parser.add_argument('--root_img', type=str, default='./SceneParsing/images')
	parser.add_argument('--root_seg', type=str, default='./SceneParsing/annotations')
	parser.add_argument('--imgSize', default=384, type=int,help='input image size')
	parser.add_argument('--segSize', default=96, type=int,help='output image size')

	args = parser.parse_args()
	print(args)

	setting(args)