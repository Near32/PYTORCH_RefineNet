import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms 

from skimage import io, transform
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
	semantic_labels_nbr = 1
	#refinenet = ResNet34RefineNet1(img_dim_in=img_dim, img_depth_in=img_depth, img_depths=img_depths,conv_dim=conv_dim,use_cuda=use_cuda,semantic_labels_nbr=semantic_labels_nbr)
	refinenet = ResNet34RefineNet1(img_dim_in=img_dim, img_depth_in=img_depth,conv_dim=conv_dim,use_cuda=use_cuda,semantic_labels_nbr=semantic_labels_nbr)
	frompath = True
	print(refinenet)
		
	# LOADING :
	path = 'SceneParsing--img{}-lr{}-conv{}'.format(img_dim,lr,conv_dim)
	
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
	



def train_model(refinenet,data_loader, optimizer, SAVE_PATH,path,args,nbr_epoch=100,batch_size=32, offset=0, stacking=False) :
	global use_cuda
	
	img_depth=refinenet.img_depth
	pred_depth = 1
	pred_dim = args.segSize
	img_dim = refinenet.img_dim_in

	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Debug :
	sample = next(data_iter)
	fixed_x = sample['image']
	
	fixed_x = fixed_x.view( (-1, img_depth, img_dim, img_dim) )
	torchvision.utils.save_image(fixed_x.cpu(), './data/{}/real_images.png'.format(path))
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
				'''
				reconst_images = refinenet(fixed_x)
				reconst_images = reconst_images.view(-1, img_depth, pred_dim, pred_dim).cpu().data
				#orimg = fixed_x.cpu().data.view(-1, img_depth, img_dim, img_dim)
				#ri = torch.cat( [orimg, reconst_images], dim=2)
				torchvision.utils.save_image(reconst_images,'./data/{}/reconst_images/{}.png'.format(path,(epoch+offset+1) ) )
				'''
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
	parser.add_argument('--list_train', type=str, default='./SceneParsing/ADE20K_object150_train.txt')
	parser.add_argument('--list_val', type=str, default='./SceneParsing/ADE20K_object150_val.txt')
	parser.add_argument('--root_img', type=str, default='./SceneParsing/images')
	parser.add_argument('--root_seg', type=str, default='./SceneParsing/annotations')
	parser.add_argument('--imgSize', default=512, type=int,help='input image size')
	parser.add_argument('--segSize', default=128, type=int,help='output image size')

	args = parser.parse_args()
	print(args)

	setting(args)