import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

import torchvision
from torchvision import models
from torchvision.models.resnet import model_urls,BasicBlock

import numpy as np
from math import floor


class Distribution(object) :
	def sample(self) :
		raise NotImplementedError

	def log_prob(self,values) :
		raise NotImplementedError

class Bernoulli(Distribution) :
	def __init__(self, probs) :
		self.probs = probs

	def sample(self) :
		return torch.bernoulli(self.probs)

	def log_prob(self,values) :
		log_pmf = ( torch.stack( [1-self.probs, self.probs] ) ).log()
		dum = values.unsqueeze(0).long()
		return log_pmf.gather( 0, dum ).squeeze(0)
		
		#logits, value = broadcast_all(self.probs, values)
		#return -F.binary_cross_entropy_with_logits(logits, value, reduce=False)

		#return -F.binary_cross_entropy_with_logits(self.probs, values)

def conv( sin, sout,k,stride=2,pad=1,batchNorm=True,bias=True) :
	layers = []
	layers.append( nn.Conv2d( sin,sout, k, stride,pad,bias=bias) )
	if batchNorm :
		layers.append( nn.BatchNorm2d( sout) )
	return nn.Sequential( *layers )

def deconv( sin, sout,k,stride=2,pad=1,dilation=1,batchNorm=True,bias=True) :
	layers = []
	layers.append( nn.ConvTranspose2d( sin,sout, k, stride,pad,dilation=dilation,bias=bias) )
	if batchNorm :
		layers.append( nn.BatchNorm2d( sout) )
	return nn.Sequential( *layers )




class Decoder(nn.Module) :
	def __init__(self,net_depth=3, z_dim=32, img_dim=128, conv_dim=64,img_depth=3 ) :
		super(Decoder,self).__init__()
		
		self.z_dim = z_dim
		self.img_depth=img_depth
		self.img_dim = img_dim

		stride = 2 
		pad = 1
		
		self.fc = deconv( self.z_dim, 512, 32, stride=1, pad=0, batchNorm=False)
		
		self.dcs = []
		
		self.dcs.append( deconv( 512, 256, 4,stride=stride,pad=pad) )
		self.dcs.append( nn.LeakyReLU(0.05) )
		
		self.dcs.append( deconv( 256, 128, 4,stride=stride,pad=pad) )
		self.dcs.append( nn.LeakyReLU(0.05) )
		
		self.dcs.append( deconv( 128, 64, 4,stride=stride,pad=pad) )
		self.dcs.append( nn.LeakyReLU(0.05) )
		
		self.dcs.append( deconv( 64, 32, 4,stride=stride,pad=pad) )
		self.dcs.append( nn.LeakyReLU(0.05) )
		
		self.dcs = nn.Sequential( *self.dcs) 
		
		stride = 1
		pad = 0	
		self.dcout = deconv( 32, self.img_depth, 1, stride=stride, pad=pad, batchNorm=False)
		
	def decode(self, z) :
		z = z.view( z.size(0), z.size(1), 1, 1)
		out = F.leaky_relu( self.fc(z), 0.05)
		out = self.dcs(out)
		out = F.sigmoid( self.dcout(out))
		return out

	def forward(self,z) :
		return self.decode(z)


class Encoder(nn.Module) :
	def __init__(self,net_depth=3, img_dim=128, img_depth=3, conv_dim=64, z_dim=32 ) :
		super(Encoder,self).__init__()
		
		self.z_dim = z_dim
		self.img_depth=img_depth
		self.img_dim = img_dim

		stride = 2 
		pad = 1

		self.cvs = []
		
		self.cvs.append( conv( self.img_depth, 32, 4,stride=stride,pad=pad) )
		self.cvs.append( nn.LeakyReLU(0.05) )
		
		self.cvs.append( deconv(  32, 64, 4,stride=stride,pad=pad) )
		self.cvs.append( nn.LeakyReLU(0.05) )
		
		self.cvs.append( deconv( 64, 128, 4,stride=stride,pad=pad) )
		self.cvs.append( nn.LeakyReLU(0.05) )
		
		self.cvs.append( deconv( 128, 256, 4,stride=stride,pad=pad) )
		self.cvs.append( nn.LeakyReLU(0.05) )
		
		self.cvs.append( deconv( 256, 512, 4,stride=stride,pad=pad) )
		self.cvs.append( nn.LeakyReLU(0.05) )
		
		self.cvs = nn.Sequential( *self.cvs) 
		
		self.fc = deconv( 512, 1, 4, stride=1, pad=0, batchNorm=False)
		
		self.fc1 = nn.Linear( 2048, 1024)
		self.fc2 = nn.Linear( 1024, z_dim)
		
	def encode(self, x) :
		out = self.cvs(x)

		out = out.view( (-1, self.num_features(out) ) )
		print(out.size() )

		out = F.leaky_relu( self.fc(out), 0.05 )
		out = F.leaky_relu( self.fc1(out), 0.05 )
		out = self.fc2(out)
		
		return out

	def forward(self,x) :
		return self.encode(x)

	def num_features(self, x) :
		size = x.size()[1:]
		# all dim except the batch dim...
		num_features = 1
		for s in size :
			num_features *= s
		return num_features



class ResNetBlock(nn.Module) :
	def __init__(self, img_dim=128, img_depth=512, conv_dim=256,use_cuda=True ) :
		super(ResNetBlock,self).__init__()
		
		self.img_depth=img_depth
		self.img_dim = img_dim
		self.conv_dim = conv_dim
		self.use_cuda = use_cuda
		
		self.cvs = []
		
		stride = 1 
		pad = 1

		self.cvs.append( nn.LeakyReLU(0.05) )
		self.cvs.append( conv( self.img_depth, self.conv_dim, 3,stride=stride,pad=pad, batchNorm=False,bias=False) )
		
		self.cvs.append( nn.LeakyReLU(0.05) )
		self.cvs.append( conv(  self.conv_dim, self.conv_dim, 3,stride=stride,pad=pad, batchNorm=False,bias=False) )
		
		self.cvs = nn.Sequential( *self.cvs) 
		
		# downsampling :
		if self.img_depth != self.conv_dim :
			self.downsampling =	nn.Sequential(
							nn.Conv2d(self.img_depth, self.conv_dim, kernel_size=1, stride=stride, bias=False),
							nn.BatchNorm2d(self.conv_dim) 
						)
		else :
			self.downsampling = None

		if self.use_cuda :
			self = self.cuda()

	def forward(self, x) :
		out = self.cvs(x)

		# residual downsampling :
		if self.downsampling is not None :
			residual = self.downsampling(x)
		else :
			residual = x

		# SUM :
		output = out + residual

		return output



class MultiResFusionBlock(nn.Module) :
	def __init__(self, img_dim=[512,256,128,64], img_depth=256, use_cuda=True ) :
		super(MultiResFusionBlock,self).__init__()
		
		self.img_depth=img_depth
		self.use_cuda = use_cuda

		self.img_dim = img_dim
		self.nbr_paths = len(self.img_dim)
		
		self.min_dim = min(self.img_dim)
		self.max_dim = max(self.img_dim)
		
		self.paths = []
		for i in range(self.nbr_paths) :
			path = []
			
			# 3x3Conv :
			ind = self.img_depth
			outd = self.img_depth
			outdim = self.min_dim
			indim = self.img_dim[i]
			pad = 0
			stride = 1
			#k = outdim +2*pad -stride*(indim-1)
			#outdim = (indim-k+2*pad)/stride +1
			k = floor( indim-(outdim-1)*stride-2*pad )

			path.append( conv( ind, outd, k, stride=stride, pad=pad, batchNorm=False,bias=False) )


			# Upsampling :
			ind = self.img_depth
			outd = self.img_depth
			outdim = self.max_dim
			indim = self.min_dim
			pad = 0
			stride = 1
			k = floor( outdim + 2*pad - stride*(indim-1) )
			
			path.append( deconv( ind, outd, k, stride=stride, pad=pad, batchNorm=False,bias=False) )


			path = nn.Sequential( *path) 
			
			if self.use_cuda :
				path = path.cuda()

			self.paths.append(path)

		

	def __str__(self) :
		outstr = ''
		for i in range(len(self.paths) ) :
			outstr += str(self.paths[i])
			outstr += '\n\r'
		return outstr


	def forward(self, x) :
		assert(len(x) == self.nbr_paths)
		
		#for i in range(self.nbr_paths) :
		#	print(x[i].size())

		#for i in range(self.nbr_paths) :
		#	print(self.paths[i])


		out = []
		for i in range(self.nbr_paths) :
			out.append( self.paths[i](x[i]) )
			
		output = out[self.nbr_paths-1]
		for i in range(self.nbr_paths-1) :	
			output += out[i]

		return output


class ChainedResPoolBlock(nn.Module) :
	def __init__(self, img_dim=128, img_depth=512, semantic_labels_nbr=10, use_cuda=True) :
		super(ChainedResPoolBlock,self).__init__()
		
		self.img_depth=img_depth
		self.img_dim = img_dim
		self.use_cuda = use_cuda
		self.semantic_labels_nbr = semantic_labels_nbr
		
		
		self.relu1 = nn.LeakyReLU(0.05) 
		
		# MaxPool :
		stride = 1
		pad = 1
		self.pool1 = nn.MaxPool2d(kernel_size=5,stride=stride,padding=pad)
		
		# 3x3Conv :
		ind = self.img_depth
		outd = self.img_depth
		outdim = self.img_dim
		indim = (self.img_dim +2*pad - 5 + 1)/1 +1
		pad = 0
		stride = 1
		dilation=2
		#k = indim-(outdim-1)*stride-2*pad
		k = floor( outdim + 2*pad - stride*(indim-1) )
		#print('CHAINED RES : indim1 : {}'.format(indim) )
		#print('CHAINED RES : k1 : {}'.format(k) )
		#k=3
		self.cv1 =  deconv( ind, outd, k, stride=stride, pad=pad, dilation=dilation,batchNorm=False,bias=False)
					  
		# MaxPool :
		stride = 1
		pad = 1
		self.pool2 = nn.MaxPool2d(kernel_size=5,stride=stride,padding=pad)
		
		# 3x3Conv :
		ind = self.img_depth
		outd = self.img_depth
		outdim = self.img_dim
		indim = (indim + 2*pad - 5 + 1)/1 +1
		pad = 0
		stride = 1
		dilation = 2
		#k = indim-(outdim-1)*stride-2*pad
		k = floor( outdim + 2*pad - stride*(indim-1) )
		#print('CHAINED RES : indim2 : {}'.format(indim) )
		#print('CHAINED RES : k2 : {}'.format(k) )
		#k=3
		self.cv2 =  deconv( ind, outd, k, stride=stride, pad=pad, dilation=dilation, batchNorm=False,bias=False)
					
		# MaxPool :
		stride = 1
		pad = 1
		self.pool3 = nn.MaxPool2d(kernel_size=5,stride=stride,padding=pad)
		
		# 3x3Conv :
		ind = self.img_depth
		outd = self.img_depth
		outdim = self.img_dim
		indim = (indim + 2*pad - 5 + 1)/1 +1
		pad = 0
		stride = 1
		dilation = 2
		#k = indim-(outdim-1)*stride-2*pad
		k = floor( outdim + 2*pad - stride*(indim-1) )
		#print('CHAINED RES : indim3 : {}'.format(indim) )
		#print('CHAINED RES : k3 : {}'.format(k) )
		#k=3
		self.cv3 =  deconv( ind, outd, k, stride=stride, pad=pad, dilation=dilation, batchNorm=False,bias=False)

		#self.adaptation_cv = deconv( outd, self.semantic_labels_nbr, 1, stride=1,pad=0, batchNorm=False,bias=False)
					
		if self.use_cuda :
			self = self.cuda()

	def forward(self, x) :
		out = self.relu1(x)
		
		out_pool1 = self.pool1(out)
		out = self.cv1(out_pool1)

		outsum = x + out

		out_pool2 = self.pool2(out_pool1)
		out = self.cv2(out_pool2)

		outsum += out

		out_pool3 = self.pool3(out_pool2)
		out = self.cv3(out_pool3)

		outsum += out

		#outsum = self.adaptation_cv(outsum)

		return outsum


class ModelResNet(models.ResNet) :
	def __init__(self,outdim,**kwargs) :
		super(ModelResNet, self).__init__(BasicBlock, [2, 2, 2, 2], **kwargs)
		self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
		#self = models.resnet18(pretrained=True)
		# prevent learning on the resnet :
		#for param in model_conv.parameters() :
		#		param.requires_grad = False


class ModelResNet34(models.ResNet) :
	def __init__(self,**kwargs) :
		super(ModelResNet34, self).__init__(BasicBlock, [3, 4, 6, 3], **kwargs)
		self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
		# prevent learning on the resnet :
		#for param in model_conv.parameters() :
		#		param.requires_grad = False

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		self.x0 = x

		x = self.layer1(x)
		self.x1 = x

		x = self.layer2(x)
		self.x2 = x

		x = self.layer3(x)
		self.x3 = x

		x = self.layer4(x)
		self.x4 = x

		#x = self.avgpool(x)
		#x = x.view(x.size(0), -1)
		#x = self.fc(x)

		return x


class RefineNetBlock(nn.Module) :
	def __init__(self, img_dim_in=512, img_depths=[64,128,256,512], conv_dim=256, use_cuda=True, semantic_labels_nbr=10) :
		super(RefineNetBlock,self).__init__()
		self.img_dim_in = img_dim_in
		self.img_depths = img_depths
		self.conv_dim = conv_dim
		self.use_cuda = use_cuda

		self.semantic_labels_nbr = semantic_labels_nbr
		if self.semantic_labels_nbr is None :
			self.semantic_labels_nbr = 3

		self.nbr_paths = len(self.img_depths)
		self.RCUpaths = []

		self.img_dims = [ int(self.img_dim_in/4) ]
		for i in range(self.nbr_paths-1) :
			self.img_dims.append( int(self.img_dims[-1]/2) )

		for i in range(self.nbr_paths) :
			self.RCUpaths.append( ResNetBlock(img_dim=self.img_dims[i], img_depth=self.img_depths[i], conv_dim=self.conv_dim,use_cuda=self.use_cuda) )
		print('Creation of the RCUs : OK.')
		

		# Multi resolution Fusion Block :
		#print('INPUT MULTI RES : {}'.format(self.img_dims) )
		self.MultiResFusion = MultiResFusionBlock(img_dim=self.img_dims, img_depth=self.conv_dim,use_cuda=self.use_cuda )
		print('Creation of the Multi-Resolution Fusion Block : OK.')

		self.ChainedResPool = ChainedResPoolBlock( img_dim=self.img_dims[0], img_depth=self.conv_dim, semantic_labels_nbr=self.semantic_labels_nbr,use_cuda=self.use_cuda)
		print('Creation of the Chained Residual Pooling Block : OK.')

		self.finalRCU =  ResNetBlock(img_dim=self.img_dims[0], img_depth=self.conv_dim, conv_dim=self.semantic_labels_nbr,use_cuda=self.use_cuda) 
		
	def forward(self,x) :
		assert(len(x)==self.nbr_paths)

		out = []
		
		# RCUs :
		for i in range(len(x)) :
			#print('IN : RCU {} : {}'.format(i, x[i].size() ))
			out.append( self.RCUpaths[i](x[i]) )
			#print('OUT : RCU {} : {}'.format(i, out[i].size() ))

		# Multi Resolution Fusion :
		out = self.MultiResFusion(out)

		# Chained Residual Pooling :
		out = self.ChainedResPool(out)

		# Final Adaptation RCU :
		out = self.finalRCU(out)

		# log softmax :
		out = F.log_softmax(out)

		return out


class ResNet34RefineNet1(nn.Module) :
	def __init__(self, img_dim_in=512, img_depth_in=3, img_depths=[64,128,256,512], conv_dim=256,use_cuda=True, semantic_labels_nbr=10) :
		super(ResNet34RefineNet1,self).__init__()
		self.img_dim_in = img_dim_in
		self.nbr_paths = len(img_depths)
		self.img_depths = img_depths
		self.img_depth = img_depth_in
		self.conv_dim = conv_dim
		self.use_cuda = use_cuda
		self.semantic_labels_nbr = semantic_labels_nbr

		if self.use_cuda :
			self.resnet = ModelResNet34().cuda()
		else :
			self.resnet = ModelResNet34()
		
		self.refinenet = RefineNetBlock(img_dim_in=self.img_dim_in, img_depths=self.img_depths,conv_dim=self.conv_dim,use_cuda=self.cuda, semantic_labels_nbr=self.semantic_labels_nbr )

	def forward(self,x) :
		self.resnet_out = self.resnet(x)

		self.inputs = [self.resnet.x1,self.resnet.x2,self.resnet.x3,self.resnet.x4]

		#print('RESNET OUTPUTS : ')
		#for el in self.inputs :
		#	print(' size : {}'.format( el.size() ) )

		self.refinenet_out = self.refinenet(self.inputs)

		return self.refinenet_out
		

def test_refinenet() :
	import time

	img_dim = 384
	conv_dim = 64
	use_cuda=True
	batch_size = 2
	semantic_labels_nbr = 150
	refinenet = ResNet34RefineNet1(img_dim_in=img_dim,conv_dim=conv_dim,use_cuda=use_cuda,semantic_labels_nbr=semantic_labels_nbr)
	#print(refinenet)
	#print(refinenet.refinenet.MultiResFusion)

	inputs = Variable(torch.rand((batch_size,3,img_dim,img_dim))).cuda()
	t =time.time()
	outputs = refinenet(inputs)
	elt = time.time()-t
	print('ELT : {} sec.'.format(elt))
	print(outputs.size())

	elt = 0.0
	nbr = 100
	for i in range(nbr) :
		t =time.time()
		inputs = Variable(torch.rand((batch_size,3,img_dim,img_dim))).cuda()
		outputs = refinenet(inputs)
		elt += (time.time()-t)/nbr
	
	print('MEAN ELT : {} sec.'.format(elt))



class betaVAE(nn.Module) :
	def __init__(self, beta=1.0,net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		super(betaVAE,self).__init__()
		self.encoder = Encoder(z_dim=2*z_dim, img_depth=img_depth, img_dim=img_dim, conv_dim=conv_dim,net_depth=net_depth)
		self.decoder = Decoder(z_dim=z_dim, img_dim=img_dim, img_depth=img_depth, net_depth=net_depth)

		self.z_dim = z_dim
		self.img_dim=img_dim
		self.img_depth=img_depth
		
		self.beta = beta
		self.use_cuda = use_cuda

		if self.use_cuda :
			self = self.cuda()

	def reparameterize(self, mu,log_var) :
		eps = torch.randn( (mu.size()[0], mu.size()[1]) )
		veps = Variable( eps)
		#veps = Variable( eps, requires_grad=False)
		if self.use_cuda :
			veps = veps.cuda()
		z = mu + veps * torch.exp( log_var/2 )
		return z

	def forward(self,x) :
		h = self.encoder( x)
		mu, log_var = torch.chunk(h, 2, dim=1 )
		z = self.reparameterize( mu,log_var)
		out = self.decoder(z)

		return out, mu, log_var

class Rescale(object) :
	def __init__(self, output_size) :
		assert( isinstance(output_size, (int, tuple) ) )
		self.output_size = output_size

	def __call__(self, sample) :
		image = sample
		#image = np.array( sample )
		#h,w = image.shape[:2]

		new_h, new_w = self.output_size

		#img = transform.resize(image, (new_h, new_w) )
		img = image.resize( (new_h,new_w) ) 

		sample = np.reshape( img, (1, new_h, new_w) )

		return sample 


def test_mnist():
	import os
	import torchvision
	from torchvision import datasets, transforms
	dataset = datasets.MNIST(root='./data',
		                     train=True,
		                     transform=transforms.ToTensor(),
							 download=True) 

	# Data loader
	data_loader = torch.utils.data.DataLoader(dataset=dataset,
    	                                      batch_size=100, 
        	                                  shuffle=True)
	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Model :
	z_dim = 12
	img_dim = 28
	img_depth=1
	conv_dim = 32
	use_cuda = True#False
	net_depth = 2
	vae = VAE(net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	
	# Optim :
	lr = 1e-4
	optimizer = torch.optim.Adam( vae.parameters(), lr=lr)

	# Debug :
	# fixed inputs for debugging
	fixed_z = Variable(torch.randn(100, z_dim))
	if use_cuda :
		fixed_z = fixed_z.cuda()

	var_z0 = torch.zeros(100, z_dim)
	val = -0.5
	step = 1.0/10.0
	for i in range(10) :
		var_z0[i,0] = val
		var_z0[i+10,1] = val
		var_z0[i+20,2] = val
		var_z0[i+30,3] = val
		var_z0[i+40,4] = val
		var_z0[i+50,5] = val
		var_z0[i+60,6] = val
		var_z0[i+70,7] = val
		var_z0[i+80,8] = val
		var_z0[i+90,9] = val
		val += step

	var_z0 = Variable(var_z0)
	if use_cuda :
		var_z0 = var_z0.cuda()

	fixed_x, _ = next(data_iter)
	

	path = 'layers{}-z{}-conv{}-lr{}'.format(net_depth,z_dim,conv_dim,lr)
	if not os.path.exists( './data/{}/'.format(path) ) :
		os.mkdir('./data/{}/'.format(path))
	if not os.path.exists( './data/{}/gen_images/'.format(path) ) :
		os.mkdir('./data/{}/gen_images/'.format(path))

	
	torchvision.utils.save_image(fixed_x.cpu(), './data/{}/real_images.png'.format(path))
	
	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim))
	if use_cuda :
		fixed_x = fixed_x.cuda()

	
	for epoch in range(50):
	    # Save the reconstructed images
	    reconst_images, _, _ = vae(fixed_x)
	    reconst_images = reconst_images.view(-1, img_depth, img_dim, img_dim)
	    torchvision.utils.save_image(reconst_images.data.cpu(),'./data/{}/reconst_images_{}.png'.format(path,(epoch+1)) )

	    # Save generated variable images :
	    gen_images = vae.decoder(var_z0)
	    gen_images = gen_images.view(-1, img_depth, img_dim, img_dim)
	    torchvision.utils.save_image(gen_images.data.cpu(),'./data/{}/gen_images/{}.png'.format(path,(epoch+1)) )

	    for i, (images, _) in enumerate(data_loader):
	        
	        images = Variable( (images.view(-1,1,img_dim, img_dim) ) )
	        if use_cuda :
	        	images = images.cuda() 
	        
	        out, mu, log_var = vae(images)
	        
	        
	        # Compute reconstruction loss and kl divergence
	        # For kl_divergence, see Appendix B in the paper or http://yunjey47.tistory.com/43
	        

	        reconst_loss = F.binary_cross_entropy(out, images, size_average=False)
	        kl_divergence = torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1))
	        
	        # Backprop + Optimize
	        total_loss = reconst_loss + kl_divergence
	        optimizer.zero_grad()
	        total_loss.backward()
	        optimizer.step()
	        
	        if i % 100 == 0:
	            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
	                   "Reconst Loss: %.4f, KL Div: %.7f" 
	                   %(epoch+1, 50, i+1, iter_per_epoch, total_loss.data[0], 
	                     reconst_loss.data[0], kl_divergence.data[0]))
	    
	    

if __name__ == '__main__' :
	#test_mnist()
	test_refinenet()