import sys
import os
import struct
import numpy as np
from scipy.misc import *
import operator as op
from libs.mnist_util import *
from numpy.random import *
import Image as image

mnist_path = 'data/mnist'
n_syn_imgs = 15000
dest_path = 'data/mdist'



itemset = sorted([x for x in read(dataset = "training", path = mnist_path)]+[x for x in read(dataset = "testing", path = mnist_path)], key=op.itemgetter(0))

"""question types
     1) colors: no same target number
     2) objects (spatial relation): 
     3) numbers:
     4) location:
"""

numPools= []
for i in range(9):
	count = 0
	for j in range(len(itemset)):
		if itemset[j][0] != i:
			break
		count+=1
	numPools.append(itemset[:count])
	itemset = itemset[count:]
numPools.append(itemset)

colorNames = ['red', 'green', 'blue', 'yellow', 'white']
colorMap = {'red':[0x92,0x10,0x10], 'green':[0x10,0xA6,0x51], 'blue':[0x10,0x62,0xC0], 'yellow':[0xFF,0xF1,0x4E], 'white':[0xE0,0xE0,0xE0]}
for k in colorMap.keys():
	colorMap[k] = np.array(colorMap[k])

def generateSimpleColorQuestionInfo(minnum=3, maxnum=6):
	numbers = [i for i in range(10)]
	
	shuffle(numbers)
	
	maxnum = np.argmax(multinomial(1, [1./(maxnum-minnum+1)]*(maxnum-minnum+1)), axis=0)+minnum
	colors = np.argmax(multinomial(1, [1./len(colorNames)]*len(colorNames), maxnum), axis=1)
	
	res = []
	posset = set()
	for i in range(maxnum):
		res.append((numbers[i], colorNames[colors[i]], colorMap[colorNames[colors[i]]]))
	
	return res

scales = []
def renderImage(imgPath, questionInfoList, colorParam=(0, 10), scaleParam=(0.5, 1.5), num_dist=17):
	w, h = 100, 100
	sampleSize = (28, 28)
	newImg = np.zeros((w, h, 4), dtype=np.uint8)
	background = np.zeros((w, h, 4), dtype=np.uint8)
	distSize = (5, 5)
	#newImg[:]=255

	# put distractors
	for n in range(num_dist):
		sampleNum = int(uniform()*10)
		sampleIdx = int(uniform()*len(numPools[sampleNum]))

		color = colorMap[colorNames[int(uniform()*5)]]

		colorNoise = normal(colorParam[0], colorParam[1])
		color = color +colorNoise
		color[color<0] = 0
		color[color>255] = 255

		distPos = (int(uniform()*(sampleSize[0]-distSize[0])), int(uniform()*(sampleSize[1]-distSize[1])))

		distractor = numPools[sampleNum][sampleIdx][1][distPos[0]:distPos[0]+distSize[0], distPos[1]:distPos[1]+distSize[1]]
		
		scale = float(uniform(scaleParam[0], scaleParam[1]))
		size = (int(distSize[0]*scale), int(distSize[1]*scale))

		pos = np.round(np.array([uniform(0, w-size[0]), uniform(0, h-size[1])]))
		pos = pos.astype(np.uint8)

		distractor = imresize(distractor, size)
		background[pos[0]:pos[0]+size[0], pos[1]:pos[1]+size[1], 0:3] += (distractor[:, :, None]*color[None, None, :]/255.).astype('uint8')
	background[..., 3] = 255

	# put numbers
	for info in questionInfoList:
		sampleIdx = int(uniform()*len(numPools[info[0]]))

		#print mnistSample
		colorNoise = normal(colorParam[0], colorParam[1])
		color = info[2]+colorNoise
		color[color<0] = 0
		color[color>255] = 255

		while True:
			scale = float(uniform(scaleParam[0], scaleParam[1]))
			size = (int(sampleSize[0]*scale), int(sampleSize[1]*scale))
		
			pos = np.round(np.array([uniform(0, w-size[0]), uniform(0, h-size[1])]))
			pos[pos<0] = 0
			pos[pos>100-size[0]] = 100-size[0]
			pos = pos.astype(np.uint8)
			
			found = True
			for i in range(size[0]):
				for j in range(size[1]):
					if (newImg[pos[0]+i, pos[1]+j, :] != 0).any():
						found = False
						break
				if not found:
					break
			if found:
				break
		
		scales.append(scale)
		mnistSample = imresize(numPools[info[0]][sampleIdx][1], scale)

		for i in range(size[0]):
			for j in range(size[1]):
				if mnistSample[i, j] != 0:
					newImg[pos[0]+i, pos[1]+j, 0:3] = color * (mnistSample[i, j])/255.
					newImg[pos[0]+i, pos[1]+j, 3] = mnistSample[i, j]

	newImg[newImg > 255] = 255
	newImg[newImg < 0] = 0
	newImg = newImg.astype(np.uint8)

	img_ = image.fromarray(newImg)
	img = image.fromarray(background)
	img.paste(img_, None, img_)
	img = img.convert('RGB')
	
	img.save(imgPath)

labels = []
if not os.path.exists(dest_path):
	os.makedirs(dest_path)
if not os.path.exists(dest_path+'/imgs'):
	os.makedirs(dest_path+'/imgs')
for i in range(n_syn_imgs):
	print '\r', i,
	sys.stdout.flush()
	qInfo = generateSimpleColorQuestionInfo(minnum=5, maxnum=9)
	renderImage(dest_path + '/imgs/%05d.png'%i, qInfo, scaleParam=(0.5, 3.0))
	for item in qInfo:
		labels.append('\t'.join(['%05d'%i]+[str(x) for x in item[:-1]]))

labelOut = open(dest_path+'/labels.txt', 'wt')
labelOut.write('\n'.join(labels))
labelOut.close()
scaleOut = open(dest_path+'/scales.txt', 'wt')
scaleOut.write('\n'.join([str(x) for x in scales]))
scaleOut.close()
print 
