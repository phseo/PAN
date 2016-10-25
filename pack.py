import numpy as np
import sys
from scipy import misc

path = 'data/mdist'

channel = 3
height = 100
width = 100
nImg = 15000
imgs = []
for i in range(nImg):
	print '\r', i,
	sys.stdout.flush()
	im = misc.imread(path+'/imgs/%05d.png'%i)#[..., :3]
	im = misc.imresize(im, (height,width))
	im = np.rollaxis(im, 2).reshape((1, channel, height, width))
	imgs.append(im)
print 

data = np.concatenate(imgs)


refs=[]
labels=[]
questions=[]
f = open(path+'/labels.txt')
for l in f:
	ts = l.strip().split()

	refs.append(int(ts[0]))
	question = int(ts[1])
#	if question == 0:
#		question = 10
	questions.append(question)
	labels.append(ts[2])

scales = []
f = open(path+'/scales.txt')
for l in f:
	scale = float(l.strip())
	scales.append(scale)

keys = set(labels)
colorDict = {}
i=0
for key in keys:
	colorDict[key] = i
	i += 1

for key in keys:
	print colorDict[key], key


refs = np.array(refs)
labels = np.array([colorDict[k] for k in labels])
questions = np.array(questions)
scales = np.array(scales)

np.savez(path+'/mdist.npz', labels=labels, questions=questions, data=data, refs=refs, scales=scales)

