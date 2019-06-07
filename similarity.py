import sys
import gensim 
import numpy as np
from random import sample

def main():

	sm = SimModel(source='glove') # or 'word2vec' , adding etc.

	top10 = sm.model.most_similar(positive=['hello'], topn=10)

	print(np.asarray(top10))

	embeddingMatrix = sm.getEmbeddingMatrix(['hello','world'])

	print(np.asarray(embeddingMatrix))

	import pdb; pdb.set_trace()


class SimModel:
	def __init__(self, filename=None, binary=False, size=300, source='glove' ):

		self.size = size
		if not filename:
			if source.lower() == 'glove':
				file  = zipfile.ZipFile('./data/glove.6B.300d.txt.zip').open('glove.6B.300d.txt','r')
				binary = False
			elif source.lower() == 'word2vec':
				file = './data/GoogleNews-vectors-negative300.bin'
				binary = True
			else:
				assert False, 'need a source for SimModel'
		else:
			pass # open the passed file

		sys.stderr.write("Loading Similarity Model: {} ...\n".format(source))
		self.model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=binary)  

	# return a tuple of the closest word and it's vector
	def closest(self,word):
		try:
			closest_word = self.closestWord(word)
			return (closest_word, self.model.wv[closest_word] )
		except KeyError:
			return (None,None)

	# return a tuple of the farthest word and it's vector
	def farthest(self,word):
		try:
			farthest_word = self.model.most_similar(positive=[str(word)], topn=100000)[-1][0]
			return ( farthest_word, self.model.wv[farthest_word] )
		except KeyError:
			return (None,None)
			
	def opposite(self,word):
		try:
			vector = self.wordVector(str(word))	
			opposite_vector = np.asarray(list(map(lambda x: -x, vector)))
			closest_opposite_word = self.model.most_similar(positive=[opposite_vector], topn=1)[0]
			return (closest_opposite_word, self.wordVector(closest_opposite_word) )
		except KeyError:
			return (None,None)
			
	def wordVector(self, word):
		try:
			return self.model.wv[str(word)]
		except:
			return None # npself.model.wv[str('unknown')]

	def closestWord(self,word):
		return self.model.most_similar(positive=[str(word)], topn=1)[0][0]

	def writeHistograms():
		import matplotlib.pyplot as plt
		from matplotlib.patches import Rectangle

		#create legend info
		colors  = ['red','blue']
		labels  = ['GloVe','Word2Vec']
		handles = [Rectangle((0,0),1,1,color=c,ec="k",alpha=.75) for c in colors]

		for i,mod in enumerate(labels):

			sm = SimModel(source=mod.lower())

			em = sample(list(sm.model.vectors),10000)
			em = [item for sublist in em for item in sublist]#flatten it

			# density : bool, optional
			# If False, the result will contain the number of samples in each bin. 
			# If True, the result is the value of the probability density function 
			# at the bin, normalized such that the integral over the range is 1. 
			# Note that the sum of the histogram values will not be equal to 1 
			# unless bins of unity width are chosen; 
			# it is not a probability mass function.

			n, bins, patches = plt.hist(em, bins=500, density=1, color=colors[i], alpha=.75)

			# if you want to show standard deviation
			# mu = np.mean(em)
			# sigma = np.std(em)
			# plt.axvline(mu-sigma, color='r', linestyle='dashed', linewidth=1)
			# plt.axvline(mu+sigma, color='r', linestyle='dashed', linewidth=1)
			# plt.axvline(mu-sigma*2, color='r', linestyle='dashed', linewidth=1)
			# plt.axvline(mu+sigma*2, color='r', linestyle='dashed', linewidth=1)

		plt.legend(handles, labels)
		plt.ylabel("Probability Density", fontsize=14)  
		plt.xlabel("Dimension Values", fontsize=14)
		plt.savefig('EmbeddingsHistogram.png')

	def getEmbeddingSize(self):
		if not self.size:
			self.size = len(self.wordVector('hello'))
		return self.size

	def getEmbeddingMatrix(self,vocabulary):
		embeddingMatrix = np.zeros((len(vocabulary), self.getEmbeddingSize()))
		for i,word in enumerate(vocabulary):
			embeddingMatrix[i] = self.wordVector(word)
		return embeddingMatrix

	def optimize(matrix,n,mode='std'):
		# standard deviations for each dimension
		if mode == 'std':
			dev = np.std(matrix,axis=0)
		elif mode == 'var':
			dev = np.var(matrix,axis=0)
		else:
			assert False

		srt = dev.argsort() # indexes of the highest at the end

		maxidx = srt[-n:][::-1] # slice n from the end and reverse
		minidx = srt[:n]		# slice n smallest from beginning

		retmax = [ y[maxidx] for y in matrix]
		retmin = [ y[minidx] for y in matrix]
		return np.array(retmin),np.array(retmax)

	def optimizePosition(matrix,n,minmax='max'):
		# standard deviations for each dimension
		dev = np.std(matrix,axis=0)

		srt = dev.argsort() # indexes of the highest at the end

		# set all columns identified by idx to 0
		if minmax == 'max':
			maxidx = srt[-n:][::-1] # slice n from the end and reverse
			matrix[...,tuple([maxidx])] = 0
		elif minmax == 'min':
			minidx = srt[:n]		# slice n smallest from beginning
			matrix[...,tuple([minidx])] = 0

		return matrix
		# retmax = [ y[maxidx] for y in matrix]
		# retmin = [ y[minidx] for y in matrix]
		# return np.array(retmin),np.array(retmax)

	def getEmbeddingMatrix(self,vocabulary):
		embeddingMatrix = np.zeros((len(vocabulary), self.getEmbeddingSize()))
		for i,word in enumerate(vocabulary):
			embeddingMatrix[i] = self.wordVector(word)
		return embeddingMatrix


	# def oppositeVector(self,word):
	# 	vector = self.wordVector(str(word))
	# 	return np.asarray(list(map(lambda x: -x, vector))) # TAC

	# def oppositeWord(self,word):
	# 	return self.model.most_similar(positive=[self.oppositeVector(word)], topn=1)[0][0]

	# def oppositeWordVector(self,word):
	# 	return self.wordVector(self.oppositeWord(word))

if __name__ == '__main__':
	main()
