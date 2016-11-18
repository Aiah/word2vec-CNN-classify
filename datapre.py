#encoding="utf-8"
#!/usr/bin/env python

import sys
import numpy as np
import jieba
jieba.load_userdict( "./../../Paper/Travel/step1_classfication/config/usedict" )
import re
import random
import unicodedata
import tensorflow as tf
import time

reload( sys )
sys.setdefaultencoding( "utf-8" )

"""
#  parameters the raw line
#  return the clear line that can segment to words
"""

def clearStr( line ) :
	line = line.replace( '=' , '' )
	line = line.replace( '~' , '' )
	line = line.replace( '-' , '' )
	line = line.replace( '#' , '' )
	line = line.replace( '*' , '' )
	return line.strip()

"""
# load the stopwords 
"""

def getstopwords( stopwordsPath ) :
	with open( stopwordsPath , 'r' ) as f :
		return { line.strip().decode( "utf-8" ) for line in f }

"""
# load the word2vec path
"""

def lodaModel( w2vModelPath ) :
	wordModel = dict()
	with open( w2vModelPath , 'r' ) as f :
		for line  in f :
			fields = line.strip().split( " " )
			word = fields[ 0 ]
			vector = map( float , fields[ 1 : ] )
			if len( vector ) != 200 :
				continue
			else :
				wordModel[ word.decode( "utf-8" ) ] = vector
	return wordModel


def wordTovector( segListFilter , wordModel , Suqence_length ) :
	vector = list()
	zeroVector = [ 0.0 ] * 200
	count = 0

	for word in segListFilter :
		if wordModel.has_key( word ) :
			count += 1
			if count > Suqence_length :
				break
			Vector = wordModel.get( word )
			vector.extend( Vector )
	if count > Suqence_length :
		return vector
	else :
		while count < Suqence_length :
			vector.extend( zeroVector )
			count += 1
	return vector

"""
#  parameters the file path
#  return the feature vector and his label 
"""

def load_data_label( dataPath , stopwordsPath , w2vModelPath , sequence_length ) :
	begin = time.clock()

	wordModel = lodaModel( w2vModelPath )
	stopwords = getstopwords( stopwordsPath )
	i = 0
	X = list()
	Y = list()
	with open( dataPath , 'r' ) as f :
		for line in f :
			i += 1
			if i > 3000 :
				break
			line = line.strip().split( "," , 1 )
			label = float( line[ 0 ] )
			content = clearStr( line[ 1 ] )
			seg_list = jieba.cut( content , cut_all = False )
			segListFilter = [ word for word in seg_list if word not in stopwords and not re.match( r'.*(\w)+.*' , unicodedata.normalize('NFKD', word ).encode('ascii','ignore') ) ]
			vector = wordTovector( segListFilter , wordModel , sequence_length )
			X.append( vector )
			Y.append( label )

	#change the label to softmax format
	Y = [ [ 1 - label , label ] for label in Y ]  

	#change the list format to matrix format
	train_x = np.array( X )
	train_y = np.array( Y )

	# we random the sequence of the train datasets
	np.random.seed( 10 )
	shuffle_indices = np.random.permutation( np.arange( train_y.shape[ 0 ] )  )

	x_shuffled = train_x[ shuffle_indices ]
	y_shuffled = train_y[ shuffle_indices ]

	end = time.clock()
	print "load the data and label end , {:f}".format( end - begin )
	return x_shuffled , y_shuffled


"""
#  @iterater time
#  @batch_size
#  return generater the data
"""

def batch_iter( data , batch_size , num_epochs , shuffle = True ) :
	"""
	# we mainly generate a batch of data for training
	"""
	begin = time.clock()

	data = np.array( data )
	data_size = data.shape[ 0 ]
	num_batchs_each_epoch = int( data_size / batch_size ) + 1

	for epochs in range( num_epochs ) :
		if shuffle :
			shuffle_indices = np.random.permutation( data_size )
			shuffle_data = data[ shuffle_indices ]
		else :
			shuffle_data = data
		for num_batch in range( num_batchs_each_epoch ) :
			start_index = num_batch * batch_size
			end_index = min( ( num_batch + 1 ) * batch_size , data_size )

			yield shuffle_data[ start_index : end_index ]
	end = time.clock()
	print "generate the batch of data for training end : {:f}".format( end - begin )

def main() :
	dataPath = "./../../Paper/Travel/step1_classfication/data/step1_data"
	stopwordsPath = "./stopwords.txt"
	w2vModelPath = "./../../Paper/Travel/result_min5_iter5.bin"

	sequence_length = 40
	X , Y = load_data_label( dataPath , stopwordsPath , w2vModelPath , sequence_length )
	printf(X , Y) 

if __name__ == '__main__' :
	main()
