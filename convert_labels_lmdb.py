import sys 
import os
import numpy as np
import lmdb
import argparse
import caffe
import numpy

def main(args):
	caffe_root = '../'
	sys.path.insert(0, caffe_root + 'python')

	# read labels
	all_labels = []
	labelFile = open(args.labels_file) 
	env = lmdb.open(args.lmdb_file, map_size=int(1e12))
	for labelLine in labelFile:
		labelLine = labelLine.rstrip()
		labels = labelLine.split(',')
		all_labels.append(labels)
	
	# create lmdb
	key = 0
	with env.begin(write=True) as txn: 
	    for labels in all_labels:
		labels_list = numpy.array(labels)
		datum = caffe.proto.caffe_pb2.Datum()
		datum.channels = labels_list.shape[0]
		datum.height = 1
		datum.width =  1
		datum.data = labels_list.tostring()          # or .tobytes() if numpy < 1.9
		
		#print(datum.data)
		datum.label = 0
		key_str = '{:08}'.format(key)

		txn.put(key_str.encode('ascii'), datum.SerializeToString())
		key += 1


def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('labels_file', 
			type=str, 
			help='The label file name (.txt)')
	parser.add_argument('lmdb_file',
			type=str, 
			help='The lmdb file name')
	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
