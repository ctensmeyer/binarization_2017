#!/usr/bin/python

import os
import sys
import numpy as np
import caffe
import cv2
import scipy.ndimage as nd

DEBUG = False


# acceptable image suffixes
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp', '.ppm', '.pgm')

NET_FILE = os.path.join(os.path.dirname(__file__), "plm_model.prototxt")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights_plm")

TILE_SIZE = 256
PADDING_SIZE = 21

# number of subwindows processed by a network in a batch
# Higher numbers speed up processing (only marginally if BATCH_SIZE > 16)
# The larger the batch size, the more memory is consumed (both CPU and GPU)
BATCH_SIZE=1

LEFT_EDGE = -2
TOP_EDGE = -1
MIDDLE = 0
RIGHT_EDGE = 1
BOTTOM_EDGE = 2


def relative_darkness(im, window_size=5, threshold=10):
	if im.ndim == 3:
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# find number of pixels at least $threshold less than the center value
	def below_thresh(vals):
		center_val = vals[vals.shape[0]/2]
		lower_thresh = center_val - threshold
		return (vals < lower_thresh).sum()

	# find number of pixels at least $threshold greater than the center value
	def above_thresh(vals):
		center_val = vals[vals.shape[0]/2]
		above_thresh = center_val + threshold
		return (vals > above_thresh).sum()
		
	# apply the above function convolutionally
	lower = nd.generic_filter(im, below_thresh, size=window_size, mode='reflect')
	upper = nd.generic_filter(im, above_thresh, size=window_size, mode='reflect')

	# number of values within $threshold of the center value is the remainder
	# constraint: lower + middle + upper = window_size ** 2
	middle = np.empty_like(lower)
	middle.fill(window_size*window_size)
	middle = middle - (lower + upper)

	# scale to range [0-255]
	lower = lower * (255 / (window_size * window_size))
	middle = middle * (255 / (window_size * window_size))
	upper = upper * (255 / (window_size * window_size))

	return np.concatenate( [lower[:,:,np.newaxis], middle[:,:,np.newaxis], upper[:,:,np.newaxis]], axis=2)
	

def setup_network(weight_file):
	return caffe.Net(NET_FILE, os.path.join(WEIGHTS_DIR, weight_file), caffe.TEST)


def fprop(network, ims, batchsize=BATCH_SIZE):
	# batch up all transforms at once
	idx = 0
	responses = list()
	while idx < len(ims):
		sub_ims = ims[idx:idx+batchsize]

		network.blobs["data"].reshape(len(sub_ims), ims[0].shape[2], ims[0].shape[1], ims[0].shape[0])

		for x in range(len(sub_ims)):
			transposed = np.transpose(sub_ims[x], [2,0,1])
			transposed = transposed[np.newaxis, :, :, :]
			network.blobs["data"].data[x,:,:,:] = transposed

		idx += batchsize

		# propagate on batch
		network.forward()
		output = np.copy(network.blobs["prob"].data)
		responses.append(output)
		print "Progress %d%%" % int(100 * idx / float(len(ims)))
	return np.concatenate(responses, axis=0)


def predict(network, ims):
	all_outputs = fprop(network, ims)
	predictions = np.squeeze(all_outputs, axis=1)
	return predictions


def get_subwindows(im):
	height, width, = TILE_SIZE, TILE_SIZE
	y_stride, x_stride, = TILE_SIZE - (2 * PADDING_SIZE), TILE_SIZE - (2 * PADDING_SIZE)
	if (height > im.shape[0]) or (width > im.shape[1]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, tokens)
		exit(1)
	ims = list()
	bin_ims = list()
	locations = list()
	y = 0
	y_done = False
	while y  <= im.shape[0] and not y_done:
		x = 0
		if y + height > im.shape[0]:
			y = im.shape[0] - height
			y_done = True
		x_done = False
		while x <= im.shape[1] and not x_done:
			if x + width > im.shape[1]:
				x = im.shape[1] - width
				x_done = True
			locations.append( ((y, x, y + height, x + width), 
					(y + PADDING_SIZE, x + PADDING_SIZE, y + y_stride, x + x_stride),
					 TOP_EDGE if y == 0 else (BOTTOM_EDGE if y == (im.shape[0] - height) else MIDDLE),
					  LEFT_EDGE if x == 0 else (RIGHT_EDGE if x == (im.shape[1] - width) else MIDDLE) 
			) )
			ims.append(im[y:y+height,x:x+width,:])
			x += x_stride
		y += y_stride

	return locations, ims


def stich_together(locations, subwindows, size):
	output = np.zeros(size, dtype=np.float32)
	for location, subwindow in zip(locations, subwindows):
		outer_bounding_box, inner_bounding_box, y_type, x_type = location
		y_paste, x_paste, y_cut, x_cut, height_paste, width_paste = -1, -1, -1, -1, -1, -1
		#print outer_bounding_box, inner_bounding_box, y_type, x_type

		if y_type == TOP_EDGE:
			y_cut = 0
			y_paste = 0
			height_paste = TILE_SIZE - PADDING_SIZE
		elif y_type == MIDDLE:
			y_cut = PADDING_SIZE
			y_paste = inner_bounding_box[0]
			height_paste = TILE_SIZE - 2 * PADDING_SIZE
		elif y_type == BOTTOM_EDGE:
			y_cut = PADDING_SIZE
			y_paste = inner_bounding_box[0]
			height_paste = TILE_SIZE - PADDING_SIZE

		if x_type == LEFT_EDGE:
			x_cut = 0
			x_paste = 0
			width_paste = TILE_SIZE - PADDING_SIZE
		elif x_type == MIDDLE:
			x_cut = PADDING_SIZE
			x_paste = inner_bounding_box[1]
			width_paste = TILE_SIZE - 2 * PADDING_SIZE
		elif x_type == RIGHT_EDGE:
			x_cut = PADDING_SIZE
			x_paste = inner_bounding_box[1]
			width_paste = TILE_SIZE - PADDING_SIZE

		#print (y_paste, x_paste), (height_paste, width_paste), (y_cut, x_cut)

		output[y_paste:y_paste+height_paste, x_paste:x_paste+width_paste] = subwindow[y_cut:y_cut+height_paste, x_cut:x_cut+width_paste]

	return output

def avg_ims(ims):
	ims = np.concatenate(ims, axis=0)
	avg_im = ims.mean(axis=0, dtype=float)

	high_indices = avg_im > 128
	low_indices = avg_im <= 128
	avg_im[high_indices] = 255
	avg_im[low_indices] = 0

	avg_im = avg_im.astype(np.uint8)


def main(in_image, out_image):
	print "Loading Image"
	image = cv2.imread(in_image, cv2.IMREAD_COLOR)

	print "Computing RD features"
	rd_im = relative_darkness(image)
	if DEBUG:
		cv2.imwrite('rd.png', rd_im)

	print "Concating inputs"
	data = np.concatenate([image, rd_im], axis=2)

	print "Preprocessing"
	data = 0.003921568 * (data - 127.)
	print data.shape

	print "Tiling input"
	locations, subwindows = get_subwindows(data)

	results = list()

	weight_files = os.listdir(WEIGHTS_DIR)
	for idx, weight_file in enumerate(weight_files):
		network = setup_network(weight_file)
		print "Starting predictions for network %d/%d" % (idx+1, len(weight_files))
		raw_subwindows = predict(network, subwindows)

		print "Reconstructing whole image from binarized tiles"
		result = stich_together(locations, raw_subwindows, tuple(image.shape[0:2]))

		result = result
		if DEBUG:
			cv2.imwrite("out_%d.png" % idx, 255 - (255 * result).astype(np.uint8))

		results.append(result)

	print "Averaging Predictions"
	average_result = np.average(np.concatenate(map(lambda arr: arr[np.newaxis,:,:], results), axis=0), axis=0)
	low_idx = average_result < 0.5
	bin_result = np.zeros(average_result.shape, dtype=np.uint8)
	bin_result[low_idx] = 255
	
	print "Writing Final Result"
	cv2.imwrite(out_image, bin_result)

	print "Done"
	print "Exiting"


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "USAGE: python binarize.py in_image out_image [gpu#]"
		print "\tin_image is the input image to be binarized"
		print "\tout_image is where the binarized image will be written to"
		print "\tgpu is an integer device ID to run networks on the specified GPU.  If ommitted, CPU mode is used"
		exit(1)
	# only required argument
	in_image = sys.argv[1]

	# attempt to parse an output directory
	out_image = sys.argv[2]

	# use gpu if specified
	try:
		gpu = int(sys.argv[3])
		if gpu >= 0:
			caffe.set_mode_gpu()
			caffe.set_device(gpu)
	except:
		caffe.set_mode_cpu()

	main(in_image, out_image)
	
