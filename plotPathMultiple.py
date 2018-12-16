#plotPathMultiple.py

import pickle
import os.path as osp
import os
from track import *


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('pklname', nargs='*')
	args = parser.parse_args()

	num_pkl = len(args.pklname)
	print("number of pickle", num_pkl)
	record_buff_arr = []

	for pklname in args.pklname:
		path_to_pickle = "graphicReplayData/{}.pkl".format(pklname)
		with open(path_to_pickle, 'rb') as handle:
			record_buff_arr.append(pickle.load(handle))

	output_dir_name = "graphicReplay/{}".format(args.pklname[0])

	plot_path(record_buff_arr, output_dir_name)



def plot_path(record_buff_arr, output_dir_name):
	# Initialize directory
	assert not osp.exists(output_dir_name), "Log dir %s already exists! Delete it first or use a different dir"%output_dir_name
	os.makedirs(output_dir_name)

	obs_out_dir_name = output_dir_name + "/obs_images"
	assert not osp.exists(obs_out_dir_name), "Obs log dir %s already exists! Delete it first or use a different dir"%obs_output_dir_name
	os.makedirs(obs_out_dir_name)

	grid_out_dir_name = output_dir_name + "/grid_images"
	assert not osp.exists(grid_out_dir_name), "Grid log dir %s already exists! Delete it first or use a different dir"%grid_output_dir_name
	os.makedirs(grid_out_dir_name)


	num_car = len(record_buff_arr)
	curTrack = Track(0.6, 0.3)
	length = record_buff_arr[0]['length']
	width = record_buff_arr[0]['width']
	env_window_w = record_buff_arr[0]['env_window_w']
	obs_window_w = record_buff_arr[0]['obs_window_w']

	max_t = min([record_b['lastTime'] for record_b in record_buff_arr]) 


	def getObservationWindow(cur_location, cur_velocity):
		v_unit = cur_velocity/np.linalg.norm(cur_velocity)
		rotated_v_unit = utils.rotateVector(v_unit, 90)
		new_head = cur_location + np.multiply(v_unit, length/2)
		corner = new_head - np.multiply(rotated_v_unit, env_window_w/2)
		corner_1 = corner + np.multiply(v_unit, env_window_w)
		corner_2 = corner_1 + np.multiply(rotated_v_unit, env_window_w)
		corner_3 = corner_2 - np.multiply(v_unit, env_window_w)

		pts_src = np.zeros((4, 2))

		# Courtesy to https://www.learnopencv.com/homography-examples-using-opencv-python-c/
		corners = np.array([corner, corner_1, corner_2, corner_3])
		pts_src[:,0] = corners[:,1]
		pts_src[:,1] = corners[:,0]

		pts_dst = np.array([[0, 0],[obs_window_w, 0], [obs_window_w, obs_window_w], [0, obs_window_w]])

		# Calculate Homography
		h, status = cv2.findHomography(pts_src, pts_dst)

		im_dst = cv2.warpPerspective(curTrack.getGrid(), h, (obs_window_w, obs_window_w))
		return corner, im_dst

	for index in range(max_t - 1):
		g = curTrack.getGrid()
		fig,ax = plt.subplots(1)
		ax.imshow(g, cmap='gray')

		for record_buff in record_buff_arr:
			location = record_buff['data'][index, 0:2]
			velocity = record_buff['data'][index, 2:4]

			rect = utils.transformRectangle(location, velocity, length, width)
			curTrack.updateGrid(rect, 1)
			
			corner, im_dst = getObservationWindow(location, velocity)
			obsname = obs_out_dir_name + "/" + str(index) + ".png"
			plt.imsave(obsname, im_dst, cmap = 'gray')
			plt.close()

			corner_reorder = (corner[1], corner[0])

			theta = utils.getAngle(velocity) * 180/math.pi
			#print("theta", theta)

			g = curTrack.getGrid()
			fig,ax = plt.subplots(1)
			ax.imshow(g, cmap='gray')

			rect = patches.Rectangle(corner_reorder,env_window_w,env_window_w,angle = theta, linewidth=1,edgecolor='r',facecolor='none')
			ax.add_patch(rect)

		gridname = grid_out_dir_name +"/" + str(index) + ".png"
		plt.savefig(gridname, cmap = 'gray')
		plt.close()

		# clean up
		for record_buff in record_buff_arr:
			location = record_buff['data'][index, 0:2]
			velocity = record_buff['data'][index, 2:4]
			rect = utils.transformRectangle(location, velocity, length, width)
			curTrack.updateGrid(rect, 0)

		if index % 5 == 0:
			curTrack.rebuildTrack()


if __name__ == "__main__":
    main()