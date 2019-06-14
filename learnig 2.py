import os, math
import numpy as np
from PIL import Image
from pybrain3.datasets import SupervisedDataSet

										#THIS PARTIE OF CODE IS NOT ENDED!!!

class Neuron_network():

	def __init__(self):
		self.learnig_data_placement = os.path.dirname(os.path.realpath(__file__))+'/learnig_data/Folder'
		self.all_files = [[],[],[],[]]
		self.weights = SupervisedDataSet(1024, 2)
		self.imgs_arrays = [[],[],[],[]]


class reading_imgs(Neuron_network):

	def Found_Learnig_Filles(self):
		for i in range(4):
			self.all_files[i].append(os.listdir(self.learnig_data_placement+str(i)))

	def transform_img_array(self,img_adresse,resultat_nember=0):
		for image_name in img_adresse:
			img = self.binnarizing_img(self.learnig_data_placement+str(resultat_nember)+'/'+image_name)
			self.imgs_arrays[resultat_nember].append(np.asarray(img))

		for img_nember in range(len(self.all_files[0][0])-1):	
			end_response = self.flatting_array(self.imgs_arrays[resultat_nember][img_nember].tolist())
			self.weights.addSample((end_response), (0,0))

	def binnarizing_img(self, img_adresse):
		file = Image.open(img_adresse)
		img_convert = file.convert("L")
		data = np.asarray(img_convert)
		resultat = (data < 200) * 1
		return resultat
	
	def flatting_array(self, array):
		flatten = lambda l: [item for sublist in l for item in sublist]
		return flatten(array)
	
def main():
	dowload_img_base = reading_imgs()
	dowload_img_base.Found_Learnig_Filles()
	dowload_img_base.transform_img_array(dowload_img_base.all_files[0][0],0)

if __name__ == '__main__':
	main()