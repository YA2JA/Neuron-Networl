import os, math
import numpy as np
from PIL import Image
from pybrain3.datasets import SupervisedDataSet

										#THIS PARTIE OF CODE IS NOT ENDED!!!

class Neuron_network():

	def __init__(self):
		self.learnig_data_placement = os.path.dirname(os.path.realpath(__file__))+'/learnig_data/Folder'
		self.all_right_answer = ['0,0','0,1','1,0','1,1']
		self.all_files = []
		self.imgs_arrays = [[],[],[],[]]
		self.weights = SupervisedDataSet(1024, 2)

class reading_imgs(Neuron_network):
		
	def Found_Learnig_Filles(self):
		for i in range(4):
			self.all_files.append(os.listdir(self.learnig_data_placement+str(i)))

	def add_information(self,resultat_nember=0):
		img_adresse = self.all_files[resultat_nember]
		right_answer = self.all_right_answer[resultat_nember]
		for image_name in img_adresse:
			img = self.binnarizing_img(self.learnig_data_placement+str(resultat_nember)+'/'+image_name)
			self.imgs_arrays[resultat_nember].append(np.asarray(img))

		for img_nember in range(len(self.all_files[resultat_nember])-1):	
			end_response = self.flatting_arrays(self.imgs_arrays[resultat_nember][img_nember].tolist())
			self.weights.addSample((end_response), self.tranlater(right_answer))

	def binnarizing_img(self, img_adresse):
		file = Image.open(img_adresse)
		img_convert = file.convert("L")
		data = np.asarray(img_convert)
		resultat = (data < 200) * 1
		return resultat
	
	def flatting_arrays(self, array):
		flatten = lambda l: [item for sublist in l for item in sublist]
		return flatten(array)

	def tranlater(self, x):
		if '0,0' == x:
			return 0,0
		if '0,1'== x:
			return 0,1
		if '1,0' == x:
			return 1,0
		if '1,1'== x:
			return 1,1 

class learning(Neuron_network):

	def __init__(self, weights):
		self.alphas = [ 1/10**i for i in range(5,-5,-1)]
		self.start_layer = np.array(weights['input'])
		self.right_answer = np.array(weights['target'])
		self.nember_training = 5000

	def trening(self):

		synapse_0 = 2*np.random.random((1024,256))-1
		synapse_1 = 2*np.random.random((256,2))-1
		min_Error = 0.99
		for alpha in self.alphas:
			print("\nAlpha:" + str(alpha))

			for j in range(self.nember_training):
				layer_0 = self.start_layer
				layer_1 = self.sigmoid(np.dot(layer_0,synapse_0))
				layer_2 = self.sigmoid(np.dot(layer_1,synapse_1))
				layer_2_error = layer_2 - self.right_answer
				Pro_error = np.mean(np.abs(layer_2_error))
				if (j% 1000) == 0:
					print("Ошибка после "+str(j)+" повторений:" + str(Pro_error))
					if min_Error == Pro_error: continue

					if min_Error > Pro_error: min_Error = Pro_error 

				layer_2_delta = layer_2_error*self.sigmoid_output_to_derivative(layer_2)

				layer_1_error = layer_2_delta.dot(synapse_1.T)

				layer_1_delta = layer_1_error * self.sigmoid_output_to_derivative(layer_1)

				synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
				synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))

			self.nember_training += 1000
		
	def dot_sigmoid(self, x, y):
		inputs = np.dot(x, y)
		output = 1/(1+np.exp(-inputs))
		return output

	def sigmoid(self,x):
		return 1/(1+np.exp(-x))

	def sigmoid_output_to_derivative(self, output):
		return output*(1-output)



def main():
	dowload_img_base = reading_imgs()
	dowload_img_base.Found_Learnig_Filles()
	for folder_number in range(4):
		dowload_img_base.add_information(folder_number)
	network = learning(dowload_img_base.weights)
	network.trening()
	input()


if __name__ == '__main__':
	main()