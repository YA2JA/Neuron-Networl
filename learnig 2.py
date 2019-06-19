import os, math
import numpy as np
from PIL import Image
from pybrain3.datasets import SupervisedDataSet
import pandas as pd

										#THIS PARTIE OF CODE IS NOT ENDED!!!
class download_data_to_learn():
	
	def __init__(self):
		self.learnig_data_placement = os.path.dirname(os.path.realpath(__file__))+'/learnig_data/Folder'
		self.all_files = []
		self.imgs_arrays = [[],[],[],[]]
		self.weights = SupervisedDataSet(1024, 2)
		
	def Found_Learnig_Filles(self):
		for i in range(4):
			self.all_files.append(os.listdir(self.learnig_data_placement+str(i)))

	def add_learning_information(self,resultat_nember=0):
		img_adresse = self.all_files[resultat_nember]
		for image_name in img_adresse:
			img = self.binnarizing_img(self.learnig_data_placement+str(resultat_nember)+'/'+image_name)
			self.imgs_arrays[resultat_nember].append(np.asarray(img))

		for img_nember in range(len(self.all_files[resultat_nember])-1):	
			end_response = self.flatting_arrays(self.imgs_arrays[resultat_nember][img_nember].tolist())
			self.weights.addSample((end_response), self.tranlater(resultat_nember))

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
		binar_number = '{0:02b}'.format(x)
		return [int(i) for i in list(binar_number)]

class learning():

	def __init__(self, weights):
		self.alphas = [ 1/10**i for i in range(4,-15,-1)]
		self.start_layer = np.array(weights['input'])
		self.right_answer = np.array(weights['target'])
		self.nember_training = 9000
		self.min_Error = 0.99

	def trening(self):

		synapse_0 = 2*np.random.random((1024,256))-1
		synapse_1 = 2*np.random.random((256,2))-1

		for alpha in self.alphas:
			print("\nAlpha:" + str(alpha))

			for j in range(int(self.nember_training)):
				layer_0 = self.start_layer
				layer_1 = self.sigmoid(np.dot(layer_0,synapse_0))
				layer_2 = self.sigmoid(np.dot(layer_1,synapse_1))
				layer_2_error = layer_2 - self.right_answer
				Pro_error = np.mean(np.abs(layer_2_error))
				if (j% 1000) == 0:
					print("Ошибка после "+str(j)+" повторений:" + str(Pro_error))

					if self.min_Error > Pro_error: 
						self.min_Error = Pro_error
						self.__save_learn_data__(synapse_0,synapse_1)


				layer_2_delta = layer_2_error*self.sigmoid_output_to_derivative(layer_2)

				layer_1_error = layer_2_delta.dot(synapse_1.T)

				layer_1_delta = layer_1_error * self.sigmoid_output_to_derivative(layer_1)

				synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
				synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))

			
		
	def dot_sigmoid(self, x, y):
		inputs = np.dot(x, y)
		output = 1/(1+np.exp(-inputs))
		return output

	def sigmoid(self,x):
		return 1/(1+np.exp(-x))

	def sigmoid_output_to_derivative(self, output):
		return output*(1-output)

	def __save_learn_data__(self, synapse_0, synapse_1):
		np.save("W_0", synapse_0)
		np.save("W_1", synapse_1)



def main():
	
	data = download_data_to_learn()
	data.Found_Learnig_Filles()
	for answer_nember in range(4):
		data.add_learning_information(answer_nember)
	
	network = learning(data.weights)
	network.trening()


if __name__ == '__main__':
	main()
	