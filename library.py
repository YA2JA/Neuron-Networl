#library to use
import numpy as np
from PIL import Image
class analyse():
	def transform_in_data(self, adress):
		img = self.__img_read(adress)
		one = []
		for y in img:
			for x in y:
				one.append(x)
		return one

	def __img_read(self, adress):
		file = Image.open(adress)
		img_convert = file.convert("L")
		data = np.asarray(img_convert)
		resultat = (data > 200) * 1 
		return resultat

class calculet():
	def dot_sigmoid(self, x, y):
		inputs = np.dot(x, y)
		output = 1/(1+np.exp(-inputs))
		return output