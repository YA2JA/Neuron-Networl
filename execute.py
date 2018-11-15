import numpy as np
from PIL import Image
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def analyse(adress):
        img = img_read(adress)
        one = []
        for y in img:
            for x in y:
                one.append(x)
        return one

def img_read(adress):
        file = Image.open(adress)
        img_convert = file.convert("L")
        data = np.asarray(img_convert)
        resultat = (data > 200) * 1 
        return resultat

def Action_Potential(M_applicated):
	if M_applicated >= 0.5:
		return 1
	return 0

def main():
	w_0 = np.load("W_0.npy")

	w_1 = np.load("W_1.npy")
	
	for N in range(1,13):
		inputs = analyse("toTest/"+str(N)+".jpg")

		lvl_0 = sigmoid(np.dot(inputs, w_0))

		lvl_1 = sigmoid(np.dot(lvl_0, w_1))

		resultat = Action_Potential(lvl_1)

		print("картинка номер "+str(N),"грусная" if resultat == 0 else " весёлая")
if __name__ == '__main__':
	main()