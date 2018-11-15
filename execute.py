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
	resultat =[]
	for N in range(1,18):
		inputs = analyse("toTest/"+str(N)+".jpg")

		lvl_0 = sigmoid(np.dot(inputs, w_0))

		lvl_1 = sigmoid(np.dot(lvl_0, w_1))

		resultat.append(Action_Potential(lvl_1[0]))
		resultat.append(Action_Potential(lvl_1[1]))

		if resultat[0]==0 and resultat[1]==0:
			print("Картинка номер: ",N, "грусная	", *resultat)

		elif resultat[0]==1 and resultat[1]==0:
			print("Картинка номер: ",N, "весёлая	", *resultat)

		elif resultat[0]==0 and resultat[1]==1:
			print("Картинка номер: ",N, "это цифра 1: ", *resultat)

		elif resultat[0]==1 and resultat[1]==1:
			print("Картинка номер: ",N, "это цифра 2: ", *resultat)

		resultat.clear()
	input()

if __name__ == '__main__':
	main()
