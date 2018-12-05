from numpy import load
import library as lb

def Action_Potential(M_applicated):
	if M_applicated >= 0.5:
		return 1
	return 0

def say_TO_User(resultat, N):
	if resultat[0]==0 and resultat[1]==0:
		print("Картинка номер: ",N, "грусная	", *resultat)

	elif resultat[0]==1 and resultat[1]==0:
		print("Картинка номер: ",N, "весёлая	", *resultat)

	elif resultat[0]==0 and resultat[1]==1:
		print("Картинка номер: ",N, "это цифра 1: ", *resultat)

	else: #resultat[0]==1 and resultat[1]==1:
		print("Картинка номер: ",N, "это цифра 2: ", *resultat)


def main():
	w_0 = load("W_0.npy")
	w_1 = load("W_1.npy")
	analysing =  lb.analyse()
	summ_it = lb.calculet()
	resultat = []

	for N in range(1,18):

		inputs = analysing.transform_in_data("toTest/"+str(N)+".jpg")

		lvl_0 = summ_it.dot_sigmoid(inputs, w_0)

		lvl_1 = summ_it.dot_sigmoid(lvl_0, w_1)

		resultat.append(Action_Potential(lvl_1[0]))
		resultat.append(Action_Potential(lvl_1[1]))

		say_TO_User(resultat, N)

		resultat.clear()
	input()

if __name__ == '__main__':
	main()
