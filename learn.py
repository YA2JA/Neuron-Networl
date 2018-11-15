import numpy as np
from pybrain3.datasets import SupervisedDataSet
from PIL import Image
alphas = [0.006]

# подсчитаем нелинейную сигмоиду
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output


def cycle():
        lerns = SupervisedDataSet(1024, 2)
        for i in range(1,104):
            adress = "folder 1/"+str(i)+".jpg"
            data = analyse(adress)
            lerns.addSample((data), (1,0))

        for i in range(1,105):
            adress = "folder 2/"+str(i)+".jpg"
            data = analyse(adress)
            lerns.addSample((data), (0,0))

        for i in range(1,61):
        	adress = "folder 3/"+str(i)+".jpg"
        	data = analyse(adress)
        	lerns.addSample((data), (0,1))

        for i in range(1,62):
            adress = "folder 4/"+str(i)+".jpg"
            data = analyse(adress)
            lerns.addSample((data), (1,1))
        return lerns
        
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


# преобразуем результат сигмоиды к производной
def sigmoid_output_to_derivative(output):
    return output*(1-output)

def lost_use():
    file = open("info_about.txt", "r")
    data = file.read()
    to_return = ""
    N = 0
    for i in data:
        if i == " ":
            N+=1
        if N==9:
            to_return+=i
    return float(to_return)

now = cycle()

X = np.array(now["input"])

y = np.array(now["target"])

try:
    minPro_error = lost_use()
except:
    minPro_error = 1

nember_training = 90000
for alpha in alphas:

    print ("\nТренируемся при Alpha:" + str(alpha))
    np.random.seed(1)

    # случайная инициализация весов со средним 0
    synapse_0 = 2*np.random.random((1024,200)) - 1
    synapse_1 = 2*np.random.random((200,2)) - 1

    for j in range(nember_training):

        # Прямое распространение по уровням 0, 1 и 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,synapse_0))
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # как сильно ошиблись?
        layer_2_error = layer_2 - y
        
        Pro_error = np.mean(np.abs(layer_2_error))


        if (j% 1000) == 0:
            print("Ошибка после "+str(j)+" повторений:" + str(Pro_error))
            if Pro_error<minPro_error:
                minPro_error = Pro_error
                info_about = open("info_about.txt", "w")
                info_about.write("Alpha = "+str(alpha)+
                				"\nTotal nember of training :"+str(j)+
                				"\nMin of errors: " + str(Pro_error))
                np.save("W_0", synapse_0)
                np.save("W_1", synapse_1)
            if Pro_error==minPro_error:
                continue



        # в каком направлении цель?
        # уверены ли мы? Если да, то не нужно слишком сильных изменений
        layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)

        # насколько каждое значение из l1 влияет на ошибку в l2 (в соответствии с весами)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # в каком направлении цель l1?
        # уверены ли мы? Если да, то не нужно слишком сильных изменений
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
        synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))