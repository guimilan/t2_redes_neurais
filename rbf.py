import numpy as np
import imageio
import math
import pickle
import os
import json

#Classe que representa o multilayer perceptron
class RBFNet():
	#Construtor. Recebe o tamanho da cadama de entrada, centroides, dispersao, tamanho da camada de saida
	#e taxa de aprendizado
	def __init__(self, input_length, centers, spread, output_length, learning_rate=5e-1):
		self.input_length = input_length
		self.centers = centers
		self.spread = spread
		self.hidden_length = centers.shape[0]
		self.output_length = output_length
		self.learning_rate = learning_rate

		#Inicializa os pesos da camada de saida aleatoriamente, representado-os na forma de matriz
		#Os pesos e vies de cada neuronio sao dispostos em linhas
		#Em hidden_length+1, o +1 serve para representar o vies
		self.output_layer = np.random.uniform(-0.5, 0.5, (self.output_length, self.hidden_length+1))

	def save_to_disk(self, file_name):
		print('Saving model to', file_name)
		with open(file_name, 'wb') as file:
			pickle.dump(self, file)

	#Funcao de ativacao (gaussiana)
	#net, neste caso, é só o dado de entrada (nao tem camada anterior à camada RBF)
	def activ(self, net):
		#Cria um vetor com a distancia da entrada para cada centroide
		dists = np.array([np.linalg.norm(net - self.centers[i]) for i in range(self.centers.shape[0])])
		#Calcula a funcao de ativacao (gaussiana) para cada distancia
		fnet = np.exp(-((dists**2)/(2*(self.spread**2))))
		return fnet

	#Faz forward propagation, retornando apenas o vetor produzido pela camada de saida
	#Isto eh feito porque, para usos do forward propagation fora do treinamento, 
	#nao interessa saber o valor produzido pela camada oculta
	def forward(self, input_vect):
		return self.forward_training(input_vect)[1]

	#Faz forward propagation (calcula a predicao da rede)
	#Retorna tanto a saida da camada oculta quanto da camada de saida, 
	#usados no algoritmo de treinamento
	def forward_training(self, input_vect):
		input_vect = np.array(input_vect)
		#Checa se o tamanho da entrada corresponde ao que eh esperado pela rede
		if(input_vect.shape[0] != self.input_length):
			message = 'Tamanho incorreto de entrada. Recebido: {} || Esperado: {}'.format(input_vect.shape[0], self.input_length)
			raise Exception(message)

		#Passa o vetor de entrada pela camada oculta, calculando a sua distancia para cada centroide
		hidden_fnet = self.activ(input_vect)

		#Adiciona um componente "1" ao vetor produzido pela camada oculta para permitir calculo do bias
		#na camada de saida
		biased_hidden_activ = np.zeros((self.hidden_length+1))
		biased_hidden_activ[0:self.hidden_length] = hidden_fnet[:]
		biased_hidden_activ[self.hidden_length] = 1
		
		#Calcula a transformacao feita pela camada de saida usando produto de matriz por vetor
		#Wo x H = net, sendo Wo a matriz de pesos da camada de saida e H o vetor produzido pela ativacao
		#da camada oculta
		out_net = np.dot(self.output_layer, biased_hidden_activ)
		#Como a camada de saida usa ativacao linear, nenhuma ativacao é aplicada

		#Retorna ativacao da camada oculta 
		return hidden_fnet, out_net

	#Treina a rede aplicando recursive least-squares (RLS)
	def fit(self, input_samples, target_labels, threshold, learning_rate=None):
		if(learning_rate is not None):
			self.learning_rate = learning_rate

		#Erro quadratico medio eh inicializado com um valor arbitrario (maior que o threshold de parada)
		#p/ comecar o treinamento
		mean_squared_error = 2*threshold
		previous_mean_squared_error = 0.001

		#Inicializa o numero de epocas ja computadas
		epochs = 0

		#Enquanto não chega no erro quadratico medio desejado ou atingir 5000 epocas, continua treinando
		while(mean_squared_error > threshold and epochs < 5000 and \
		 math.fabs(mean_squared_error-previous_mean_squared_error)/previous_mean_squared_error > 10e-2):
			#Erro quadratico medio da epoca eh inicializado com 0
			previous_mean_squared_error = mean_squared_error
			mean_squared_error = 0
			
			#Passa por todos os exemplos do dataset
			for i in range(0, input_samples.shape[0]):
				#if(i % 200 == 0):
				#	print('Adjusting for sample', i)
				#Pega o exemplo da iteracao atual
				input_sample = input_samples[i]
				#Pega o label esperado para o exemplo da iteracao atual
				target_label = target_labels[i]

				#Pega net e f(net) da camada oculta e da camada de saida
				hidden_net, hidden_fnet, out_net, out_fnet = self.forward_training(input_samples[i])
				
				#Cria um vetor com o erro de cada neuronio da camada de saida
				error_array = (target_label - out_fnet)

				#Calcula a variacao dos pesos da camada de saida com a regra delta generalizada
				#delta_o_pk = (Ypk-Ok)*Opk(1-Opk), sendo p a amostra atual do conjunto de treinamento,
				#e k um neuronio da camada de saida. Ypk eh a saida esperada do neuronio pelo exemplo do dataset,
				#Opk eh a saida de fato produzida pelo neuronio
				delta_output_layer = error_array * self.deriv_activ(out_fnet)

				#Calcula a variacao dos pesos da camada oculta com a regra delta generalizada
				#delta_h_pj = f'(net_h)*(1-f(net_h))*somatoria(delta_o_k*wkj)
				output_weights = self.output_layer[:,0:self.hidden_length]

				hidden_layer_local_gradient = np.zeros(self.hidden_length)
				for hidden_neuron in range(0, self.hidden_length):
					for output_neuron in range(0, self.output_length):
							hidden_layer_local_gradient[hidden_neuron] += delta_output_layer[output_neuron]*\
								output_weights[output_neuron, hidden_neuron]
				
				delta_hidden_layer = self.deriv_activ(hidden_fnet) * hidden_layer_local_gradient
				
				hidden_fnet_with_bias = np.zeros(hidden_fnet.shape[0]+1)
				hidden_fnet_with_bias[0:self.hidden_length] = hidden_fnet[:]
				hidden_fnet_with_bias[self.hidden_length] = 1
				#Atualiza os pesos da camada de saida
				#Wkj(t+1) = wkj(t) + eta*deltak*Ij
				for neuron in range(0, self.output_length):
					for weight in range(0, self.output_layer.shape[1]):
						self.output_layer[neuron, weight] = self.output_layer[neuron, weight] + \
							self.learning_rate * delta_output_layer[neuron] * hidden_fnet_with_bias[weight]

				#Atualiza os pesos da camada oculta com a regra delta generalizada
				#Pega os pesos dos neuronios da camada de saida (bias da camada de saida nao entra)
				#Wji(t+1) = Wji(t)+eta*delta_h_j*Xi
				input_sample_with_bias = np.zeros(input_sample.shape[0]+1)
				input_sample_with_bias[0:input_sample.shape[0]] = input_sample[:]
				input_sample_with_bias[input_sample.shape[0]] = 1
				for neuron in range(0, self.hidden_length):
					for weight in range(0, self.hidden_layer.shape[1]):
						self.hidden_layer[neuron, weight] = self.hidden_layer[neuron, weight] + \
							self.learning_rate*delta_hidden_layer[neuron]*input_sample_with_bias[weight]
							#np.dot(delta_hidden_layer.T, input_sample_with_bias)

				#O erro da saída de cada neuronio é elevado ao quadrado e somado ao erro total da epoca
				#para calculo do erro quadratico medio ao final
				mean_squared_error = mean_squared_error + np.sum(error_array**2)			
			
			#Divide o erro quadratico total pelo numero de exemplos para obter o erro quadratico medio
			mean_squared_error = mean_squared_error/input_samples.shape[0]
			#print('Erro medio quadratico', mean_squared_error)
			epochs = epochs + 1
			#if(epochs % 1000 == 0):
			#print('End of epoch no. {}. rmse={}'.format(epochs, mean_squared_error))

		print('Total epochs run', epochs)
		print('Final rmse', mean_squared_error)
		return None

#Testa a mlp com funcoes logicas
def test_logic():
	mlp = MLP(*(2, 2, 1))

	print('\n\noutput before backpropagation')
	print('[0,0]=', mlp.forward([0,0]))
	print('[0,1]=', mlp.forward([0,1]))
	print('[1,0]=', mlp.forward([1,0]))
	print('[1,1]=', mlp.forward([1,1]))

	print('layers before backprop')
	print('hidden', mlp.hidden_layer)
	print('output layer', mlp.output_layer)
	print('\n')	

	x = np.array([[0,0],[0,1],[1,0],[1,1]])
	target = np.array([0, 0, 0, 1])
	mlp.fit(x, target, 5e-1, 10e-1)

	print('\n\noutput after backpropagation')
	print('[0,0]=', mlp.forward([0,0]))
	print('[0,1]=', mlp.forward([0,1]))
	print('[1,0]=', mlp.forward([1,0]))
	print('[1,1]=', mlp.forward([1,1]))
	print('layers after backprop')
	print('hidden', mlp.hidden_layer)
	print('output layer', mlp.output_layer)

#Carrega o dataset de digitos
def load_digits():
	data = np.zeros([1593, 256])
	labels = np.zeros([1593, 10])

	with open('semeion.data') as file:
		for image_index, line in enumerate(file):
			number_list = np.array(line.split())
			image = number_list[0:256].astype(float).astype(int)
			classes = number_list[256:266].astype(float).astype(int)
			data[image_index,:] = image[:]
			labels[image_index,:] = classes[:]
			
	return data, labels

#Plota uma imagem
def plot_image(image):
	news_image = image.reshape(16,16)
	plt.imshow(new_image)
	plt.show()

#Faz predicao da classe de todos os dados e compara com as classes esperadas
def measure_score(mlp, data, target):
	dataset_size = target.shape[0]
	score = 0
	
	for index, data in enumerate(data):
		expected_class = np.argmax(target[index])
		predicted_class = np.argmax(mlp.forward(data))
		if(expected_class == predicted_class):
			score += 1

	return score, (score/dataset_size)*100	

#Embaralha dois arrays de forma simetrica
def shuffle_two_arrays(data, labels):
	permutation = np.random.permutation(data.shape[0])
	return data[permutation], labels[permutation]

#Gera os indices de cada um dos k-folds
#ex: dataset de 10 elementos dividido em 5 folds
#retorna [[0,1][2,3][4,5][6,7][8,9]], em que o n-esimo vetor interno
#tem os indices dos elementos que pertencem ao n-esimo fold 
def k_folds_split(dataset_size, k):
	fold_size = int(dataset_size/k)
	folds = np.zeros((k, fold_size), dtype=int)

	for current_k in range(0, k):
		fold_indexes = range(current_k*fold_size, (current_k+1)*fold_size)
		folds[current_k] = fold_indexes

	print('folds', folds)
	return folds

#Recebe os folds e retorna as listas (dos indices) dos elementos que pertencem 
#ao conjunto de treinamento e de teste, para todos os testes
#ex. se tem 10 folds, retorna 10 listas de treino e 10 de teste
#o primeiro par (treino, teste) usa o primeiro fold para teste e os demais para treino, o segundo
#par (treino, teste) usa o segundo fold para teste e os demais para treino, e assim por diante
#Retorna estes conjuntos no formato de uma lista de indices dos elementos que pertencem a cada
#conjunto
def train_test_split(folds):
	fold_qtt = folds.shape[0]
	fold_size = folds.shape[1]
	train_set_size = (fold_qtt-1)*fold_size
	test_set_size = fold_size

	train_sets = np.zeros((fold_qtt, train_set_size), dtype=int)
	test_sets = np.zeros((fold_qtt, test_set_size), dtype=int)

	for fold_to_skip in range(0, fold_qtt):
		train_set = np.zeros(train_set_size, dtype=int)
		test_set = np.zeros(test_set_size, dtype=int)
		added_folds = 0

		for fold_index, current_fold in enumerate(folds):
			if(fold_index != fold_to_skip):
				train_set[added_folds*fold_size:(added_folds+1)*fold_size] = current_fold
				added_folds += 1
			else:
				test_set[0:fold_size] = current_fold

		train_sets[fold_to_skip] = train_set
		test_sets[fold_to_skip] = test_set

	return train_sets, test_sets

#Faz k-fold cross validation em uma mlp
def k_fold_cross_validation(mlp, data, labels, k):
	print('k-fold cross validation')
	print('Shuffling data...')
	shuffled_data, shuffled_labels = shuffle_two_arrays(data, labels)
	print('Splitting in folds')
	folds = k_folds_split(shuffled_data.shape[0], 5)
	print('Building train and test set indexes...')
	train_sets, test_sets = train_test_split(folds)
	
	scores = []
	accuracies = []
	for index, (train_set, test_set) in enumerate(zip(train_sets, test_sets)):
		#print('Performing validation with fold no. {}...'.format(index))
		#print('Training...')
		mlp = MLP(mlp.input_length, mlp.hidden_length, mlp.output_length, mlp.learning_rate)
		mlp.fit(data[train_set], labels[train_set], 5e-2)
		#print('Testing with test set', index)
		score, accuracy = measure_score(mlp, data[test_set], labels[test_set])
		print('accuracy {}: {}%'.format(index, accuracy))
		scores.append(score)
		accuracies.append(accuracy)

	print('===========================')
	print('NUMBER OF NEURONS', mlp.hidden_length)
	print('AVERAGE ACCURACY', np.sum(accuracies)/len(accuracies))
	print('===========================')

	return scores, accuracies

#Carrega uma mlp do disco a partir de um arquivo .pickle
def load_mlp_from_disk(filename):
	mlp = None
	if(os.path.isfile(filename)):
		with open(filename, 'rb') as file:
			mlp = pickle.load(file)
			print('MLP successfully loaded. Properties:')
			print('Input layer size: {}'.format(mlp.input_length))
			print('Hidden layer size: {}'.format(mlp.hidden_length))
			print('Output layer size: {}'.format(mlp.output_length))
			print('Learning rate: {}'.format(mlp.learning_rate))
	return mlp

#Grava os resultados do teste em um arquivo .json
def record_test_results(test_results, filename):
	print('Recording test results to', filename)
	with open(filename, 'w') as file:
		json.dump(test_results, file)

#Compoe os resultados do teste em um dicionario pra facilitar gravacao
def build_test_result_dict(mlp, scores, accuracies):
	test_results = dict()
	test_results['hidden_layer_size'] = mlp.hidden_length
	test_results['learning_rate'] = mlp.learning_rate
	test_results['scores'] = scores
	test_results['accuracies'] = accuracies
	return test_results

def main():
	#test_logic()
	'''data, labels = load_digits()
	#Testa a rede variando o tamanho da camada oculta e mantendo fixa a taxa de aprendizado
	tests = []
	for hidden_layer_size in range(1, 129):
		print('Testing for hidden layer size', hidden_layer_size)
		mlp = MLP(*[256, hidden_layer_size, 10], 5e-1)
		scores, accuracies = k_fold_cross_validation(mlp, data, labels, 5)
		tests.append(build_test_result_dict(mlp, scores, accuracies))
	record_test_results(tests, 'hidden_layer_results.json')
	
	#Testa a rede mantendo fixo o tamanho da camada oculta e variando a taxa de aprendizado
	tests = []
	for learning_rate in np.arange(10e-2, 1, 10e-2):
		print('Testing for learning rate', learning_rate)
		mlp = MLP(*[256, 128, 10], learning_rate)
		scores, accuracies = k_fold_cross_validation(mlp, data, labels, 5)
		tests.append(build_test_result_dict(mlp, scores, accuracies))
	record_test_results(tests, 'learning_rate_results.json')'''
	test_logic()

if __name__ == '__main__':
	main()