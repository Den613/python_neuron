""" входные данные: 
|0|0|0|->|0|
|0|0|1|->|1|
|0|1|0|->|0|
|0|1|1|->|0|
|1|0|0|->|1|
|1|0|1|->|1|
|1|1|0|->|0|
|1|1|1|->|0|
стрелками указонно то что нужно получит """

		

import math
import random
import numpy as np

class Neuron():
	def __init__(self, x = []):
		
		w1 = []
		w2 = []
		
		"""веса для первых нейронов"""
		for i in range(6):
			w1.append(random.uniform(-(2**-0.5),2**0.5))
		self.w1 = w1
		
		"""веса для выходного нейрона"""
		for i in range(2):
			w2.append(random.uniform(-(2**-0.5),2**0.5))
		self.w2 = w2
		
		"""входы"""
		self.x = x

	def sigmoid(self,A):
		"""функция активации"""
		f_A =  1/(1 + math.exp(-A))
		return f_A
		
	def print_w(self):
		"""функция печати весов и входов"""
		print(" W1: ", self.w1)
		print(" W2: ", self.w2)
		print(" X : ", self.x)
		
	def sum_x_w1(self):
		"""суммирование для первого скрытого нейрона"""
		
		A1 = x[0]*self.w1[0] + x[1]*self.w1[1] + x[2]*self.w1[2]
				
		return self.sigmoid(A1)
		
		
	def sum_x_w2(self):
		"""суммирование для второго скрытого нейрона"""
		
		A2 = x[0]*self.w1[3] + x[1]*self.w1[4] + x[2]*self.w1[5]
		
		return self.sigmoid(A2)
	
	def sum_f_w3(self):
		"""суммирование для выходного нейрона"""
		
		A3 = self.sum_x_w1()*self.w2[0] + self.sum_x_w2()*self.w2[1]
		
		return self.sigmoid(A3)
	
	
	def learn_w2(self,acl, exd,l_r):
		"""обучение"""
		self.w_2 = []
		
		err = acl - exd
		
		self.w_d = err*(self.sum_f_w3()*(1 - self.sum_f_w3()))
		
		w_1 = self.w2[0] - self.sum_x_w1()*self.w_d*l_r
		w_2 = self.w2[1] - self.sum_x_w2()*self.w_d*l_r
		
		self.w_2 = [w_1, w_2]
		
		return self.w_2
		
	def learn_w1(self,l_r):
		"""обучение"""
		w_1 = []
		
		e_rr1 = self.w_2[0]*self.w_d
		e_rr2 = self.w_2[1]*self.w_d
	
		w_d1 = e_rr1*(self.sum_x_w1()*(1 - self.sum_x_w1()))
		w_d2 = e_rr2*(self.sum_x_w2()*(1 - self.sum_x_w2()))
		
		w_1 = self.w1[0] - self.x[0]*w_d1*l_r
		w_2 = self.w1[1] - self.x[1]*w_d1*l_r
		w_3 = self.w1[2] - self.x[2]*w_d1*l_r
		
		w_4 = self.w1[3] - self.x[0]*w_d2*l_r
		w_5 = self.w1[4] - self.x[1]*w_d2*l_r
		w_6 = self.w1[5] - self.x[2]*w_d2*l_r
		
		w_1 = [w_1, w_2, w_3, w_4, w_5, w_6]
		 
		return w_1
	
	def out(self):
		"""результат выхода"""
		
		if self.sum_f_w3() <= 0.5:
			n = 0
		else:
			n = 1
	
		return n

def MSE(y, Y):
	return np.mean((y-Y)**2) 

"""входы"""
x = [0,0,0]
A = Neuron(x)
#A.out()
l = 0.1
eps = 0.42
N = 6000000
#print(A.sum_f_w3())
print("Begin W2: ",A.w2)
print("Begin W1: ",A.w1)
#A.w1 = A.learn_w1()
#A.w2 = A.learn_w2(A.sum_f_w3(),0,0.1)

#print("Out: ",A.sum_f_w3())

print("====================================================")

 
b = [
	[0,0,0],
	[1,1,1],
	[1,0,0],
	[0,1,0],
	[0,0,1],
	[1,1,0],
	[0,0,1],
	[1,0,1]]
n = len(b)
print()
"""учитель"""
S1 = [0,0,1,0,1,0,0,1]
size = []	
h = 0
w = 1
error = [0 for i in range(2000)]
"""обучаем для каждых 3-х входов"""
for j in range(N):
	for i in range(n):
		print("==============")
		#b.append(b[i])
		#S1.append(S1[i])
		#n.append(i)
	
		x = b[i]
		S = S1[i]
	
	
		A.x = x
		A.w2 = A.learn_w2(A.sum_f_w3(),S,l)
		A.w1 = A.learn_w1(l)
		B = A.sum_f_w3()
		#Br = round(B,5)
		M = MSE(A.sum_f_w3(),S)
	 
		print("out: ",B, " n: ",i+1, " ",S ," = ",A.out())
		size.append(A.w2)
		h = M + h
		if i == 7:
			
			if h < eps:
				w = 0
				h = h
				break
			print(h)
			h = 0
	print(" opoha: ",j)
	if w == 0:
		break
print("-------------------")
"""результаты весов"""
print(" W2: ",A.w2)
print(" W1: ",A.w1)
print(" M : ",h)
print("-------------------")


#Tt = 0
#for i in range(5):
#	Tt += i
#	size.append(Tt)

#print(size)


with open('test.txt','w') as file:
	for i in range(0, len(size)):
		file.write(str(size[i][0])+","+'\n')
