 1/1: %matplotlib inline
 1/2: from IPython.core.pylabtools import figsize
 1/3: import numpy as np
 1/4: from matplotlib import pyplot as plt
 2/1: %quickref
 2/2: %python main.py
 2/3: %%python main.py
 2/4: %run main.py
 2/5: %run ./main.py
 2/6: %run main.py
 2/7: %runmain.py
 2/8: %run .main.py
 2/9: pwd
2/10: cd ..
2/11: ls
2/12: cd bayes/
2/13: %run main.py
2/14: pwd
2/15: ls
2/16: %run src/main.py
 3/1: %run main.py
 3/2: %run main.py
 3/3: %run main.py
 3/4: %run main.py
 4/1: %paste
 5/1: import cv2 as cv
 5/2: img = cv.imread("fa67a7fb334114f440227930beff02d0.jpg")
 5/3: cv.namedWindow("Image")
 5/4: cv.imshow("Image",img)
 5/5: cv.waitKey(0)
 6/1: import pymc as pm
 8/1: import pymc as pm
 8/2: figsize(12,4)
 8/3: import numpy as np
 8/4: import scipy.stats as stats
 8/5: import matplotlib
 8/6: figsize(12,4)
 8/7: from IPython.core.pylabtools import figsize
 8/8: figsize(12,4)
 8/9: a = np.arange(16)
8/10: a
8/11: poi = stats.poisson
8/12: lambda_ = [1.5,4.25]
8/13: colors = ['#348ABD','#A60628']
8/14: import matplotlib as plt
8/15: plt.bar
8/16: from matplotlib import pyplot as plt
8/17: pmf
8/18: pm
8/19: pm(a,lambda_[0])
8/20: poi.pmf(a,lambda_[0])
8/21: poi.pmf(a,lambda_[0])
8/22: plt.bar(a,poi.pmf(a,lambda_[0]),color=colors[0],lable="$\lambda = %.1f$" % lambda_[0],alpha=0.60,edgecolor=colors[0],lw="3")
8/23: plt.bar(a,poi.pmf(a,lambda_[0]),color=colors[0],label="$\lambda = %.1f$" % lambda_[0],alpha=0.60,edgecolor=colors[0],lw="3")
8/24: plt.bar(a,poi.pmf(a,lambda_[1]),color=colors[1],label="$\lambda = %.1f$" % lambda_[1],alpha=0.60,edgecolor=colors[1],lw="3")
8/25: plt.xticks( a + 0.4,a)
8/26: plt.legend()
8/27: plt.ylabel("Probability of $k$")
8/28: plt.xlabel("$k$")
8/29: plt.title("Probability mass funciton of a Poisson random \$\lambda$ values")
8/30: plt.show()
 9/1: from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
 9/2: import numpy as np
 9/3:
disasters_array =   \
     np.array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                   3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                   2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                   1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                   0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                   3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
 9/4: switchPoint = DiscreteUniform('switchpoint', lower=0, upper=110, doc='Switchpoint[year]')
 9/5: early_mean = Exponential('early_mean', beta=1.)
 9/6: late_mean = Exponential('late_mean', beta=1.)
 9/7:
@deterministic(plot=false)
def rate(s=switchPoint, e=early_mean, l=late_mean):
    ''' Concatyente Poisson means '''
    out = np.empty(len(disasters_array))
    out[:s] = e
    out[s:] = l
    return out
 9/8:
@deterministic(plot=False)
def rate(s=switchPoint, e=early_mean, l=late_mean):
    ''' Concatyente Poisson means '''
    out = np.empty(len(disasters_array))
    out[:s] = e
    out[s:] = l
    return out
 9/9: disasters = Poisson('disasters', mu=rate, value=disasters_array, observed=True)
9/10: from pymc import MCMC
9/11: M = MCMC(disasters)
9/12: M.sample(iter=10000, burn=1000, thin=10)
9/13: M.trace('switchPoint')[:]
9/14: M.trace('switchpoint')[:]
10/1: fromm open
10/2: from opencv2
12/1: import cv2 as cv
13/1: import scipy.io
13/2: from scipy import io
13/3: import scipy
14/1:
class neuraNetwork:
    def __init__():
        pass
    
    def train():
        pass
    def query():
        pass
14/2:
class neuraNetwork:
    def __init__():
        pass
    
    def train():
        pass
    def query():
        pass
14/3:
class neuraNetwork:
    def __init__():
        pass
    
    def train():
        pass
    def query():
        pass
14/4: import numpy
15/1:
class neuraNetwork:
    def __init__():
        pass
    
    def train():
        pass
    def query():
        pass
15/2: import numpy
15/3: numpy.random.rand(2,3)
15/4: numpy.random.rand(3,3)
16/1:
import numpy
import scipy
17/1:
import numpy
import scipy
17/2:
import numpy
import scipy
import matplotlib.pyplot
%matplotlib inline
class neuraNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs = num.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.whi += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
17/3:
import numpy
import scipy
import matplotlib.pyplot
%matplotlib inline
class neuraNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs = num.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.whi += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
19/1:
import numpy
import scipy
import matplotlib.pyplot
%matplotlib inline
class neuraNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs = num.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.whi += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
19/2: data_file = open("mnist_train.csv","r")
19/3: data_list = data_file.readlines()
19/4: data_file.close()
19/5: all_values = data_list[0].split(',')
19/6: image_array = numpy.asfarray(all_values[1:].reshape(28,28))
20/1:
import numpy
import scipy
import matplotlib.pyplot
%matplotlib inline
class neuraNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs = num.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.whi += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
20/2: data_file = open("mnist_train.csv","r")
20/3: data_list = data_file.readlines()
20/4: data_file.close()
20/5: all_values = data_list[0].split(',')
20/6: image_array = numpy.asfarray(all_values[1:]).reshape(28,28))
21/1:
import numpy
import scipy
import matplotlib.pyplot
%matplotlib inline
class neuraNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs = num.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.whi += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
21/2: data_file = open("mnist_train.csv","r")
21/3: data_list = data_file.readlines()
21/4: data_file.close()
21/5: all_values = data_list[0].split(',')
21/6: image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
21/7: matplot.pyplot.imshow(image_array,cmap='Greys',interpolation='None')
22/1:
import numpy
import scipy
import matplotlib.pyplot
%matplotlib inline
class neuraNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs = num.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.whi += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
22/2: data_file = open("mnist_train.csv","r")
22/3: data_list = data_file.readlines()
22/4: data_file.close()
22/5: all_values = data_list[0].split(',')
22/6: image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
22/7: matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')
22/8: scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
22/9: print(scaled_input)
22/10:
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
22/11: learning_rate = 0.3
22/12: n = neuraNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
22/13:
training_data_file = open("mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()
22/14:
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    pass
23/1:
import numpy
import scipy
import matplotlib.pyplot
%matplotlib inline
class neuraNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.whi += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
23/2:
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
23/3: learning_rate = 0.3
23/4: n = neuraNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
23/5:
training_data_file = open("mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()
23/6:
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    pass
24/1:
import numpy
import scipy.special
import matplotlib.pyplot
%matplotlib inline
class neuraNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.whi += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
24/2:
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
24/3: learning_rate = 0.3
24/4: n = neuraNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
24/5:
training_data_file = open("mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()
24/6:
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    pass
25/1:
import numpy
import scipy.special
import matplotlib.pyplot
%matplotlib inline
class neuraNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.whi += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
25/2:
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
25/3: learning_rate = 0.3
25/4: n = neuraNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
25/5:
training_data_file = open("mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()
25/6:
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    pass
26/1:
import numpy
import scipy.special
import matplotlib.pyplot
%matplotlib inline
class neuraNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
26/2:
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
26/3: learning_rate = 0.3
26/4: n = neuraNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
26/5:
training_data_file = open("mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()
26/6:
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    pass
26/7: test_data_file = open("mnist_test.csv",'r')
26/8: test_data_list = test_data_file.readlines()
26/9: test_data_file.close()
26/10: all_values = test_data_list[0].split(',')
26/11: print(all_values)
26/12: print(all_values[0])
26/13: n.query((numpy.asfarray(all_values(1:) / 255.0 * 0.99) + 0.01)
26/14: n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
26/15: image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
26/16: matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')
28/1: matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')
29/1:
import numpy
import scipy.special
import matplotlib.pyplot
%matplotlib inline
class neuraNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
29/2:
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
29/3: learning_rate = 0.3
29/4: n = neuraNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
29/5:
training_data_file = open("mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()
29/6:
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    pass
29/7: test_data_file = open("mnist_test.csv",'r')
29/8: test_data_list = test_data_file.readlines()
29/9: test_data_file.close()
29/10: all_values = test_data_list[0].split(',')
29/11: print(all_values[0])
29/12: n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
29/13: image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
29/14: matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')
30/1: import numpy as np
30/2: normal_list = range(1000)
30/3: %timeit [i**2 for i in normal_list]
31/1: path = './usagov_bitly_data2013-05-17-1368828605.txt'
31/2: import json
31/3: records = [json.loads(line) for line in open(path)]
31/4: records
31/5: from pandas import DataFrame, Series
31/6: import pandas as pd
31/7: import numpy as np
31/8: frame = DataFrame(records)
31/9: frame
31/10: frame['tz']
31/11: frame['tz'][:10]
31/12: import pylab
31/13: import _tkinter
31/14: import tkinter
31/15: import Tkinter
31/16: :version
31/17: help
31/18: help()
31/19: tz_counts = frame['tz'].value_counts()
31/20: tz_counts
31/21: clean_tz = frame['tz
31/22: clean_tz = frame['tz'].fillna('Missing')
31/23: clean_tz[clean_tz == ''] = 'Unknown'
31/24: tz_counts = clean_tz.value_counts()
31/25: tz_counts[:10]
31/26: tz_counts[:10].plot(kind='barh', rot=0)
31/27: import matplotlib
   1: import pylab
   2: import json
   3: import pandas
   4: from pandas import DataFrame, Series
   5: import pandas as pd
   6: import numpy as np
   7: path = 'echo 'export PATH="/usr/local/opt/sqlite/bin:$PATH"' >> ~/.bash_profile'
   8: path = 'usagov_bitly_data2013-05-17-1368828605.txt'
   9: records = [json.loads(line) for line in open(path)]
  10: records
  11: tz_counts = frame['tz'].value_counts()
  12: frame = DataFrame(records)
  13: tz_counts = frame['tz'].value_counts()
  14: tz_counts[:10]
  15: tz_counts[:10].plot(kind='barh',rot=0)
  16: results = Series([x.split()[0] for x in frame.a.dropna()])
  17: results
  18: results[:5]
  19: results.value_counts()[:8]
  20: tz_counts[:10].plot(kind='barh',rot=0)
  21: import pylab as pl
  22: pl.show()
  23: x = range(10)
  24: y = [i**2 for i in x]
  25: pl.plot(x,y)
  26: pl.show()
  27: pl.legend()
  28: pl.show()
  29: %history -g -f usapy.py
