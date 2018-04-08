from numpy import exp, array, random, dot
import numpy


def norm_val(val , min_v, max_v):
    min_in = 0.0000
    min_out = 1.0
    x = ((float(val)-float(min_v))/(float(max_v) - float(min_v)))*(float(min_out - min_in)) + min_in
    return x

def re_norm_val(val , min_v, max_v):
    min_in = min_v
    min_out = max_v
    #x = ((float(val)-float(min_v))/(float(max_v) - float(min_v)))*(float(min_out - min_in)) + min_in
    x = float(val)*(max_v-min_v)+min_v
    return x

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(4)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((2, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)
            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output
            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    choice = 'y'
    while choice == 'y':
        print "Enter choice\n1 : Breakfast\n2 : Lunch\n3 : Hi Tea\n4 :Dinner"
        type = raw_input()
        maxrange_x=0 
        minrange_x=0
        maxrange_y=0 
        minrange_y=0
        if type =='1':
            dataset = numpy.loadtxt("data_breakfast.csv", delimiter=",")
            dataset_test = numpy.loadtxt("breakfast_test.csv", delimiter=",")
            maxrange_x=2200
            minrange_x=400
            maxrange_y=60
            minrange_y=1
        elif type == '2' :
            dataset = numpy.loadtxt("data_lunch.csv", delimiter=",")
            dataset_test = numpy.loadtxt("lunch_test.csv", delimiter=",")
            maxrange_x=2200
            minrange_x=400
            maxrange_y=140
            minrange_y=1
        elif type == '3' :
            dataset = numpy.loadtxt("data_hitea.csv", delimiter=",")
            dataset_test = numpy.loadtxt("hitea_test.csv", delimiter=",")
            maxrange_x=1800
            minrange_x=300
            maxrange_y=20
            minrange_y=0
        else :
            dataset = numpy.loadtxt("data_dinner.csv", delimiter=",")
            dataset_test = numpy.loadtxt("dinner_test.csv", delimiter=",")
            maxrange_x=2600
            minrange_x=400
            maxrange_y=170
            minrange_y=3
        #Intialise a single neuron neural network.
        neural_network = NeuralNetwork()

        print "Random starting synaptic weights:"
        print neural_network.synaptic_weights
        # split into input (X) and output (Y) variables
        X = dataset[:,0:2]
        Y = dataset[:,2]
        for i in range(0,120):
            X[i][1] = norm_val(X[i][1],minrange_x,maxrange_x)
            X[i][0] = norm_val(X[i][0],1,7)
            Y[i] = norm_val(Y[i],minrange_y,maxrange_y)

        training_set_inputs = array(X)
        training_set_outputs = array([Y]).T

        # Train the neural network using a training set.
        # Do it 10,000 times and make small adjustments each time.
        neural_network.train(training_set_inputs, training_set_outputs, 10000)

        print "New synaptic weights after training: "
        print neural_network.synaptic_weights

        testX=norm_val(3 , 1, 7)
        testY=norm_val(1000 , minrange_x, maxrange_x)
        result = neural_network.think(array([testX, testY]))
        out = re_norm_val(result,minrange_y,maxrange_y)
        print "\nFor Choice selected on Wednesday with 1000 plates, Their will be ",out," kg wasted !!!\n"
        # Test the neural network with a new situation.
        print "Calculating Accuracy using Test data"

        X_test = dataset_test[:,0:2]
        Y_test = dataset_test[:,2]
        for i in range(0,31):
            X_test[i][1] = norm_val(X_test[i][1],minrange_x,maxrange_x)
            X_test[i][0] = norm_val(X_test[i][0],1,7)
            Y_test[i] = norm_val(Y_test[i],minrange_y,maxrange_y)

        accuracy=0.0
        for i in range(0,31):
            result=neural_network.think(array([X_test[i][0], X_test[i][1]]))
            accuracy+=(Y_test[i]-abs(result-Y_test[i]))/(Y_test[i])

        accuracy/=31
        print accuracy[0]*100,"%\n"

        print "Do you want to Enter Again : (y/n)"
        choice = raw_input()

