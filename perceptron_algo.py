# implemention of perceptron algorithm on banknote authentication dataset

#import necessary libraries
from pyforest import *

# function to implement the algorithm
def perceptron_algo(data, num_of_iter):
    features = data[:, :-1]
    labels = data[:, -1]
    
    # initialising weight matrix , +1 because w0 will be taken as 1
    w = np.zeros(shape=(1, features.shape[1]+1))
    misclassified_ = []
    
    for epoch in range(num_iter):
        misclassified = 0
        for x, label in zip(features, labels):
            x = np.insert(x,0,1)
            y = np.dot(w, x.transpose())
            
            # update the temp var target 
            target = 1.0 if (y > 0) else 0.0
            
            delta = (label.item(0,0) - target)
            
            if(delta): # if misclassified
                misclassified += 1
                w += (delta * x)
        
        misclassified_.append(misclassified)
    return (w, misclassified_)

# defining the dataset
col = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
note_metadata = pd.read_csv("data_banknote_authentication.csv", names=col)

# extracting first 1000 values and converting them into a matrix
dtset = note_metadata[:1000]
dtset_mat = np.asmatrix(dtset, dtype = 'float64')
#print(dtset_mat)

num_iter = 10
w, misclassified_ = perceptron_algo(dtset_mat, num_iter)

# plotting iternations vs misclassified graph
# if the dataset is linearly separable, the algorithm will converge at some point
epochs = np.arange(1, num_iter+1)
plt.plot(epochs, misclassified_)
plt.xlabel('iterations')
plt.ylabel('misclassified')
plt.show()


# conclusion : it turned out that this dataset isn't linearly separable
#              or we have to consider more number of iterations with variating the feature matrix 
