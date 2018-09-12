
"""
Source: https://github.com/dimostufts/Implementation-of-the-Kaczmarz-algorithm-in-python

We implement the Kaczmarz algorithm as well as the randomized Kaczmarz algorithm from this paper https://people.eecs.berkeley.edu/~brecht/cs294docs/week1/09.Strohmer.pdf

It is an alternative to Gradient Descent. In addition we use a novel way to evaluate convergence namely the algorithm stops, or breaks, when the error rate of the most recent iteration (to certain decimal places) is GREATER than the average of the error rates of the last 500 (or other) iterations. This means that the algorithm just passed the global minimum

"""


#We first implement the Kaczmarz algorithm for finding regression coefficients
#We then implement the randomized Kaczmarz algorithm

#the paper can be found here https://people.eecs.berkeley.edu/~brecht/cs294docs/week1/09.Strohmer.pdf

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

# the algorithm stops, or breaks, when the error rate of the most recent iteration (to certain decimal places) is GREATER than the average of the error rates of the last 500 (or other) iterations
# this means that the algorithm just passed the global minimum

# only modify N and learning rate
N = 100 # size of square matrix of features and rows
learning_rate = 1 # i would not increase the learning rate.. the algorithm stops too soon. 
most_recent_iter = 50 # use error measurements from these most recent iterations. i.e. 50 most recent errors <- from last 50 iterations
# the more we increase "most_recent_iter" the closer to convergence we are. the smaller the error
decimal_mean = 2 # when computing the mean of these most recent errors, round to that many decimal places.
# the more we increase "decimal_mean" the more iterations it consumes to stop 
A = np.random.randn(N,N)
b = np.random.randn(N,1)
x = np.random.randn(N,1)

initial_sq_error = (((b - A.dot(x))**2).sum())/N
k=0 # initialize iteration value
error_list = []
while(True):
    i = k%N + 1
    if i<N:
        ai = A[i,:][np.newaxis].T # row i converted to column vector using transpose
        bi=b[i] # element i of column vector b
        x = x + learning_rate*(((bi-ai.T.dot(x))/ai.T.dot(ai))*ai)
        error_ = (((b - A.dot(x))**2).sum())/N
        print(k,error_)
        error_list.append([k,error_])
        if k>N:
            last_numberof_errors = map(lambda x: x[1],error_list[-most_recent_iter:])
            if round((reduce(lambda x1, y1: x1 + y1, last_numberof_errors) / len(last_numberof_errors)),decimal_mean)<round(error_,decimal_mean): 
                break
    k=k+1

np_error_list = np.array(error_list)
plt.scatter(np_error_list[:,0],np_error_list[:,1]) # scatterplot showing the convergence
plt.show()

print(str(k)+" iterations")
print("Initial error: "+str(initial_sq_error))
print("Final error: "+str(error_list[-1][-1]))


##############################################

#randomized Kaczmarz

N = 100 # size of square matrix of features and rows
learning_rate = 1 # i would not increase the learning rate.. the algorithm stops too soon. 
most_recent_iter = 200 # use error measurements from these most recent iterations. i.e. 50 most recent errors <- from last 50 iterations
# the more we increase "most_recent_iter" the closer to convergence we are. the smaller the error
decimal_mean = 4 # when computing the mean of these most recent errors, round to that many decimal places.
# the more we increase "decimal_mean" the more iterations it consumes to stop 
A = np.random.randn(N,N)
b = np.random.randn(N,1)
x = np.random.randn(N,1)
x0 = np.array(x,copy=True)

initial_sq_error = (((b - A.dot(x))**2).sum())/N
k=0 # initialize iteration value
error_list = []
euclidian_list = []
# this loop creates a list where each element is the square of the euclidian norm of every row of the A matrix 
for row in range(0,N): 
    row_i = A[row,:][np.newaxis].T
    euclidian_list.append(row_i.T.dot(row_i)[0][0])

euclidian_array = np.array(euclidian_list)
prob_array = euclidian_array/euclidian_array.sum() # this converts the norm into a probability by dividing by the total sum of norms
# i.e. euclidian_array.sum is the squared Frobenius norm
while(True):
    i = np.random.choice(np.arange(N),1,p=prob_array)[0]# it selects a row number leveraging the discrete probabilities of every row
    ai = A[i,:][np.newaxis].T # row i converted to column vector using transpose
    bi=b[i] # element i of column vector b
    x = x + learning_rate*(((bi-ai.T.dot(x))/ai.T.dot(ai))*ai)
    x0=np.append(x0,x,axis=1) #this appends the new coefficients column that was just update by this iteration
    error_ = (((b - A.dot(x))**2).sum())/N
    #print(k,error_)
    error_list.append([k,error_])
    if k>N:
        last_numberof_errors = map(lambda x: x[1],error_list[-most_recent_iter:])
        if round((reduce(lambda x1, y1: x1 + y1, last_numberof_errors) / len(last_numberof_errors)),decimal_mean)<round(error_,decimal_mean): 
            break
    k=k+1

np_error_list = np.array(error_list)
plt.scatter(np_error_list[:,0],np_error_list[:,1])
plt.show()



#############################
#prove the convergence of randomized Kaczmarz
# we use equation (5) i.e. Theorem 2 from paper 

# k() is the scaled condition number by Demmel
#left side of k_() function return is Frobenius norm i.e. square root of the sum of the squares of all singular values of A
#right side is the Spectral norm of the inverse of A i.e. largest signular value
def k_(A):
    return (LA.norm(A))*(LA.norm((np.linalg.inv(A)),2))


list_of_errors = []
for col in range(0,(x0.shape[1]-2)):
    list_of_errors.append((x0[:,col:(col+1)]-x).T.dot(x0[:,col:(col+1)]-x)[0][0])

#for every step in the iteration, the left hand side should be less or equal to the right hand
for k in range(1,(x0.shape[1]-2)):
    np.array(list_of_errors)[0:k].mean()<=((1-k_(A)**(-2))**k)*((x0[:,0:1]-x).T.dot(x0[:,0:1]-x)) # key inequality that must be true
    print(k,np.array(list_of_errors)[0:k].mean(),((1-k_(A)**(-2))**k)*((x0[:,0:1]-x).T.dot(x0[:,0:1]-x)))

# the discount number appears to be very close to 1 so the right hand side does not decrease alot with iterations..
# still convergence is proven because inequality always returns True
