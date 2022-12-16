import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    array1 = np.array(X)
    Xt1 = array1.T
    a = len(y)*['orange']
    for x in range(len(a)):
        if y[x]==-1: a[x]='orange'
        elif y[x]==1: a[x]='green'
    plt.scatter(Xt1[0], Xt1[1], c=a)
    plt.xlabel('symmetry')
    plt.ylabel('intensity')
    plt.savefig(r"train_features.png")

    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    x2 = -(W[0]+W[1]*X[:,0])/W[2]
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.plot(X[:, 0], x2, 'k-')
    plt.xlabel('symmetry')
    plt.ylabel('intensity')
    plt.savefig('train_result_sigmoid.png')
    ### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].

    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    class1 = -(W[0, 0] + W[1, 0] * X[:, 0]) / W[2, 0]
    class2 = -(W[0, 1] + W[1, 1] * X[:, 0]) / W[2, 1]
    class3 = -(W[0, 2] + W[1, 2] * X[:, 0]) / W[2, 2]

    plt.figure()
    a = len(y) * ['orange']
    for x in range(len(a)):
        if y[x]==2: a[x]='orange'
        elif y[x]==1: a[x]='green'
        elif y[x] == 0: a[x] = 'gray'

    plt.scatter(X[:, 0], X[:, 1], c=a, cmap='Accent')
    plt.plot(X[:, 0], class1, 'b-')
    plt.plot(X[:, 0], class2, 'r-')
    plt.plot(X[:, 0], class3, 'g-')
    plt.xlabel('symmetry')
    plt.ylabel('intensity')
    plt.ylim([-1, 1])
    plt.savefig('train_result_softmax.png')
    ### END YOUR CODE

def main():
    # ------------Data Preprocessing------------
    # Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0]

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)
    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    print("\nexploring different hyper parameters for binary logistic regression\n")
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=500)
    logisticR_classifier.fit_miniBGD(train_X, train_y, 200)
    print('\n1) miniBGD, size = 200')
    print(logisticR_classifier.get_params())
    print("training data score: " + str(logisticR_classifier.score(train_X, train_y)))
    print("validation data score: " + str(logisticR_classifier.score(valid_X, valid_y)))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 400)
    print('\n2) miniBGD, size = 400')
    print(logisticR_classifier.get_params())
    print("training data score score: " + str(logisticR_classifier.score(train_X, train_y)))
    print("validation data score: " + str(logisticR_classifier.score(valid_X, valid_y)))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 800)
    print('\n3) miniBGD, size = 800')
    print(logisticR_classifier.get_params())
    print("training data score: " + str(logisticR_classifier.score(train_X, train_y)))
    print("validation data score: " + str(logisticR_classifier.score(valid_X, valid_y)))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print('\n4) SGD')
    print(logisticR_classifier.get_params())
    print("training data score: " + str(logisticR_classifier.score(train_X, train_y)))
    print("validation data score: " + str(logisticR_classifier.score(valid_X, valid_y)))

    best_logisticR = logistic_regression(learning_rate=0.5, max_iter=500)
    best_logisticR.fit_BGD(train_X, train_y)
    print('\n5) BGD')
    print(best_logisticR.get_params())
    print("training data score: " + str(best_logisticR.score(train_X, train_y)))
    print("validation data score: " + str(best_logisticR.score(valid_X, valid_y)))
    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    ### YOUR CODE HERE

    print('checking validation accuracy for best model')
    print(best_logisticR.score(valid_X, valid_y))

    ### END YOUR CODE
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    raw_data_test, labels_test = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(raw_data_test)
    test_y_all, test_idx = prepare_y(labels_test)
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y==2)] = -1
    print('checking test accuracy for best model')
    print(best_logisticR.score(test_X, test_y))
    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k=3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=500, k=3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 500)
    print("\nexploring different hyper parameters for multiclass logistic regression\n")
    print('\n1) learning rate = 0.5, max_iter = 500, size = 500')
    print(logisticR_classifier_multiclass.get_params())
    print("training data score: " + str(logisticR_classifier_multiclass.score(train_X, train_y)))
    print("validation data score: " + str(logisticR_classifier_multiclass.score(valid_X, valid_y)))

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=500, k=3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 1000)
    print('\n2) learning rate = 0.5, max_iter = 500, size = 1000')
    print(logisticR_classifier_multiclass.get_params())
    print("training data score: " + str(logisticR_classifier_multiclass.score(train_X, train_y)))
    print("validation data score: " + str(logisticR_classifier_multiclass.score(valid_X, valid_y)))

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=500, k=3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, train_X.shape[0])
    print('\n3) learning rate = 0.5, max_iter = 500, size = '+ str(train_X.shape[0]))
    print(logisticR_classifier_multiclass.get_params())
    print("training data score: " + str(logisticR_classifier_multiclass.score(train_X, train_y)))
    print("validation data score: " + str(logisticR_classifier_multiclass.score(valid_X, valid_y)) + "\n")

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.9, max_iter=1000, k=3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, train_X.shape[0])
    print('\n4) learning rate = 0.9, max_iter = 1000, size = '+ str(train_X.shape[0]))
    print(logisticR_classifier_multiclass.get_params())
    print("training data score: " + str(logisticR_classifier_multiclass.score(train_X, train_y)))
    print("validation data score: " + str(logisticR_classifier_multiclass.score(valid_X, valid_y)))

    best_logistic_multi_R = logistic_regression_multiclass(learning_rate=0.5, max_iter=1000,  k= 3)
    best_logistic_multi_R.fit_miniBGD(train_X, train_y, train_X.shape[0])
    print('\n5) learning rate = 0.5, max_iter = 1000, size = ' + str(train_X.shape[0]))
    print(logisticR_classifier_multiclass.get_params())
    print("training data score: " + str(best_logistic_multi_R.score(train_X, train_y)))
    print("validation data score: " + str(best_logistic_multi_R.score(valid_X, valid_y)))


    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE

    raw_data_test, labels_test = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(raw_data_test)
    test_y_all, test_idx = prepare_y(labels_test)

    print('\nchecking test accuracy for best model')
    print(best_logistic_multi_R.score(test_X_all, test_y_all))

    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=10000, k=2)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 100)
    print('\nsoftmax: learning rate = 0.5, max_iter = 10000, k = 2')
    print(logisticR_classifier_multiclass.get_params())
    print("training data score: " + str(logisticR_classifier_multiclass.score(train_X, train_y)))
    print("validation data score: " + str(logisticR_classifier_multiclass.score(valid_X, valid_y)))

    ### END YOUR CODE


    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE

    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=10000)
    logisticR_classifier.fit_miniBGD(train_X, train_y, 100)
    print('\nsigmoid: learning rate = 0.5, max_iter = 10000')
    print(logisticR_classifier.get_params())
    print("training data score: " + str(logisticR_classifier.score(train_X, train_y)))
    print("validation data score: " + str(logisticR_classifier.score(valid_X, valid_y)))

    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''
    ### YOUR CODE HERE

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    train_y[np.where(train_y == 2)] = -1
    valid_y[np.where(valid_y == 2)] = -1

    for it in [1000, 2000, 3000, 4000, 5000]:

        logisticR_classifier = logistic_regression(learning_rate=0.60, max_iter=it)
        logisticR_classifier.fit_miniBGD(train_X, train_y, 1000)
        print('learning rate = 0.60, max_iter = ' + str(it) + ", weights = " + str(logisticR_classifier.get_params()))

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    train_y[np.where(train_y == 2)] = 0
    valid_y[np.where(valid_y == 2)] = 0
    print("\n")
    for it in [1000, 2000, 3000, 4000, 5000]:
        ### softmax
        logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.30, max_iter=it, k=2)
        logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 1000)

        Wv = logisticR_classifier_multiclass.get_params()
        Wd = Wv[:,1]-Wv[:,0]
        print('learning rate = 0.30, max_iter = ' + str(it) + ", w1-w2 = " + str(Wd))
    ### END YOUR CODE

    # ------------End------------
    

if __name__ == '__main__':
    main()
    
    
