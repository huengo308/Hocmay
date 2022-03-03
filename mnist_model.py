import mnist
from sklearn import svm
from sklearn.svm import LinearSVC
import pickle
from matplotlib import pyplot as plt

x_train = mnist.train_images()
y_train = mnist.train_labels()
x_test = mnist.test_images()
y_test = mnist.test_labels()

n_samples, nx, ny = x_train.shape
n_samples_test, nx_test, ny_test = x_test.shape
x_train = x_train.reshape((n_samples, nx*ny))
x_test_ = x_test.reshape((n_samples_test,nx_test*ny_test))


model_file = open("mnist.pickle", "rb")
svm = pickle.load(model_file)
predict = svm.predict(x_test_) 

for i in range(len(predict)):
    plt.imshow(x_test[i], cmap = plt.cm.get_cmap("binary"))
    print("Dự đoán: ", predict[i])
    plt.show()