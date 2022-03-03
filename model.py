import mnist  # pip install mnist
from sklearn.svm import LinearSVC
import pickle

#lấy tập dữ liệu
x_train = mnist.train_images()
y_train = mnist.train_labels()
x_test = mnist.test_images()
y_test = mnist.test_labels()

#Định hình bộ dữ liệu từ 3 chiều sang 2 chiều
n_samples, nx, ny = x_train.shape
n_samples_test, nx_test, ny_test = x_test.shape
x_train = x_train.reshape((n_samples, nx*ny))
x_test_ = x_test.reshape((n_samples_test,nx_test*ny_test))

#bắt đầu đào tạo và tính độ chính xác, lưu mô hình ra file
svm = LinearSVC()
for i in range(10):
    svm.fit(x_train, y_train)
    acc = svm.score(x_train, y_train)
    if acc >= 0.87:
        print("Saving...")
        model_file = open("mnist.pickle", "wb")
        pickle.dump(svm, model_file)
        print("Độ chính xác: ", acc)
        break
