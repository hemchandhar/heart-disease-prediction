from sklearn.metrics import confusion_matrix

class Accuracy:

	def finding_accuracy(x_train,y_train,x_test,y_test,classifier):
		print("\nTrained successfully.......")
		# Predicting the Test set results
		y_pred = classifier.predict(x_test)
		cm_test = confusion_matrix(y_pred, y_test)
		#Report
		y_pred_train = classifier.predict(x_train)
		cm_train = confusion_matrix(y_pred_train, y_train)
		print("\nAccuracy:-\r")
		print('Accuracy for training set  = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
		print('Accuracy for test set = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))