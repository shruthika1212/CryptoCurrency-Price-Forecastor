#from sklearn.tree import DecisionTreeRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn import metrics
#from sklearn.decomposition import PCA
#decision tree
''''def result(X,y):
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0) 
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train,y_train)
    x_projection = np.array(data[['Close']])[-n_days:]
    desReg_prediction = regressor.predict(x_projection)
    des_acc = regressor.score(X_test,y_test)
    return x_projection,desReg_prediction,des_acc
n, m, p= result(X,y)'''
#knn regression
'''def result(X,y):
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0) 
    knn_model = KNeighborsRegressor(n_neighbors=3)
    knn_model.fit(X_train,y_train)
    x_projection = np.array(data[['Close']])[-n_days:]
    knn_prediction = knn_model.predict(x_projection)
    knn_acc = knn_model.score(X_test,y_test)
    return x_projection,knn_prediction,knn_acc
n, m, p= result(X,y)'''
#svr model
'''def result(X,y):
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2) 
    svrReg = SVR(kernel="rbf")
    svrReg.fit(X_train,y_train)
    x_projection = np.array(data[['Close']])[-n_days:]
    svrReg_prediction = svrReg.predict(x_projection)
    svr_acc = svrReg.score(X_test,y_test)
    return x_projection,svrReg_prediction,svr_acc'''