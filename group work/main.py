# Reading the cleaned numeric car prices data
import pandas as pd
import numpy as np

# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)

CarPricesDataNumeric = pd.read_pickle('CarPricesData.pkl')
CarPricesDataNumeric.head()
# Reading the cleaned numeric car prices data
import pandas as pd
import numpy as np

# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)

CarPricesDataNumeric = pd.read_pickle('CarPricesData.pkl')
CarPricesDataNumeric.head()

# Separate Target Variable and Predictor Variables
TargetVariable = ['Price']
Predictors = ['Age', 'KM', 'Weight', 'HP', 'MetColor', 'CC', 'Doors']

X = CarPricesDataNumeric[Predictors].values
y = CarPricesDataNumeric[TargetVariable].values

### Sandardization of data ###
from sklearn.preprocessing import StandardScaler

PredictorScaler = StandardScaler()
TargetVarScaler = StandardScaler()

# Storing the fit object for later reference
PredictorScalerFit = PredictorScaler.fit(X)
TargetVarScalerFit = TargetVarScaler.fit(y)

# Generating the standardized values of X and y
X = PredictorScalerFit.transform(X)
y = TargetVarScalerFit.transform(y)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Quick sanity check with the shapes of Training and testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# Separate Target Variable and Predictor Variables
TargetVariable = ['Price']
Predictors = ['Age', 'KM', 'Weight', 'HP', 'MetColor', 'CC', 'Doors']

X = CarPricesDataNumeric[Predictors].values
y = CarPricesDataNumeric[TargetVariable].values

### Sandardization of data ###
from sklearn.preprocessing import StandardScaler

PredictorScaler = StandardScaler()
TargetVarScaler = StandardScaler()

# Storing the fit object for later reference
PredictorScalerFit = PredictorScaler.fit(X)
TargetVarScalerFit = TargetVarScaler.fit(y)

# Generating the standardized values of X and y
X = PredictorScalerFit.transform(X)
y = TargetVarScalerFit.transform(y)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Quick sanity check with the shapes of Training and testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Installing required libraries
!pip install tensorflow
!pip install keras
# Installing required libraries
!pip install tensorflow
!pip install keras

# importing the libraries
from keras.models import Sequential
from keras.layers import Dense

# create ANN model
model = Sequential()

# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=5, input_dim=7, kernel_initializer='normal', activation='relu'))

# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))

# The output neuron is a single fully connected node
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal'))

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size=20, epochs=50, verbose=1)
# importing the libraries
from keras.models import Sequential
from keras.layers import Dense

# create ANN model
model = Sequential()

# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=5, input_dim=7, kernel_initializer='normal', activation='relu'))

# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))

# The output neuron is a single fully connected node
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal'))

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size=20, epochs=50, verbose=1)


# Function to generate Deep ANN model
def make_regression_ann(Optimizer_trial):
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(units=5, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=Optimizer_trial)
    return model


###########################################
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# Listing all the parameters to try
Parameter_Trials = {'batch_size': [10, 20, 30],
                    'epochs': [10, 20],
                    'Optimizer_trial': ['adam', 'rmsprop']
                    }

# Creating the regression ANN model
RegModel = KerasRegressor(make_regression_ann, verbose=0)

###########################################
from sklearn.metrics import make_scorer


# Defining a custom function to calculate accuracy
def Accuracy_Score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig - pred) / orig))
    print('#' * 70, 'Accuracy:', 100 - MAPE)
    return (100 - MAPE)


custom_Scoring = make_scorer(Accuracy_Score, greater_is_better=True)

#########################################
# Creating the Grid search space
# See different scoring methods by using sklearn.metrics.SCORERS.keys()
grid_search = GridSearchCV(estimator=RegModel,
                           param_grid=Parameter_Trials,
                           scoring=custom_Scoring,
                           cv=5)

#########################################
# Measuring how much time it took to find the best params
import time

StartTime = time.time()

# Running Grid Search for different paramenters
grid_search.fit(X, y, verbose=1)

EndTime = time.time()
print("########## Total Time Taken: ", round((EndTime - StartTime) / 60), 'Minutes')

print('### Printing Best parameters ###')
grid_search.best_params_


# Defining a function to find the best parameters for ANN
def FunctionFindBestParams(X_train, y_train, X_test, y_test):
    # Defining the list of hyper parameters to try
    batch_size_list = [5, 10, 15, 20]
    epoch_list = [5, 10, 50, 100]

    import pandas as pd
    SearchResultsData = pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])

    # initializing the trials
    TrialNumber = 0
    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber += 1
            # create ANN model
            model = Sequential()
            # Defining the first layer of the model
            model.add(Dense(units=5, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))

            # Defining the Second layer of the model
            model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))

            # The output neuron is a single fully connected node
            # Since we will be predicting a single number
            model.add(Dense(1, kernel_initializer='normal'))

            # Compiling the model
            model.compile(loss='mean_squared_error', optimizer='adam')

            # Fitting the ANN to the Training set
            model.fit(X_train, y_train, batch_size=batch_size_trial, epochs=epochs_trial, verbose=0)

            MAPE = np.mean(100 * (np.abs(y_test - model.predict(X_test)) / y_test))

            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:', 'batch_size:', batch_size_trial, '-', 'epochs:', epochs_trial,
                  'Accuracy:', 100 - MAPE)

            SearchResultsData = SearchResultsData.append(
                pd.DataFrame(data=[[TrialNumber, str(batch_size_trial) + '-' + str(epochs_trial), 100 - MAPE]],
                             columns=['TrialNumber', 'Parameters', 'Accuracy']))
    return (SearchResultsData)


######################################################
# Calling the function
ResultsData = FunctionFindBestParams(X_train, y_train, X_test, y_test)


# Defining a function to find the best parameters for ANN
def FunctionFindBestParams(X_train, y_train, X_test, y_test):
    # Defining the list of hyper parameters to try
    batch_size_list = [5, 10, 15, 20]
    epoch_list = [5, 10, 50, 100]

    import pandas as pd
    SearchResultsData = pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])

    # initializing the trials
    TrialNumber = 0
    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber += 1
            # create ANN model
            model = Sequential()
            # Defining the first layer of the model
            model.add(Dense(units=5, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))

            # Defining the Second layer of the model
            model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))

            # The output neuron is a single fully connected node
            # Since we will be predicting a single number
            model.add(Dense(1, kernel_initializer='normal'))

            # Compiling the model
            model.compile(loss='mean_squared_error', optimizer='adam')

            # Fitting the ANN to the Training set
            model.fit(X_train, y_train, batch_size=batch_size_trial, epochs=epochs_trial, verbose=0)

            MAPE = np.mean(100 * (np.abs(y_test - model.predict(X_test)) / y_test))

            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:', 'batch_size:', batch_size_trial, '-', 'epochs:', epochs_trial,
                  'Accuracy:', 100 - MAPE)

            SearchResultsData = SearchResultsData.append(
                pd.DataFrame(data=[[TrialNumber, str(batch_size_trial) + '-' + str(epochs_trial), 100 - MAPE]],
                             columns=['TrialNumber', 'Parameters', 'Accuracy']))
    return (SearchResultsData)


######################################################
# Calling the function
ResultsData = FunctionFindBestParams(X_train, y_train, X_test, y_test)


%matplotlib inline
ResultsData.plot(x='Parameters', y='Accuracy', figsize=(15,4), kind='line')
%matplotlib inline
ResultsData.plot(x='Parameters', y='Accuracy', figsize=(15,4), kind='line')

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size=15, epochs=5, verbose=0)

# Generating Predictions on testing data
Predictions = model.predict(X_test)

# Scaling the predicted Price data back to original price scale
Predictions = TargetVarScalerFit.inverse_transform(Predictions)

# Scaling the y_test Price data back to original price scale
y_test_orig = TargetVarScalerFit.inverse_transform(y_test)

# Scaling the test data back to original scale
Test_Data = PredictorScalerFit.inverse_transform(X_test)

TestingData = pd.DataFrame(data=Test_Data, columns=Predictors)
TestingData['Price'] = y_test_orig
TestingData['PredictedPrice'] = Predictions
TestingData.head()
# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size=15, epochs=5, verbose=0)

# Generating Predictions on testing data
Predictions = model.predict(X_test)

# Scaling the predicted Price data back to original price scale
Predictions = TargetVarScalerFit.inverse_transform(Predictions)

# Scaling the y_test Price data back to original price scale
y_test_orig = TargetVarScalerFit.inverse_transform(y_test)

# Scaling the test data back to original scale
Test_Data = PredictorScalerFit.inverse_transform(X_test)

TestingData = pd.DataFrame(data=Test_Data, columns=Predictors)
TestingData['Price'] = y_test_orig
TestingData['PredictedPrice'] = Predictions
TestingData.head()

# Computing the absolute percent error
APE = 100 * (abs(TestingData['Price'] - TestingData['PredictedPrice']) / TestingData['Price'])
TestingData['APE'] = APE

print('The Accuracy of ANN model is:', 100 - np.mean(APE))
TestingData.head()
# Computing the absolute percent error
APE = 100 * (abs(TestingData['Price'] - TestingData['PredictedPrice']) / TestingData['Price'])
TestingData['APE'] = APE

print('The Accuracy of ANN model is:', 100 - np.mean(APE))
TestingData.head()


# Function to generate Deep ANN model
def make_regression_ann(Optimizer_trial):
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(units=5, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=Optimizer_trial)
    return model


###########################################
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# Listing all the parameters to try
Parameter_Trials = {'batch_size': [10, 20, 30],
                    'epochs': [10, 20],
                    'Optimizer_trial': ['adam', 'rmsprop']
                    }

# Creating the regression ANN model
RegModel = KerasRegressor(make_regression_ann, verbose=0)

###########################################
from sklearn.metrics import make_scorer


# Defining a custom function to calculate accuracy
def Accuracy_Score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig - pred) / orig))
    print('#' * 70, 'Accuracy:', 100 - MAPE)
    return (100 - MAPE)


custom_Scoring = make_scorer(Accuracy_Score, greater_is_better=True)

#########################################
# Creating the Grid search space
# See different scoring methods by using sklearn.metrics.SCORERS.keys()
grid_search = GridSearchCV(estimator=RegModel,
                           param_grid=Parameter_Trials,
                           scoring=custom_Scoring,
                           cv=5)

#########################################
# Measuring how much time it took to find the best params
import time

StartTime = time.time()

# Running Grid Search for different paramenters
grid_search.fit(X, y, verbose=1)

EndTime = time.time()
print("########## Total Time Taken: ", round((EndTime - StartTime) / 60), 'Minutes')

print('### Printing Best parameters ###')
grid_search.best_params_