# -*- coding: utf-8 -*-
from hotgrad.module import Module
from sklearn.model_selection import KFold
from hotgrad.functions.layers import Linear

""" Implementation of the Sequential module """

class Sequential(Module):
    """
        modules: list of modules that compose the network
        loss_criterion: the function that is used for computing the loss
        optimizer: the optimized used for updating the gradients
    """
    def __init__(self, modules, loss_criterion, optimizer):
        self.modules = modules
        self.loss_criterion = loss_criterion
        
        self.set_params(modules)
        optimizer.set_params(self.params())
        self.optimizer = optimizer
        
    def clear(self):
        for module in self.modules:
            if isinstance(module, Linear):
                module.clear()

        self.loss_criterion.clear()
        self.optimizer.clear()

        self.__init__(self.modules, self.loss_criterion, self.optimizer)

    """
        computes the forward pass of all the modules
    """
    def forward(self, input):        
        for module in self.modules:
            input = module.forward(input)
            
        return input
    
    def get_loss(self, predicted_value, target):
        return self.loss_criterion.forward(predicted_value, target)
    
    def fit(self, X_train, y_train, X_test=None, y_test=None, batch_size=20, epochs=25, verbose=True, log_error=False):
        compute_test_err = X_test is not None and y_test is not None
        
        if log_error:
            file = open("errors.txt", "w") 
        
        for e in range(0, epochs):
            sum_loss_train = 0
#             for b in range(0, X_train.shape[0], batch_size):
#                 output = self.forward(X_train[b : b+batch_size])
            output = self.forward(X_train)
#             print("output: ", output)
#                 loss = self.loss_criterion(output, y_train[b : b+batch_size])
            loss = self.loss_criterion(output, y_train)
#             print("loss: ", loss)
            
            sum_loss_train += loss.data[0]
            
            self.zero_grad()
            # calls all the other backward() methods
            loss.backward()
            self.optimizer.step()
                
            if verbose:
                print(
                    "Epoch " + str(e) + ": " +
                    "Train loss:", str(sum_loss_train) + ". " +
                    'Train accuracy {:0.2f}%'.format(self.score(X_train, y_train)*100) + ". " +
                    ('Test accuracy {:0.2f}%'.format(self.score(X_test, y_test)*100) if compute_test_err else ""))
            if log_error:
                file.write("Epoch " + str(e) + ": " +
                    "Train loss: " + str(sum_loss_train) + ". " +
                    'Train error {:0.2f}%'.format(self.compute_nb_errors(X_train, y_train)*100) + ". " +
                    ('Test error {:0.2f}%'.format(self.compute_nb_errors(X_test, y_test)*100) if compute_test_err else "") + "\n")

        if log_error:
            print("Last Epoch: " +
            "Train loss:", str(sum_loss_train) + ". " +
            'Train error {:0.2f}%'.format(self.compute_nb_errors(X_train, y_train)*100) + ". " +
            ('Test error {:0.2f}%'.format(self.compute_nb_errors(X_test, y_test)*100) if compute_test_err else ""))

    def predict(self, X):
        return self.forward(X).data.max(1)[1]
        
    def score(self, X, y):
        true_classes = y.data.max(1)[1] if y.data.dim() == 2 else y.data
        return (self.predict(X) == true_classes).sum() / X.shape[0]
    
    def compute_nb_errors(self, X, y):
        true_classes = y.data.max(1)[1] if y.data.dim() == 2 else y.data
        return (self.predict(X) != true_classes).sum() / X.shape[0]
    
    def cross_validate(self, X, y, n_splits=4, epochs=100, verbose=False):
        """ Run cross validation on the model and return the obtained test and train scores. """

        kf = KFold(n_splits=n_splits, random_state=1, shuffle=True)
        tr_scores = []
        va_scores = []

        result = {
            "train_score": [],
            "test_score" : []
        }

        split_n = 1
        for tr_indices, va_indices in kf.split(X):
            if verbose:
                print("----------------- fold " + str(split_n) + "/" + str(n_splits) + " -----------------")
            tr_indices = tr_indices.tolist()
            va_indices = va_indices.tolist()
            X_tr, y_tr = X[tr_indices], y[tr_indices]
            X_te, y_te = X[va_indices], y[va_indices]

            self.clear()
            self.fit(X_tr, y_tr, epochs=epochs, verbose=verbose)

            result["train_score"].append(self.score(X_tr, y_tr))
            result["test_score"].append(self.score(X_te, y_te))

            split_n = split_n + 1

        return result

    def set_params(self, modules):
        params = []
        for module in self.modules:
            for parameter in module.params():
                params.append(parameter)
                
        self.parameters = params
                
    def params(self):
        return self.parameters
    
    def zero_grad(self):
        for variable in self.params():
            variable.zero_grad()