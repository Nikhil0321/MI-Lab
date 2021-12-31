import numpy as np
from sklearn.tree import DecisionTreeClassifier



class AdaBoost:


    def __init__(self, n_stumps=20):

        self.n_stumps = n_stumps
        self.stumps = []

    def fit(self, X, y):
        
        self.alphas = []

        sample_weights = np.ones_like(y) / len(y)
        for _ in range(self.n_stumps):

            st = DecisionTreeClassifier(
                criterion='entropy', max_depth=1, max_leaf_nodes=2)
            st.fit(X, y, sample_weights)
            y_pred = st.predict(X)

            self.stumps.append(st)

            error = self.stump_error(y, y_pred, sample_weights=sample_weights)
            alpha = self.compute_alpha(error)
            self.alphas.append(alpha)
            sample_weights = self.update_weights(
                y, y_pred, sample_weights, alpha)

        return self

    def stump_error(self, y, y_pred, sample_weights):

        Indices = np.where(y != y_pred)[0]
        error = np.sum(sample_weights[Indices]) #/ np.sum(sample_weights)
        return error

    def compute_alpha(self, error):

        eps = 1e-9
        error += eps    # For numerical stability
        alphaa = 0.5 * np.log((1 - error) / error)
        return alphaa

    def update_weights(self, y, y_pred, sample_weights, alpha):


        indeq = ( y == y_pred )
        indne = ( y != y_pred )


        sample_weights[indeq] = sample_weights[indeq] * (np.e ** (-alpha))

        sample_weights[indne] = sample_weights[indne] * (np.e ** (alpha))


        sample_weights = (sample_weights / np.sum(sample_weights))

        return sample_weights


    def predict(self, X):


        final_predds = []

        

        predds = []
        for stump in self.stumps:
            predds.append(stump.predict(X))

        
        for i in range(X.shape[0]):
            
            wt_preds = dict()

            for j in range(self.n_stumps):
                if predds[j][i] not in wt_predds:
                    wt_predds[predds[j][i]] = self.alphas[j]
                else:
                    wt_predds[predds[j][i]] += self.alphas[j]

            
            sorted_predds = sorted(wt_predds.items(), key = lambda x: x[1], reverse = True)
            final_predds.append(sorted_predds[0][0])

        return np.array(final_predds, dtype=np.int64)


    def evaluate(self, X, y):

        pred = self.predict(X)
       
        correct = (pred == y)

        accuracy = np.mean(correct) * 100  
        return accuracy
