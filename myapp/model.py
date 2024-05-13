import os
import pickle

class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.pkl')
        # your code here
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)

    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''
        # your code here
        model_path = os.path.join('myapp', 'model.pkl')
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        #pred = chr(clf.predict([x]))
        pred = clf.predict([x])
        #print(pred[0])
        return int(pred[0])
            
