
import numpy as np
from keras.utils import to_categorical
import keras.backend as K

 
def preprocess_data(dataset, vocab, T):
    '''

    Parameters:
    -----------
    dataset : list 
    a list of tuples containing the misspeled and corrected spelling of text
    vocab : dictionary
    a dictionary containing the indexes of persian vocabulary
    T: int
    the maximum length of the query
    '''

    X, Y = zip(*dataset)
    
    X = np.array([string_to_int(i, T, vocab) for i in X])
    Y = [string_to_int(t, T, vocab) for t in Y]

    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(vocab)), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh


def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"
    
    Parameters:
    ------------
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"
    
    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """
    
    #make lower to standardize
    string = string.lower()
    string = string.replace(',','')
    string = string.replace('ي', 'ی')
    
    if len(string) > length:
        string = string[:length]
        
    rep = list(map(lambda x: vocab.get(x, vocab['<unk>']), string))
    
    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))
    
    #print (rep)
    return rep


def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')