import numpy as np
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve
from collections import Counter

#%%

class Kernel:
    """Base class for kernel functions."""

    def __add__(self, other):
        return KernelSum(self, other)

    def __mul__(self, scalar):
        return ScaledKernel(self, scalar)

    __rmul__ = __mul__


###############################################################################

    
class ScaledKernel(Kernel):
    """Kernel scaled by a scalar value."""

    def __init__(self, kernel, scalar):
        self.kernel = kernel
        self.scalar = scalar

    def __call__(self, X, Y=None):
        return self.scalar * self.kernel(X, Y)



class KernelSum(Kernel):
    """Sum of two kernels."""

    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def __call__(self, X, Y=None):
        return self.kernel1(X, Y) + self.kernel2(X, Y)
    
    
    def fit_precompute(self, support_vectors, support_indices):
        """ This is here for SpectrumKernel."""
        self.kernel1.fit_precompute(support_vectors, support_indices)
        self.kernel2.fit_precompute(support_vectors, support_indices)

########################################################

class LinearKernel(Kernel):
    """
    The linear kernel: k(x, y) = x.T * y
    """

    def __call__(self, X, Y=None):
        if Y is None:
            return np.dot(X, X.T)
        else:
            return np.dot(X, Y.T)


class PolynomialKernel(Kernel):
    """
    The polynomial kernel: k(x, y) = (x.T * y + c)^d
    """

    def __init__(self, degree=3, c=0):
        self.degree = degree
        self.c = c

    def __call__(self, X, Y=None):
        if Y is None:
            K = np.dot(X, X.T)
        else:
            K = np.dot(X, Y.T)
        return (K + self.c)**self.degree


class GaussianKernel(Kernel):
    """
    The Gaussian kernel: k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, X, Y=None):
        if Y is None:
            sq_dists = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, X.T) + np.sum(X**2, axis=1)
        else:
            sq_dists = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1)

        return np.exp(-sq_dists / (2 * self.sigma**2))



###########################################################################################


class SpectrumKernel(Kernel):
    """
    The Spectrum Kernel described in the report.
    """

    def __init__(self, k=6, X = None):
        self.k = k
        self.X = X
        self.X_kmer_counts = self._kmer_counts(X)
        
        vocabulary = set()
        for counts in self.X_kmer_counts:
                vocabulary.update(counts.keys())
        self.vocabulary = list(vocabulary)
        
        self.X_features = np.array([[counts.get(kmer, 0) for kmer in self.vocabulary] for counts in self.X_kmer_counts])


    def _precompute_support_vector_kmers(self, support_vectors, support_vectors_indices):
        """
        Pre-computes k-mer counts for all support vectors.
        """
        support_vector_kmer_counts = []
        for sv_seq in support_vectors:
            kmers = [sv_seq[i:i+self.k] for i in range(len(sv_seq) - self.k + 1) if len(sv_seq[i:i+self.k]) == self.k]
            support_vector_kmer_counts.append(Counter(kmers))
        self.support_vector_kmer_counts_precomputed = support_vector_kmer_counts
        self.support_vectors_data = {
            'support_vectors': support_vectors,
            'support_vectors_indices': support_vectors_indices
        }
        self.X_features = self.X_features[self.support_vectors_data['support_vectors_indices']]


    def fit_precompute(self, support_vectors, support_indices):
        """
        Pre-computation step during training to calculate and store k-mer counts for support vectors. (Artifact)
        """
        self._precompute_support_vector_kmers(support_vectors, support_indices)


    def __call__(self, X, Y = None):
        if Y is None:
            K_matrix = np.dot(self.X_features, self.X_features.T)
            diag_sqrt = np.sqrt(np.diag(K_matrix))
            diag_sqrt_outer = np.outer(diag_sqrt, diag_sqrt)
            K_matrix_normalized = K_matrix / diag_sqrt_outer
            return K_matrix_normalized
        
        else:
            if not isinstance(Y, list):
                Y = [Y]
            
            Y_kmer_counts = self._kmer_counts(Y)
            Y_features = np.array([[counts.get(kmer, 0) for kmer in self.vocabulary] for counts in Y_kmer_counts])
            K_matrix = np.dot(self.X_features, Y_features.T)
            
            diag_X_sqrt = np.sqrt(np.diag(np.dot(self.X_features, self.X_features.T)))
            diag_Y_sqrt = np.sqrt(np.diag(np.dot(Y_features, Y_features.T)))
            diag_sqrt_outer = np.outer(diag_X_sqrt, diag_Y_sqrt)
            K_matrix_normalized = K_matrix / diag_sqrt_outer
            return K_matrix_normalized


    def _kmer_counts(self, seqs):
        """
        Helper function to count k-mers in a list of sequences.
        """
        all_kmer_counts = []
        for seq in seqs:
            kmers = [seq[i:i+self.k] for i in range(len(seq) - self.k + 1) if len(seq[i:i+self.k]) == self.k]
            kmer_counts = Counter(kmers)
            all_kmer_counts.append(kmer_counts)
        return all_kmer_counts



#################################################################################


class WeightedDegreeKernel(Kernel):
    """
    Variant of Weighted Degree Kernel, as described in the report.
    """

    def __init__(self, degree=8):
        self.degree = degree
        self.weight = [1e-2, 1e-2, 1e-2, .2, .3, .5, .8, 1.3] 


    def __call__(self, X, Y = None):
        
        kernel_matrix = np.zeros((len(X), len(Y)))

        for i, seq1 in enumerate(X):
            for j, seq2 in enumerate(Y):
                kernel_value = 0
                seq_length = len(seq1)
                
                for pos in range(seq_length - self.degree + 1):
                    length = 1
                    while seq1[pos:pos+length] == seq2[pos:pos+length] and length <= self.degree - 1:
                        length += 1
                    kernel_value += self.weight[max(length - 1, 0)]

                kernel_matrix[i, j] = kernel_value

        return kernel_matrix

    
    
################################################################################################    


class MismatchSpectrumKernel(Kernel):
    """
    Mismatch Spectrum Kernel, as described in the report.
    """

    def __init__(self, k=6, m=1, X = None):

        self.k = k
        self.mismatches = m
        self.support_vector_kmer_counts_precomputed = None
        self.support_vectors_data = None
        self.X = X
        self.X_kmer_counts = self._kmer_counts(X)
        
        vocabulary = set()
        for counts in self.X_kmer_counts:
                vocabulary.update(counts.keys())
        self.vocabulary = list(vocabulary)
        
        self.X_features = self._calculate_mismatch_features(self.X_kmer_counts, vocabulary)



    def _create_vocabulary(self, X):
        vocabulary = set()
        X_kmer_counts = self._kmer_counts(X)
        for counts in X_kmer_counts:
            vocabulary.update(counts.keys())
        return list(vocabulary), X_kmer_counts


    def _precompute_support_vector_kmers(self, support_vectors, support_vectors_indices):
        support_vector_kmer_counts = []
        for sv_seq in support_vectors:
            kmers = [sv_seq[i:i+self.k] for i in range(len(sv_seq) - self.k + 1) if len(sv_seq[i:i+self.k]) == self.k]
            support_vector_kmer_counts.append(Counter(kmers))
        self.support_vector_kmer_counts_precomputed = support_vector_kmer_counts
        self.support_vectors_data = {
            'support_vectors': support_vectors,
            'support_vectors_indices': support_vectors_indices
        }
        self.X_features = self.X_features[self.support_vectors_data['support_vectors_indices']]


    def fit_precompute(self, support_vectors, support_indices):
        self._precompute_support_vector_kmers(support_vectors, support_indices)


    def __call__(self, X, Y = None, normalized = True):
        if Y is None:
            K_matrix = np.dot(self.X_features, self.X_features.T)
            diag_sqrt = np.sqrt(np.diag(K_matrix))
            diag_sqrt_outer = np.outer(diag_sqrt, diag_sqrt)
            K_matrix_normalized = K_matrix / diag_sqrt_outer
            return K_matrix_normalized


        else:
            if not isinstance(Y, list):
                Y = [Y]

            Y_kmer_counts = self._kmer_counts(Y)
            Y_features = self._calculate_mismatch_features(Y_kmer_counts, self.vocabulary) # Use mismatch features
            K_matrix_XY = np.dot(self.X_features, Y_features.T)
            
            diag_X_sqrt = np.sqrt(np.diag(np.dot(self.X_features, self.X_features.T)))
            diag_Y_sqrt = np.sqrt(np.diag(np.dot(Y_features, Y_features.T)))
            diag_sqrt_outer = np.outer(diag_X_sqrt, diag_Y_sqrt)
            K_matrix_XY_normalized = K_matrix_XY / diag_sqrt_outer
            return K_matrix_XY_normalized


    def _kmer_counts(self, seqs):
        all_kmer_counts = []
        for seq in seqs:
            kmers = [seq[i:i+self.k] for i in range(len(seq) - self.k + 1) if len(seq[i:i+self.k]) == self.k]
            kmer_counts = Counter(kmers)
            all_kmer_counts.append(kmer_counts)
        return all_kmer_counts


    def evaluate_kernel_vectors(self, test_sequences):
        kernel_vectors = np.zeros((len(test_sequences), len(self.support_vectors_data['support_vectors'])))
        seq = 0

        for test_sequence in test_sequences:
            test_kmer_counts = self._kmer_counts([test_sequence])
            test_features = self._calculate_mismatch_features(test_kmer_counts, self.vocabulary)

            for i in range(len(self.support_vectors_data['support_vectors'])):
                support_vector_features = self.X_features[np.where(self.X == self.support_vectors_data['support_vectors'][i])[0][0]]
                kernel_value = np.dot(test_features, support_vector_features.T)
                kernel_vectors[seq, i] = kernel_value

            seq += 1
        return kernel_vectors.T


    def _calculate_mismatch_features(self, X_kmer_counts, vocabulary):
        features_list = []
        for counts in X_kmer_counts:
            feature_vector = []
            for vocab_kmer in vocabulary:
                count_for_vocab_kmer = 0
                for seq_kmer, seq_count in counts.items():
                    if self._hamming_distance(vocab_kmer, seq_kmer) <= self.mismatches:
                        count_for_vocab_kmer += seq_count
                feature_vector.append(count_for_vocab_kmer)
            features_list.append(feature_vector)
        return np.array(features_list)


    def _hamming_distance(self, s1, s2):
        distance = 0
        for char1, char2 in zip(s1, s2):
            if char1 != char2:
                distance += 1
        return distance





