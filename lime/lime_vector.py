"""
Functions for explaining classifiers that use Image data.
"""
import copy
from functools import partial

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from skimage.color import gray2rgb

from . import lime_base


def segmentation_func(vec):
    assert vec.size == 9000

    segments = np.zeros_like(vec)
    size = 500
    s = 0 
    for i in range(0, vec.size, size):
        segments[i:i+size] = s 
        s += 1
    
    return segments 


class VectorExplanation(object):
    def __init__(self, vector, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.vector = vector
        self.segments = segments

        assert self.vector.shape == self.segments.shape 
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None

    
    def get_vector_and_mask(self, 
                           label, 
                           positive_only=True, 
                           hide_rest=False,
                           num_features=5, 
                           min_weight=0.0):
        if label not in self.local_exp:
            raise ValueError('Label not in explanation')

        segments = self.segments
        vector = self.vector
        exp = self.local_exp[label]
        mask = np.zeros_like(segments)

        for e in exp:
            print(e)

        if hide_rest:
            temp = np.zeros_like(self.vector)
        else:
            temp = self.vector.copy()

        selected_segments_pos = []
        selected_segments_neg = []
        fs = [x[0] for x in exp if x[1] > 0][:num_features]

        for f in fs:
            indices = np.where(segments == f)[0]
            start, stop = indices.min(), indices.max()
            selected_segments_pos.append((start, stop))

        if not positive_only:
            fs = [x[0] for x in exp if x[1] <= 0]
            print(fs)
            fs = fs[:num_features]

            for f in fs:
                indices = np.where(segments == f)[0]
                start, stop = indices.min(), indices.max()
                selected_segments_neg.append((start, stop))

        return self.vector, selected_segments_pos, selected_segments_neg


class LimeVectorExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=0.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """
        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self, vector, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, 
                         num_features=100000, 
                         num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            vector: 1 dimension vector.

            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            
            labels: iterable with labels to be explained.
            
            hide_color: TODO
            
            top_labels: if not None, ignore labels and produce explanations for
                the 'top_labels' labels with highest prediction probabilities.

            num_features: maximum number of features present in explanation
            
            num_samples: size of the neighborhood to learn the linear model
            
            batch_size: TODO
            
            distance_metric: the distance metric to use for weights.
            
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            
            segm    entation_fn: SegmentationAlgorithm, wrapped skimage
                segmentation function
            
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
        assert vector.ndim == 1

        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            segmentation_fn = segmentation_func
        
        try:
            segments = segmentation_fn(vector)
        except ValueError as e:
            raise e


        fudged_vector = vector.copy()
        if hide_color is None:
            assert False 
        else:
            fudged_vector[:] = hide_color

        #return image.copy(), fudged_image
        

        # Create data set
        data, labels = self.data_labels(vector, fudged_vector, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size)
        
        top = labels
        
        #print(data.shape, data[0].reshape(1, -1))
    
        # distance original and permutations
        distances = sklearn.metrics.pairwise_distances(
            data, # data[0] orig, rest permuted
            data[0].reshape(1, -1),  # orig image
            metric=distance_metric
        ).ravel()

        ret_exp = VectorExplanation(vector, segments)


        if top_labels:
            # take 'top_labels' with highest confidence
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()

        for label in top:

            d = self.base.explain_instance_with_data(
                data, 
                labels,
                distances, 
                label, 
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection
            )

            ret_exp.intercept[label], ret_exp.local_exp[label], ret_exp.score, ret_exp.local_pred = d

        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image

            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            
            segments: segmentation of the image
            
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            
            num_samples: size of the neighborhood to learn the linear model
            
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)
        data = data.reshape((num_samples, n_features))

        labels = []
        data[0, :] = 1
        imgs = []

        for row in data:
            # temp is the orig image
            temp = copy.deepcopy(image)

            # find index of superpixels which should be disabled  
            zeros = np.where(row == 0)[0]  
            mask = np.zeros(segments.shape, dtype=bool)

            # for superpixel index in superpixels
            for z in zeros:
                mask[segments == z] = True

            # replace superpixel index we don't want in the image 
            # with a predefined value. 
            temp[mask] = fudged_image[mask]
            imgs.append(temp)

            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []

        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)

        return data, np.array(labels)
