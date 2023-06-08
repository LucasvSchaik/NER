    def __init__(self, feature_function, training_sentences, algorithm="NaiveBayes", verbose=0):
        """
        Initialises and trains a tagger using the given features
         and training sentences
        :param feature_function: function that maps (untagged sentence, word index,
         history) to a dict of features (from features.py)
        :param training_sentences  : training sentences as list of
                            ((word, pos_tag), iob_tag) pairs
        :param algorithm: str:  which training algorithm to use. Default NaiveBayes.
                                Other options are IIS, GIS, and DecisionTree.
        :param verbose: int: IIS and GIS only: how much to print during training (0 = nothing)
        """

        self.train_set = []  # initialise self.train_set

        # TODO: store the feature_function parameter as self.feature_function
        self.feature_function = feature_function
        
        # TODO: call self.create_training_data on training_sentences
        self.create_training_data(training_sentences)
        
        # TODO: check that algorithm is one of "NaiveBayes", "DecisionTree", "IIS", and "GIS"
        try:
            algorithm == "NaiveBayes" or algorithm == "DecisionTree" or algorithm == "IIS" or algorithm == "GIS"
        # and raise an error if it's not
        except ImportError:
            print("algorithm is not one of NaiveBayes, DecisionTree, IIS, or GIS")
            raise
            
            
       ######################################################################
       
       
       
       def create_training_data(self, training_sentences):
        """
        Creates training data from the corpus of training_sentences and self.feature_function
        stores a list of (dict, IOB tag) pairs as self.train_set

        :param training_sentences: list of nltk.Trees with IOB tags
        :return list of (dict, IOB tag) pairs
        """
        # TODO reformat sentences to ((word, pos_tag), iob_tag) pairs
        reform = self.reformat_corpus_for_tagger(training_sentences)
                  
        # TODO turn the sentences into appropriate training data by finding their features
        t_set = []
        for n in range(len(training_sentences)):
          iobs = [iob for ((word, pos), iob) in reform[n]]
          history = tuple(iobs)
          for i in range(len(iobs)):
              feature = self.feature_function([(word, pos) for ((word, pos), iob) in reform[n]], i, history[:i])
              t_set.append((feature, iobs[i]))
                  
        # TODO store them in self.train_set
        self.train_set = t_set
