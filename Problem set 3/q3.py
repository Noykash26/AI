import numpy as np
from scipy.stats import chi2
import pandas as pd
import pprint

from datetime import datetime
before = datetime.now()

class decision_tree_classifier:
    
    checked_attributes = np.empty((0,1), int)
    non_leaf = 0
    leaf = 1
    
    def __init__(self, attributes, p_value = 0.05):
        self.tree = dict()
        self.attributes = attributes
        self.get_attributes_ids = self.__get_attributes_ids
        self.p_value = p_value
        
        
    def sort_ascending(self, attribute_col, default_col):
        sorted_index = attribute_col.argsort()
        return attribute_col[sorted_index], default_col[sorted_index]    


    def entropy(self, target_col):
        elements, counts = np.unique(target_col, return_counts = True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy


    def weighted_entropy(self, x_up, y_up, x_down, y_down):
        total_samples = (len(x_up) + len(x_down))
        prob_x_up = len(x_up)/total_samples
        prob_x_down = len(x_down)/total_samples
        return(prob_x_up*self.entropy(y_up) + prob_x_down*self.entropy(y_down))


    def __get_attributes_ids(self, num_of_attributes):
        attributes_ids = np.arange(num_of_attributes)
        np.random.shuffle(attributes_ids) 
        return attributes_ids
   
    
    def split_samples(self, data, target_col, attribute_index, threshold):
        left_mask = data[:, attribute_index] > threshold
        right_mask = ~left_mask
        return data[left_mask], data[right_mask], target_col[left_mask], target_col[right_mask]
    
    
    def find_threshold(self, attribute_index, attribute_col, target_col):    
      
# =============================================================================
#         # if we already chose this attribute -> we won't choose it again
#         if attribute_index in self.checked_attributes:
#             return float('+inf'), None
# =============================================================================
                
        # if values in default pay coloumn are the same
        different_answers = np.unique(target_col)
        if len(different_answers) <= 1:
            return float('+inf'), None
        
        potential_thresholds = np.unique(attribute_col)        
        entropies = np.empty((0,1), float)
        
        for potential_threshold in potential_thresholds:
            # rows above threshold, and below it 
            index_above = np.argwhere(attribute_col > potential_threshold)
            index_below = np.setdiff1d(np.arange(len(attribute_col)), index_above)
            
            # split attribute and default columns according to threshold
            x_up, y_up = attribute_col[index_above], target_col[index_above]
            x_down, y_down = attribute_col[index_below], target_col[index_below]
            
            # calculate entropies for the potential threshold
            entropies = np.append(entropies, self.weighted_entropy(x_up, y_up, x_down, y_down))
        
        lowest_entropy_index = np.argmin(entropies)
        return entropies[lowest_entropy_index], potential_thresholds[lowest_entropy_index]
     
        
    def build_decision_tree(self, data, target_col, node_id, attribute_idx):
        # number of samples in current data on this step of reccursion
        total_samples = data.shape[0]
        # the share of each class of target attribute (T/F) in current default column
        target_values, target_counts = np.unique(target_col, return_counts = True)
        target_probs = target_counts.astype(float) / total_samples
        # most common value - True or False
        target_index = np.argmax(target_probs)
        
        # find the best attribute to split with threshold with minimum entropy
        num_of_attributes = len(attribute_idx) #data.shape[1]
        # 1st column -> entropy , 2nd column -> chosen threshold 
        all_entropy_threshold = np.zeros((data.shape[1], 2))
        for attribute_index in self.__get_attributes_ids(num_of_attributes):
            if attribute_index in attribute_idx:
                all_entropy_threshold[attribute_index,:] = self.find_threshold(attribute_index, data[:,attribute_index], target_col)
            else:
                all_entropy_threshold[attribute_index,:] = float('+inf'), None
            
        best_attribute_index = np.argmin(all_entropy_threshold[:,0]) # according to entropy

        # if entropy is "inf" then create leaf node 
        # -> this happens when the target values are the same - True or False
        if np.min(all_entropy_threshold[:,0]) == float('+inf'):
            self.tree[node_id] = (self.leaf, target_values[target_index], target_probs[target_index])
            return
        
        # Split data by attribute and threshold
        threshold = all_entropy_threshold[best_attribute_index][1]
        data_left, data_right, target_left, target_right = self.split_samples(data, target_col, best_attribute_index, threshold)
                
        # If the best split is when one half contains all of the given data then create leaf 
        if (len(target_left) == 0) or (len(target_right) == 0):
            self.tree[node_id] = (self.leaf, target_values[target_index], target_probs[target_index])
            return
    
        ######### chi squared test #########
        # right branch
        samples_count_r = data_right.shape[0]
        actual_r = np.zeros((2)) 
        classes_r, class_counts_r = np.unique(target_right, return_counts=True)
        actual_r[classes_r] = class_counts_r # an array that contains how much of T/F
        expected_r = target_probs * samples_count_r # an array that contains how much we expected of T/F
        
        # left branch
        samples_count_l = data_left.shape[0]
        actual_l = np.zeros((2))
        classes_l, class_counts_l = np.unique(target_left, return_counts=True)
        actual_l[classes_l] = class_counts_l
        expected_l = target_probs * samples_count_l
        
        chi2_l = np.sqrt((actual_l - expected_l) ** 2 / expected_l).sum()
        chi2_r = np.sqrt((actual_r - expected_r) ** 2 / expected_r).sum()
        
        # if p-value is big then prune the tree (create leaf node)
        if (1 - chi2.cdf(chi2_l + chi2_r, 1) >= self.p_value):
            self.tree[node_id] = (self.leaf, target_values[target_index], target_probs[target_index])
            return
        
        # In other cases create a non leaf node which contains feat_id and threshold
        self.tree[node_id] = (self.non_leaf, best_attribute_index, threshold, self.attributes[best_attribute_index])
        
        # Call that function for left and right parts of splitted data
        self.build_decision_tree(data_left, target_left, 2*node_id+1, np.setdiff1d(attribute_idx, best_attribute_index))
        self.build_decision_tree(data_right, target_right, 2*node_id+2, np.setdiff1d(attribute_idx, best_attribute_index))

    
    def predict(self, query, node_id):
        # leaf node array -> (1, value - T/F, prob)
        # non leaf array -> (0, best attribute index, threshold)
        node = self.tree[node_id]
        # if node is no leaf -> we will go deeper in tree
        if node[0] == self.non_leaf:
            _, attribute_index, threshold,_ = node
            # left sub tree
            if query[attribute_index] > threshold:
                return self.predict(query, 2*node_id + 1)
            # right sub tree
            else: 
                return self.predict(query, 2*node_id + 2)
        else:
            return node[1]
        
        
    def test_tree(self, data_test):
        return (np.array([self.predict(i, 0) for i in data_test]))   
 
        
def organize_data(df, k):
    data = df.values[1:, 0:-1].astype(np.float32)
    target_col = df.values[1:,-1].astype(np.int32)
    target_col = target_col.reshape(target_col.shape[0])
    total_samples = target_col.shape[0]
    
    df_names = df.values[0, :-1]
    train_count = int(total_samples * k)
    train_test_indices = np.arange(total_samples)
    x_train, x_test = data[train_test_indices[:train_count]], data[train_test_indices[train_count:]]
    y_train, y_test = target_col[train_test_indices[:train_count]], target_col[train_test_indices[train_count:]]
     
    return x_train, x_test, y_train, y_test, df_names

# build a desicion tree    
def build_tree(k: float):
    
    print("Running build-tree")
    df = pd.read_csv("DefaultOfCreditCardClients.csv", index_col=0).dropna()
    x_train, x_test, y_train, y_test, attributes = organize_data(df, k)
    my_tree = decision_tree_classifier(attributes)
   # print(np.arange(my_tree.attributes.shape[0]))
    my_tree.build_decision_tree(x_train, y_train, 0, np.arange(my_tree.attributes.shape[0]))
    
    print("My Decision Tree Accuracy: ", np.mean(my_tree.test_tree(x_test) == y_test))
    print()
    print("Read the tree as a binary tree, i.e: 2i+1 left child, 2i+2 right child:")
    print("    Non-leaf --> (0, attribute_index, threshold, attribute_name)")
    print("    Leaf --> (1, True/False == 1/0, probability of choosing T/F)")
    pprint.pprint(my_tree.tree)
    print()


# k - fold
def tree_error(k: int):
    print("Running cross-validation")
    # shuffle data
    df = pd.read_csv("DefaultOfCreditCardClients.csv", index_col=0).dropna()
    # organize data
    data = df.values[1:, 0:-1].astype(np.float32)
    target_col = df.values[1:,-1].astype(np.int32)
    target_col = target_col.reshape(target_col.shape[0])
    attributes = df.values[0, :-1]

    # number of rows in data
    total_samples = target_col.shape[0]
    
    # create an objects of tree class
    my_tree = decision_tree_classifier(attributes)

    # here we gonna store out k folds data
    data_train_folds = []
    data_test_folds = []
    target_train_folds = []
    target_test_folds = []
    
    # split the data into k parts -> loop with increament of (number of rows)/k rounded up ->
    # that is the size of each part
    data_folds = [data[i: i+total_samples//k, :] for i in range(0, total_samples, total_samples//k)]
    target_folds = [target_col[i: i+total_samples//k]for i in range(0, total_samples, total_samples//k)]
    
    for i, (data_test, target_test) in enumerate(zip(data_folds, target_folds)):
        # each iteration we will add the specific fold into our test_fold array
        data_test_folds.append(data_test)
        
        # and we will add the rest of the folds into our train_fold array
        data_train_folds.append(np.concatenate(data_folds[:i] + data_folds[i+1:]))
        
        # we will do the same to our default_pay vector
        target_test_folds.append(target_test)
        target_train_folds.append(np.concatenate(target_folds[:i] + target_folds[i+1:]))
    
    accuracy = []
    
    for data_train, data_test, target_train, target_test in zip(data_train_folds, data_test_folds, target_train_folds, target_test_folds):
        my_tree.build_decision_tree(data_train, target_train, 0, np.arange(my_tree.attributes.shape[0]))
        accuracy.append(np.mean(my_tree.test_tree(data_test) == target_test))
  
    print("Cross-validation accuracy: ", np.mean(accuracy))
    return

# determine if a person will default or not
def will_default(array):    
    print("Running will-default")
    df = pd.read_csv("DefaultOfCreditCardClients.csv", index_col=0).dropna()
    x_train, x_test, y_train, y_test, attributes = organize_data(df, 1)
    my_tree = decision_tree_classifier(attributes)
    my_tree.build_decision_tree(x_train, y_train, 0, np.arange(my_tree.attributes.shape[0]))
    answers = my_tree.test_tree(array)
    print("1 -> the person will default, and 0 -> will not ")
    print(answers)
    return answers


if __name__ == "__main__":

   build_tree(0.6)
   tree_error(k=5)
   will_default(
             [
                 [20000.0, 2.0, 2.0, 1.0, 24.0, 2.0, 2.0, -1.0, -1.0, -2.0, -2.0, 3913.0, 3102.0, 689.0, 0.0, 0.0, 0.0, 0.0, 689.0, 0.0, 0.0, 0.0, 0.0], 
                 [120000.0, 2.0, 2.0, 2.0, 26.0, -1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2682.0, 1725.0, 2682.0, 3272.0, 3455.0, 3261.0, 0.0, 1000.0, 1000.0, 1000.0, 0.0, 2000.0]
             ]
        
                )

print()    
after = datetime.now()
print('before :',before,'       after :',after) #29 minutes.