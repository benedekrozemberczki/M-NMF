import os
import time
import community
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import tensorflow as tf
from calculation_helper import modularity_generator, overlap_generator
from calculation_helper import graph_reader, json_dumper, log_setup, log_updater, loss_printer

class MNMF:
    """
    Modularity regularized non-negative matrix factorization machine class.
    The calculations use Tensorflow.
    """
    def __init__(self, args):
        """
        Method to parse the graph setup the similarity matrices, embedding matrices and cluster centers.
        :param args: Object with parameters.
        """
        print("Model initialization started.\n")
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            self.args = args

            self.G = graph_reader(args.input)

            self.number_of_nodes = len(nx.nodes(self.G))
            if self.number_of_nodes>10000:
                os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
            self.S_0 = tf.placeholder(tf.float64, shape=(None, None))
            self.B1 = tf.placeholder(tf.float64, shape=(None, None))
            self.B2 = tf.placeholder(tf.float64, shape=(None, None))

            self.M = tf.Variable(tf.random_uniform([self.number_of_nodes, self.args.dimensions],0,1, dtype=tf.float64))
            self.U = tf.Variable(tf.random_uniform([self.number_of_nodes, self.args.dimensions],0,1, dtype=tf.float64))
            self.H = tf.Variable(tf.random_uniform([self.number_of_nodes, self.args.clusters],0,1, dtype=tf.float64))
            self.C = tf.Variable(tf.random_uniform([self.args.clusters, self.args.dimensions],0,1, dtype=tf.float64))

            self.S = np.float64(self.args.eta)*self.S_0 + self.B1
            self.init = tf.global_variables_initializer()

    def build_graph(self):
        """
        Defining the M-NMF computation graph based on the power iteration method.
        The procedure has 4 separate phases:
        1. Updating the base matrix.
        2. Updating the embedding.
        3. Updating the cluster centers.
        4. Updating the membership of nodes.
        """
        #---------------------------------
        # 1. Phase
        #---------------------------------
        self.enum_1 = tf.matmul(self.S, self.U, a_is_sparse= True)
        self.denom_1 = tf.matmul(self.M, tf.matmul(self.U,self.U, transpose_a=True))
        self.denom_2 =  tf.maximum(np.float64(self.args.lower_control), self.denom_1)  
        self.M = self.M.assign(tf.nn.l2_normalize(tf.multiply(self.M, self.enum_1/self.denom_2), 1))
        #---------------------------------
        # 2. Phase
        #---------------------------------
        self.enum_2 = tf.matmul(self.S,self.M,transpose_a=True, a_is_sparse= True)+self.args.alpha*tf.matmul(self.H,self.C)
        self.denom_3 = tf.matmul(self.U,tf.matmul(self.M,self.M,transpose_a=True)+self.args.alpha*tf.matmul(self.C,self.C,transpose_a=True))
        self.denom_4 =  tf.maximum(np.float64(self.args.lower_control), self.denom_3) 
        self.U = self.U.assign(tf.nn.l2_normalize(np.multiply(self.U,self.enum_2/self.denom_4),1))
        #---------------------------------    
        # 3. Phase
        #---------------------------------
        self.enum_3 = tf.matmul(self.H,self.U,transpose_a=True)
        self.denom_5 = tf.matmul(self.C,tf.matmul(self.U,self.U, transpose_a=True))
        self.denom_6 =  tf.maximum(np.float64(self.args.lower_control), self.denom_5) 
        self.C = self.C.assign(tf.nn.l2_normalize(tf.multiply(self.C,self.enum_3/self.denom_6),1))
        #---------------------------------    
        # 4. Phase
        #---------------------------------
        self.B1H = tf.matmul(self.B1,self.H,a_is_sparse= True)
        self.B2H = tf.matmul(self.B2,self.H,a_is_sparse= True)
        self.HHH = tf.matmul(self.H,(tf.matmul(self.H,self.H,transpose_a=True)))
        self.UC = tf.matmul(self.U,self.C,transpose_b=True)
        self.rooted = tf.square(np.float64(2*self.args.beta)*self.B2H)+tf.multiply(np.float64(16*self.args.lambd)*self.HHH,(np.float64(2*self.args.beta)*self.B1H+np.float64(2*self.args.alpha)*self.UC +(np.float64(4*self.args.lambd-2*self.args.alpha))*self.H))
        self.sqroot_1 = tf.sqrt(self.rooted)
        self.enum_4 = np.float64(-2*self.args.beta)*self.B2H+self.sqroot_1
        self.denom_7 = np.float64(8*self.args.lambd)*self.HHH
        self.denom_8 =  tf.maximum(np.float64(self.args.lower_control), self.denom_7)
        self.sqroot_2 = tf.sqrt(self.enum_4/self.denom_8)
        self.H = self.H.assign(tf.nn.l2_normalize(tf.multiply(self.H,self.sqroot_2),1))

    def update_state(self, H):
        """
        Procedure to calculate the cluster memberships and modularity.
        :param H: Cluster membership indicator.
        :return current_modularity: Modularity based on the cluster memberships.
        """
        indices = np.argmax(H, axis=1)
        indices = {int(i): int(indices[i]) for i in range(len(indices))}
        current_modularity = community.modularity(indices,self.G)
        if current_modularity > self.best_modularity:
            self.best_modularity = current_modularity
            self.optimal_indices = indices
            self.stop_index = 0
        else:
            self.stop_index = self.stop_index + 1
        return current_modularity

    def initiate_dump(self,session, feed_dict):
        """
        Method to save the clusters, node representations, cluster memberships and logs.
        """
        json_dumper(self.optimal_indices, self.args.assignment_output)
        json_dumper(self.logs, self.args.log_output)
        loss_printer(self.logs)
        if self.args.dump_matrices:
            self.optimal_clusters = pd.DataFrame(session.run(self.C, feed_dict=feed_dict), columns = map(lambda x: "X_"+ str(x), range(self.args.dimensions)))
            self.optimal_node_representations = pd.DataFrame(session.run(self.U, feed_dict=feed_dict), columns = map(lambda x: "X_"+ str(x), range(self.args.dimensions)))
            self.optimal_clusters.to_csv(self.args.cluster_mean_output, index = None)
            self.optimal_node_representations.to_csv(self.args.embedding_output, index = None)

    def optimize(self):
        """
        Method to run the optimization and halt it when overfitting started.
        The output matrices are all saved when optimization has finished.
        """
        self.best_modularity = 0
        self.stop_index = 0
        with tf.Session(graph = self.computation_graph) as session:
            self.init.run()
            self.logs = log_setup(self.args)
            print("Optimization started.\n")
            self.build_graph()
            feed_dict = {self.S_0: overlap_generator(self.G), self.B1: np.array(nx.adjacency_matrix(self.G).todense()), self.B2:modularity_generator(self.G)}
            for i in tqdm(range(self.args.iteration_number)):
                start = time.time()
                H = session.run(self.H, feed_dict=feed_dict)
                current_modularity = self.update_state(H)
                end = time.time()
                log_updater(self.logs, i,  end-start, current_modularity)
                if self.stop_index > self.args.early_stopping:
                    break
            self.initiate_dump(session, feed_dict)
