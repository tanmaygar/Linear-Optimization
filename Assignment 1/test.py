import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn import metrics
import cvxpy as cp
import os
import sys
import time
from scipy.linalg import null_space
class simplexAlgorithm():
    def __init__(self, A, b, c, z):
        self.A = np.array(A)
        self.b = np.array(b)
        self.c = np.array(c)
        self.m = len(b)
        self.n = len(c)
        self.z = np.array(z)
    
    def find_tight_rows(self, z):
        round_Az = np.round(self.A @ z, 2)
        round_b = np.round(self.b, 2)
        # val = np.where(np.isclose(self.A @ z, self.b))[0]
        val = np.where(np.isclose(round_Az, round_b))[0]
        return val
    
    def compute_direction(self, A_tight):
        A_null_space = null_space(A_tight)
        if A_null_space.shape[1] > 0:
            direction = A_null_space[:, 0]
        else:
            direction = np.zeros(self.n)
        return direction
    
    def solve(self, lr=0.1, iterations=1000):
        # Check Feasibility
        if not np.all(self.A @ self.z <= self.b):
            print("z is not feasible")
            return self.z
        
        print("Calculating Vertex")
        # Reach vertex
        while len(self.find_tight_rows(self.z)) < self.n:
            tight_rows = self.find_tight_rows(self.z)
            A_tight = self.A[tight_rows]
            direction = self.compute_direction(A_tight)
            new_z = self.z + lr * direction
            if len(self.find_tight_rows(new_z)) >= len(self.find_tight_rows(self.z)):
                self.z = new_z
            print("Z: ", self.z, " Cost: ", self.z@self.c)
        
        self.z = self.z.round(2)
        
        print("Reached Vertex")
        # Reach Optimum
        # beta = lr
        for iter_num in range(iterations):
            tight_rows = self.find_tight_rows(self.z)
            A_tight = self.A[tight_rows]
            A_inv = np.linalg.pinv(A_tight)
            # print(A_inv)
            alpha = A_inv @ self.c
            alpha = np.round(alpha, 2)
            idx = np.where(alpha < 0)[0]
            if idx.size == 0:
                print("Optimal Solution: ", self.z)
                return self.z
            
            beta = lr
            while True:
                # print("Here")
                new_z = self.z - beta * A_inv[idx[0]]
                beta += lr
                # print(len(self.find_tight_rows(new_z)), self.n)
                if len(self.find_tight_rows(new_z)) == self.n:
                    self.z = new_z
                    break
            print("Z: ", self.z, " Cost: ", self.z @ self.c)
            # return self.z
            
            # print("Alpha shape: ", alpha.shape, alpha)
            # if np.all(alpha >= 0):
            #     print("Optimal Solution Found")
            #     return self.z
            # for idx in range(len(alpha)):
            #     if alpha[idx] < 0:
            #         break
            # new_z = ().round(2)
        
            # if len(self.find_tight_rows(new_z)) == self.n:
            #     self.z = new_z
            # else:
            #     beta += lr
            # print("Z: ", self.z, " Cost: ", self.z@self.c)
            
        return self.z
                    
            
            # direction = self.compute_direction(A_tight)
            # if np.all(direction == 0):
            #     print("z is not feasible")
            #     return self.z
            # else:
            #     # Find the index of the most negative direction
            #     idx = np.argmin(direction)
            #     # Find the index of the most negative z
            #     z_idx = np.argmin(self.z)
            #     # Find the index of the most negative direction
            #     ratio = self.z[z_idx] / direction[idx]
            #     # Update z
            #     self.z = self.z - ratio * direction
            #     self.z[z_idx] = ratio

# c = np.array([3, 2], dtype=np.float32)
# z = np.array([0, 0], dtype=np.float32)
# b = np.array([100, 80, 30, 0, 0], dtype=np.float32)
# A = np.array([[2, 1], [1, 1], [1, 0], [-1, 0], [0, -1]], dtype=np.float32)


c = np.array([4, 1], dtype=np.float32)
z = np.array([0, 50], dtype=np.float32)
b = np.array([50, 90, 0, 0], dtype=np.float32)
A = np.array([[1, 1], [3, 1], [-1, 0], [0, -1]], dtype=np.float32)

# print rank of A
# print("Rank of A: ", np.linalg.matrix_rank(A))
# print(A.shape)

# perform simplex algorithm  and get the optimal solution and objective value
simplex = simplexAlgorithm(A, b, c, z)
optimal_solution = simplex.solve()
objective_value = c @ optimal_solution

print("Optimal solution: ", optimal_solution)
print("Objective value: ", objective_value)

        