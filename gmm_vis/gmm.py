#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

def get_gmms( dist_1,dist_2,
             dist_3, dist_4):
    

    data_1 = np.vstack([dist_1.T, dist_2.T])
    data_2 = np.vstack([dist_3.T, dist_4.T])
    
    gmm_1 = GaussianMixture(n_components=2,
                                covariance_type='full',
                                tol=0.001,
                                reg_covar=1e-06,
                                max_iter=100,
                                n_init=1,
                                init_params='kmeans',
                                weights_init=None,
                                means_init=None,
                                precisions_init=None,
                                random_state=42,
                                warm_start=False,
                                verbose=0,
                                verbose_interval=10)

    gmm_2 = GaussianMixture(n_components=2,
                                covariance_type='full',
                                tol=0.001,
                                reg_covar=1e-06,
                                max_iter=100,
                                n_init=1,
                                init_params='kmeans',
                                weights_init=None,
                                means_init=None,
                                precisions_init=None,
                                random_state=42,
                                warm_start=False,
                                verbose=0,
                                verbose_interval=10)

    
    gmm_1.fit(data_1)
    gmm_2.fit(data_2)

    return gmm_1, gmm_2

    
    

def main():


    points_num = 250
    
    dist_1 = multivariate_normal.rvs(mean=[2,2], cov=[[1.3,0],
                                                      [0,0.5]], size=points_num).T
    
    dist_2 = multivariate_normal.rvs(mean=[-1,-1], cov=[[0.1,0],
                                                        [0,0.8]], size=points_num).T

    dist_3 = multivariate_normal.rvs(mean=[-2,-2], cov=[[1,0],
                                                        [0,2]], size=points_num).T
    dist_4 = multivariate_normal.rvs(mean=[-5,0], cov=[[0.7,0.0],
                                                       [0.0,0.4]], size=points_num).T

    dist_1 += np.random.rand(2,points_num)
    dist_2 += np.random.rand(2,points_num)
    dist_3 += np.random.rand(2,points_num)
    dist_4 += np.random.rand(2,points_num)

    gmm_1, gmm_2 = get_gmms(dist_1,dist_2,dist_3,dist_4)

    
    # grid = plt.GridSpec(7, 3)
    # plt.figure(figsize=(9,9), dpi=100)
    # plt.subplot(2,1,1)
    # plt.title(f'GMM')
    # plt.grid(b=True, which='both')

    # s = 90
    # plt.scatter(dist_1[0], dist_1[1], color='g', s=s/3, alpha=.7,label='Class 1 data')
    # plt.scatter(dist_2[0], dist_2[1], color='y', s=s/3, alpha=.4,label='Class 1 data')
    # plt.xlim(-10,10)
    # plt.ylim(-10,10)

    # plt.xlabel('x_1')
    # plt.ylabel('x_2')
    # plt.legend()
    
    # plt.subplot(2,1,2)
    # plt.grid(b=True, which='both')
    # plt.scatter(dist_3[0], dist_3[1], color='b', s=s/3, alpha=.7,label='Class 2 data')
    # plt.scatter(dist_4[0], dist_4[1], color='r', s=s/3, alpha=.7,label='Class 2 data')
    # plt.xlim(-10,10)
    # plt.ylim(-10,10)

    # plt.xlabel('x_1')
    # plt.ylabel('x_2')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()



    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)

    X, Y = np.meshgrid(x, y)

    print(X.shape)
    print(Y.shape)
    XYpairs = np.vstack([ X.reshape(-1), Y.reshape(-1) ])
    
    gmm_data = gmm_1.score_samples(XYpairs.T).reshape(100,100)
    gmm_data = gmm_2.score_samples(XYpairs.T).reshape(100,100)


if __name__ == '__main__':
    main()
