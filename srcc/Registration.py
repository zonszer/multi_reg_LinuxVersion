from __future__ import print_function
import time
import gc

import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf       #tf2 changed
# tf.disable_v2_behavior()
from VGG16 import VGG16mo
from utilss.utilss import *
import cv2
from lap import lapjv
from utilss.shape_context import ShapeContext
import matplotlib.pyplot as plt

class CNN(object):
    def __init__(self): 
        self.height = 224       
        self.width = 224            #每一个图片patch的大小为224*224
        self.shape = np.array([224.0, 224.0])           # shape

        self.sift_weight = 2.0
        self.cnn_weight = 1.0

        self.max_itr = 200                                          #？

        self.tolerance = 1e-2
        self.freq = 5                                   # k in the paper？？
        self.epsilon = 0.5
        self.omega = 0.5
        self.beta = 2.0
        self.lambd = 0.5                            #初始化一个patch中的各类参数

        self.cnnph = tf.placeholder("float", [2, 224, 224, 3])      #tf.placeholder( dtype, shape=【? , patch_length, patch_wid, bgr】, name )  我们每次可以将 一个minibatch传入到x = tf.placeholder(tf.float32,[None,32])上，进行op
        self.vgg = VGG16mo()
        self.vgg.build(self.cnnph)          #[1]    初始化卷积层的各个参数（load）
        self.SC = ShapeContext()

    def register(self, IX, IY):         #[2]    进行2图片配准

        # set parameters
        tolerance = self.tolerance
        freq = self.freq
        epsilon = self.epsilon
        omega = self.omega
        beta = self.beta
        lambd = self.lambd

        # resize image（图像分割为patch）
        Xscale = 1.0 * np.array(IX.shape[:2]) / self.shape          #为将原图可分为224*224大小的patch的个数
        Yscale = 1.0 * np.array(IY.shape[:2]) / self.shape
        IX = cv2.resize(IX, (self.height, self.width))      #IX shape = (224, 224, 3)
        IY = cv2.resize(IY, (self.height, self.width))

        # CNN feature
        # propagate the images through VGG16
        IX = np.expand_dims(IX, axis=0)     #IX shape = (1， 224, 224, 3)
        IY = np.expand_dims(IY, axis=0)
        cnn_input = np.concatenate((IX, IY), axis=0)    #shape = (2， 224, 224, 3)
        with tf.Session() as sess:
            feed_dict = {self.cnnph: cnn_input}     #self.cnnph是一个tf.placeholder。   feed_dict相当与用于input
            D1, D2, D3 = sess.run([
                self.vgg.pool3, self.vgg.pool4, self.vgg.pool5_1        #sess内容为：3个层进行计算，input = cnn_input（ref图和reg图的合并tensor）
            ], feed_dict=feed_dict)

        # flatten 看到这里了
        DX1, DY1 = np.reshape(D1[0], [-1, 256]), np.reshape(D1[1], [-1, 256])
        DX2, DY2 = np.reshape(D2[0], [-1, 512]), np.reshape(D2[1], [-1, 512])
        DX3, DY3 = np.reshape(D3[0], [-1, 512]), np.reshape(D3[1], [-1, 512])

        # normalization
        DX1, DY1 = DX1 / np.std(DX1), DY1 / np.std(DY1)
        DX2, DY2 = DX2 / np.std(DX2), DY2 / np.std(DY2)
        DX3, DY3 = DX3 / np.std(DX3), DY3 / np.std(DY3)

        del D1, D2, D3

        # compute feature space distance
        PD1 = pairwise_distance(DX1, DY1)
        PD2 = pd_expand(pairwise_distance(DX2, DY2), 2)
        PD3 = pd_expand(pairwise_distance(DX3, DY3), 4)
        PD = 1.414 * PD1 + PD2 + PD3

        del DX1, DY1, DX2, DY2, DX3, DY3, PD1, PD2, PD3

        seq = np.array([[i, j] for i in range(28) for j in range(28)], dtype='int32')

        X = np.array(seq, dtype='float32') * 8.0 + 4.0
        Y = np.array(seq, dtype='float32') * 8.0 + 4.0

        # normalize
        X = (X - 112.0) / 224.0
        Y = (Y - 112.0) / 224.0

        # prematch and select points
        C_all, quality = match(PD)
        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 128: tau_max -= 0.01

        C = C_all[np.where(quality >= tau_max)]
        cnt = C.shape[0]

        # select prematched feature points
        X, Y = X[C[:, 1]], Y[C[:, 0]]
        PD = PD[np.repeat(np.reshape(C[:, 1], [cnt, 1]), cnt, axis=1),
                np.repeat(np.reshape(C[:, 0], [1, cnt]), cnt, axis=0)]

        N = X.shape[0]
        M = X.shape[0]
        assert M == N

        # precalculation of feature match
        C_all, quality = match(PD)

        # compute \hat{\theta} and \delta
        tau_min = np.min(quality)
        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 0.5 * cnt: tau_max -= 0.01
        tau = tau_max
        delta = (tau_max - tau_min) / 10.0

        SCX = self.SC.compute(X)

        # initialization
        Z = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)

        Pr = None

        Q = 0
        dQ = float('Inf')
        itr = 1

        # registration process
        while itr < self.max_itr and abs(dQ) > tolerance and sigma2 > 1e-4:
            Z_old = Z.copy()
            Q_old = Q

            # for every k iterations
            if (itr - 1) % freq == 0:
                # compute C^{conv}_{\theta}
                C = C_all[np.where(quality >= tau)]
                Lt = PD[C[:, 0], C[:, 1]]
                maxLt = np.max(Lt)
                if maxLt > 0: Lt = Lt / maxLt
                L = np.ones([M, N])
                L[C[:, 0], C[:, 1]] = Lt

                # compute C^{geo}_{\theta}
                SCZ = self.SC.compute(Z)
                SC_cost = self.SC.cost(SCZ, SCX)

                # compute C
                L = L * SC_cost

                # linear assignment
                C = lapjv(L)[1]

                # prior probability matrix
                Pr = np.ones_like(PD) * (1.0 - epsilon) / N
                Pr[np.arange(C.shape[0]), C] = 1.0
                Pr = Pr / np.sum(Pr, axis=0)

                tau = tau - delta
                if tau < tau_min: tau = tau_min

            # compute minimization
            Po, P1, Np, tmp, Q = compute(X, Y, Z_old, Pr, sigma2, omega)
            Q = Q + lambd / 2 * np.trace(np.dot(np.dot(A.transpose(), GRB), A))

            # update variables
            dP = np.diag(P1)
            t1 = np.dot(dP, GRB) + lambd * sigma2 * np.eye(M)
            t2 = np.dot(Po, X) - np.dot(dP, Y)
            A = np.dot(np.linalg.inv(t1), t2)
            sigma2 = tmp / (2.0 * Np)
            omega = 1 - (Np / N)
            if omega > 0.99: omega = 0.99
            if omega < 0.01: omega = 0.01
            Z = Y + np.dot(GRB, A)
            lambd = lambd * 0.95
            if lambd < 0.1: lambd = 0.1

            dQ = Q - Q_old
            itr = itr + 1

        print('finish: itr %d, Q %d, tau %d' % (itr, Q, tau))
        return ((X*224.0)+112.0)*Xscale, ((Y*224.0)+112.0)*Yscale, ((Z*224.0)+112.0)*Xscale