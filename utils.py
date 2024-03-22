import numpy as np
import matplotlib.pyplot as plt

class svm_utils:
	def make_meshgrid(x, y, h=.02):
		x_min, x_max = x.min() - 1, x.max() + 1
		y_min, y_max = y.min() - 1, y.max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		                 np.arange(y_min, y_max, h))
		return xx, yy


	def plot_contours(ax, clf, xx, yy, **params):
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		out = ax.contourf(xx, yy, Z, **params)
		return out