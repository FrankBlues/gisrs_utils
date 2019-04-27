# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:47:10 2018
Show.
@author: Administrator
"""

# import numpy as np
# import geopandas as gpd

# from scipy import ndimage

# import matplotlib.pylab as pylab
# import matplotlib.pyplot as plt

# pylab.rcParams['figure.figsize'] = 8, 6
# pts = gpd.GeoDataFrame.from_file('points_demo.shp')
# pts.plot()

# def heatmap(d, bins=(100, 100), smoothing=1.3, cmap='jet'):
#     def getx(pt):
#         return pt.coords[0][0]

#     def gety(pt):
#         return pt.coords[0][1]

#     x = list(d.geometry.apply(getx))
#     y = list(d.geometry.apply(gety))
#     heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)
#     extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]

#     logheatmap = np.log(heatmap)
#     logheatmap[np.isneginf(logheatmap)] = 0
#     logheatmap = ndimage.filters.gaussian_filter(logheatmap,
#                                                  smoothing, mode='nearest')

#     plt.imshow(logheatmap, cmap=cmap, extent=extent)
#     plt.colorbar()
#     plt.gca().invert_yaxis()
#     plt.show()


# if __name__ == '__main__':
#     imgfile = r'D:\test18.TIF'