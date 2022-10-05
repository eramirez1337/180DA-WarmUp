import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from time import sleep

def find_histogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins=numLabels)

	hist = hist.astype("float")
	hist /= hist.sum()

	return hist

def plot_colors2(hist, centroids):
	bar = np.zeros((50, 300, 3), dtype="uint8")
	startX = 0

	for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
		startX = endX

    # return the bar chart
	return bar

cap = cv2.VideoCapture(0)
while(1):
	ret, frame = cap.read()
	#img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	reshapedimg = frame.reshape((frame.shape[0] * frame.shape[1],3))
	clt = KMeans(n_clusters=3)
	clt.fit(reshapedimg)
	hist = find_histogram(clt)
	bar = plot_colors2(hist, clt.cluster_centers_)
	cv2.imshow("Bar", bar)
	cv2.imshow('Real Time', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
