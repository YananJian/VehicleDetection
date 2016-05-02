import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure
from scipy import misc
from PIL import Image
from resizeimage import resizeimage
from sklearn import svm
from sklearn.cross_validation import KFold

f = open('/Users/yanan/caffe/data/vehicle/train.txt')
data = []
label = []
feature = []
for line in f:
    data.append(line.split(' ')[0])
    label.append(line.split(' ')[1].strip())

for d in data:
    image = misc.imread(d)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualise=True)
    feature.append(fd)

kf = KFold(len(data), n_folds=10, shuffle=True)
for train_index, test_index in kf:
    tmp_feature = [feature[x] for x in train_index]
    tmp_label = [label[x] for x in train_index]
    test_feature = [feature[x] for x in test_index]
    test_label = [label[x] for x in test_index]
    clf = svm.SVC()
    clf.fit(tmp_feature, tmp_label)

    correct = 0
    wrong = 0
    for i in range(0, len(test_feature)):
        if clf.predict(test_feature[i]) == test_label[i]:
            correct += 1
        else:
            wrong += 1
    acc = correct*1.0 / (correct + wrong)
    print 'correct/wrong='+str(correct)+"/"+str(wrong) + ", acc:" + str(acc)



'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
hog_image_rescaled = hog_image

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()
'''
