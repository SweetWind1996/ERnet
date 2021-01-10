import os

floder_trianimage = ''
floder_labelimage = ''
trianimage_filenames = os.listdir(floder_trianimage)
trianimage_filenames.sort(key=lambda x: int(x[:-4]))
labelimage_filenames = os.listdir(floder_labelimage)
labelimage_filenames.sort(key=lambda x: int(x[:-4]))
trianimage_num = len(trianimage_filenames)
labelimage_num = len(labelimage_filenames)


train_count = 0

f = open('train.txt', 'w')

for j in range(labelimage_num):
    for i in range(train_count, train_count + 7):
        f.write(trianimage_filenames[i]+'\t\t')
        if (i+1) % 7 == 0:
            f.write(labelimage_filenames[j]+'\n')
    train_count = train_count + 7

f.close()
