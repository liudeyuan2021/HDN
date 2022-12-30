import numpy as np
import cv2


if __name__ == '__main__':
    train_data = np.load('examples/Pixel_Noise/Natural/data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25.npy')
    print(train_data.shape)
    a = train_data[0]
    cv2.imwrite('./1.png', a)