if __name__ == "__main__":
    # Hierarchical DivNoising - Prediction
    # This notebook contains an example on how to use a previously trained Hierarchical DivNoising model to denoise images.
    # If you haven't done so please first run '1-Training.ipynb', which will train the model. 

    # We import all our dependencies.
    import numpy as np
    import torch
    import sys
    sys.path.append('../../../')
    import lib.utils as utils
    from boilerplate import boilerplate
    from matplotlib import pyplot as plt

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load GT test data

    test_images_gt = np.load('data/BSD68_reproducibility_data/test/bsd68_groundtruth.npy', allow_pickle=True)

    # Load noisy test data

    test_images = np.load('data/BSD68_reproducibility_data/test/bsd68_gaussian25.npy', allow_pickle=True)

    # Load our model

    model = torch.load("./Trained_model/model/natural_last_vae.net")
    model.mode_pred=True
    model.eval()

    # Compute PSNR
    # The higher the PSNR, the better the denoing performance is.
    # PSNR is computed using the formula: 

    # ```PSNR = 20 * log(rangePSNR) - 10 * log(mse)``` <br> 
    # where ```mse = mean((gt - img)**2)```, gt is ground truth image and img is the prediction from HDN. All logarithms are with base 10.<br>
    # rangePSNR = 255 for natural images as used in this [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf)

    gaussian_noise_std = 25
    num_samples = 100 # number of samples used to compute MMSE estimate
    tta = False # turn on test time augmentation when set to True. It may improve performance at the expense of 8x longer prediction time
    psnrs = []
    range_psnr = 255
    for i in range(test_images.shape[0]):
        img_mmse, samples = boilerplate.predict(test_images[i][:-1,:-1],num_samples,model,gaussian_noise_std,device,tta)
        psnr = utils.PSNR(test_images_gt[i][:-1,:-1], img_mmse, range_psnr)
        psnrs.append(psnr)
        print("image:", i, "PSNR:", psnr, "Mean PSNR:", np.mean(psnrs))
    
    # Here we look at some qualitative solutions
    fig=plt.figure(figsize=(20, 10))
    gt = test_images_gt[-1]
    vmin=np.percentile(gt,0)
    vmax=np.percentile(gt,99)


    columns = 5
    rows = 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(test_images[-1],cmap='magma')
    plt.title("Raw")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(gt,vmin=vmin, vmax=vmax,cmap='magma')
    plt.title("GT")
    fig.add_subplot(rows, columns, 3)
    plt.imshow(img_mmse,vmin=vmin, vmax=vmax,cmap='magma')
    plt.title("MMSE")
    for i in range(4, columns*rows+1):
        img = samples[i]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img,vmin=vmin, vmax=vmax,cmap='magma')
        plt.title("Sample "+str(i-4))
    plt.show()