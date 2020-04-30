#       3D U-Net for Volumetric Segmentation of Brain MRIs
###                      By Arda Turkmen

#       Instructions

    To train on your data change the three folder paths on top of train.py to point at directory of training images,
    directory of label images and the directory to save the checkpoints of the model state.directory. Then you can
    start the training by running the train.py script. You can also control some of the hyperparameters by passing
    arguments to python train.py. A list of parameters can be found by calling train.py -h.

    To predict an image use the predict.py script. For this script you don't need to change any variables, all of them
    are controlled through the script arguments. A list of parameters can be found by calling predict.py -h.

    eval.py defines one validation iteration during the training.

    Rest of the scripts including the model definition are in unet3d folder.

    Model is based on
    Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., & Ronneberger, O. (2016).
    3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. ArXiv, abs/1606.06650.

