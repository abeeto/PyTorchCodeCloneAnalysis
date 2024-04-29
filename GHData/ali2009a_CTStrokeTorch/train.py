import os
import data
from model import unet_model_2d
import generator
from training import train_model, load_old_model
import argparse

config = dict()
config["pool_size"] = (2, 2)  # pool size for the max pooling operations
#config["image_shape"] = (256, 256, 64)  # This determines what shape the images will be cropped/resampled to.
config["labels"] = (1,)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["nb_channels"] = 1
config["slice_based"] = True
#iif "slice_based" in config and config["slice_based"] is True:
#    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"][:-1]))
#else:
#    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["deconvolution"] = False  # if False, will use upsampling instead of deconvolution
config["batch_size"] = 20
config["validation_batch_size"] = 20
config["n_epochs"] = 300  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["data_file"] = os.path.abspath("ich_ct_data.h5")
config["model_file"] = os.path.abspath("ich_segmentation_model.h5")
config["training_file"] = os.path.abspath("training_ids.pkl")
config["validation_file"] = os.path.abspath("validation_ids.pkl")
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
config["train_repo"] = "data/augmented/train"

def main(overwrite=False, training_repo="data/preprocessed", logging_file="training.log"):
    
    # convert input images into an hdf5 file
    if not os.path.exists(config["data_file"]):
        print("Writing the images to h5 file...")
        training_files, subject_ids = data.fetch_training_data_files(training_repo)
        data.write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"], subject_ids=subject_ids)
    data_file_opened = data.open_data_file(config["data_file"])

    print("Creating/loading the model")
    if os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        print("creaing the model")
        model = unet_model_2d(input_shape=config["input_shape"],
                              depth=4,
                              pool_size=config["pool_size"],
                              n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              batch_normalization=False)
        print("model created successfuly")

    # get training and testing generators
    print("creating the generators")
    train_generator, validation_generator, n_train_steps, n_validation_steps = generator.get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        slice_based=config["slice_based"],
        validation_batch_size=config["validation_batch_size"],
        skip_blank=config["skip_blank"])


    print("run the training...")
    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"],
                logging_file = logging_file)
    data_file_opened.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_shape", help="final image shape")
    parser.add_argument("data_file", help="absolute path to the h5 date file")
    parser.add_argument("model_file", help="absolute path to the model file")
    parser.add_argument("training_keys_file", help="absolute path to the training keys")
    parser.add_argument("validation_keys_file", help="absolute path to the validation keys")
    parser.add_argument("logging_file", help="absolute path to the logging file")

    args = parser.parse_args()
    config["training_repo"]  = "dummy"
    config["data_file"] = args.data_file
    config["model_file"] = args.model_file
    config["training_file"] = args.training_keys_file
    config["validation_file"] = args.validation_keys_file
    config["logging_file"] = args.logging_file
    config["image_shape"] = eval(args.image_shape)
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"][:-1]))
    main(overwrite=config["overwrite"], training_repo=config["training_repo"], logging_file = config["logging_file"])

