import argparse
import time
from tasks.semanticSegmentator_manager import SemanticSegmentation_Manager
from tasks.classification_manager import Classification_Manager
from config.configuration import Configuration
from loss.loss_builder import Loss_Builder
from models.model_builder import Model_builder
from utils.optimizer_builder import Optimizer_builder
from scheduler.scheduler_builder import scheduler_builder
from utils.logger import Logger
from dataloader.dataloader_builder import Dataloader_Builder

def main():
    start_time = time.time()
    # Input arguments
    parser = argparse.ArgumentParser(description="TensorFlow framework for Semantic Segmentation")
    parser.add_argument("--config_file",
                        type=str,
                        default='config/configFile.py',
                        help="configuration file path")

    parser.add_argument("--exp_name",
                        type=str,
                        default='Sample',
                        help="Experiment name")

    parser.add_argument("--exp_folder",
                        type=str,
                        default='/home/jlgomez/Experiments/',
                        help="Experiment folder path")

    args = parser.parse_args()

    # Prepare configutation
    print ('Loading configuration ...')
    config = Configuration(args.config_file, args.exp_name, args.exp_folder)
    cf = config.Load()
    # Enable log file
    logger_debug = Logger(cf.log_file_debug)

    logger_debug.write('\n ---------- Init experiment: ' + cf.exp_name + ' ---------- \n')

    # Model building
    logger_debug.write('- Building model: ' + cf.model_name + ' <--- ')
    model = Model_builder(cf)
    model.build()

    # Problem type
    if cf.problem_type == 'segmentation':
        problem_manager = SemanticSegmentation_Manager(cf, model)
    elif cf.problem_type == 'classification':
        problem_manager = Classification_Manager(cf, model)
    else:
        raise ValueError('Unknown problem type')

    # Loss definition
    criterion = Loss_Builder(cf).build().cuda()

    # Optimizer definition
    optimizer = Optimizer_builder().build(cf, model.net)

    # Learning rate scheduler
    scheduler = scheduler_builder().build(cf, optimizer)

    # Create dataloader builder
    dataloader = Dataloader_Builder(cf)

    if cf.train:
        model.net.train()  # enable dropout modules and others
        train_time = time.time()
        # Dataloaders
        logger_debug.write('\n- Reading Train dataset: ')
        dataloader.build_train()
        if cf.valid_images_txt is not None and cf.valid_gt_txt is not None and cf.valid_samples_epoch != 0:
            logger_debug.write('\n- Reading Validation dataset: ')
            dataloader.build_valid(cf.valid_samples_epoch, cf.valid_images_txt, cf.valid_gt_txt,
                                   cf.resize_image_valid, cf.valid_batch_size)
            problem_manager.trainer.start(criterion, optimizer, dataloader.train_loader, dataloader.train_set,
                                          dataloader.loader_set, dataloader.loader, scheduler)
        else:
            # Train without validation inside epoch
            problem_manager.trainer.start(criterion, optimizer, dataloader.train_loader, dataloader.train_set,
                                          scheduler=scheduler)
        train_time = time.time() - train_time
        logger_debug.write('\t Train step finished: %ds ' % (train_time))

    if cf.validation:
        valid_time = time.time()
        model.net.eval()
        if not cf.train:
            logger_debug.write('- Reading Validation dataset: ')
            dataloader.build_valid(cf.valid_samples,cf.valid_images_txt, cf.valid_gt_txt,
                                   cf.resize_image_valid, cf.valid_batch_size)
        else:
            # If the Dataloader for validation was used on train, only update the total number of images to take
            dataloader.loader_set.update_indexes(cf.valid_samples, valid=True) #valid=True avoids shuffle for validation
        logger_debug.write('\n- Starting validation <---')
        problem_manager.validator.start(criterion, dataloader.loader_set, dataloader.loader)
        valid_time = time.time() - valid_time
        logger_debug.write('\t Validation step finished: %ds ' % (valid_time))

    if cf.test:
        model.net.eval()
        test_time = time.time()
        logger_debug.write('\n- Reading Test dataset: ')
        dataloader.build_valid(cf.test_samples, cf.test_images_txt, cf.test_gt_txt,
                               cf.resize_image_test, cf.test_batch_size)
        logger_debug.write('\n - Starting test <---')
        problem_manager.validator.start(criterion, dataloader.loader_set, dataloader.loader)
        test_time = time.time() - test_time
        logger_debug.write('\t Test step finished: %ds ' % (test_time))

    if cf.predict_test:
        model.net.eval()
        pred_time = time.time()
        logger_debug.write('\n- Reading Prediction dataset: ')
        dataloader.build_predict()
        logger_debug.write('\n - Generating predictions <---')
        problem_manager.predictor.start(dataloader.predict_loader)
        pred_time = time.time() - pred_time
        logger_debug.write('\t Prediction step finished: %ds ' % (pred_time))

    total_time = time.time() - start_time
    logger_debug.write('\n- Experiment finished: %ds ' % (total_time))
    logger_debug.write('\n')

# Entry point of the script
if __name__ == "__main__":
    main()