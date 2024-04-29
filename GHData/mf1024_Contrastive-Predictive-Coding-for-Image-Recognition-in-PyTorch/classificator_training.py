import torch
from torch.utils.data import DataLoader
from imagenet_dataset import get_imagenet_datasets
from helper_functions import get_patch_tensor_from_image_batch, inspect_model, write_csv_stats

import os

def run_classificator(args, res_classificator_model, res_encoder_model, models_store_path):

    print("RUNNING CLASSIFICATOR TRAINING")
    dataset_train, dataset_test = get_imagenet_datasets(args.image_folder, num_classes = args.num_classes, train_split = 0.2, random_seed = 42)

    stats_csv_path = os.path.join(models_store_path, "classification_stats.csv")

    EPOCHS = 500
    LABELS_PER_CLASS = 25 # not used yet

    data_loader_train = DataLoader(dataset_train, args.sub_batch_size, shuffle = True)
    data_loader_test = DataLoader(dataset_test, args.sub_batch_size, shuffle = True)

    NUM_TRAIN_SAMPLES = dataset_train.get_number_of_samples()
    NUM_TEST_SAMPLES = dataset_test.get_number_of_samples()

    # params = list(res_classificator_model.parameters()) + list(res_encoder_model.parameters())
    optimizer_enc = torch.optim.Adam(params = res_encoder_model.parameters(), lr = 0.00001) # Train encoder slower than the classifier layers
    optimizer_cls = torch.optim.Adam(params = res_classificator_model.parameters(), lr = 0.001)

    best_epoch_test_loss = 1e10

    for epoch in range(EPOCHS):

        sub_batches_processed = 0

        epoch_train_true_positives = 0.0
        epoch_train_loss = 0.0
        epoch_train_losses = []

        batch_train_loss = 0.0
        batch_train_true_positives = 0.0

        epoch_test_true_positives = 0.0
        epoch_test_loss = 0.0

        epoch_test_losses = []

        for batch in data_loader_train:

            img_batch = batch['image'].to(args.device)

            patch_batch = get_patch_tensor_from_image_batch(img_batch)
            patches_encoded = res_encoder_model.forward(patch_batch)

            patches_encoded = patches_encoded.view(img_batch.shape[0], 7,7,-1)
            patches_encoded = patches_encoded.permute(0,3,1,2)

            classes = batch['cls'].to(args.device)

            y_one_hot = torch.zeros(img_batch.shape[0], args.num_classes).to(args.device)
            y_one_hot = y_one_hot.scatter_(1, classes.unsqueeze(dim=1), 1)

            labels = batch['class_name']

            pred = res_classificator_model.forward(patches_encoded)
            loss = torch.sum(-y_one_hot * torch.log(pred))
            epoch_train_losses.append(loss.detach().to('cpu').numpy())
            epoch_train_loss += loss.detach().to('cpu').numpy()
            batch_train_loss += loss.detach().to('cpu').numpy()

            epoch_train_true_positives += torch.sum(pred.argmax(dim=1) == classes)
            batch_train_true_positives += torch.sum(pred.argmax(dim=1) == classes)

            loss.backward()
            sub_batches_processed += img_batch.shape[0]

            if sub_batches_processed >= args.batch_size:

                optimizer_enc.step()
                optimizer_cls.step()

                optimizer_enc.zero_grad()
                optimizer_cls.zero_grad()

                sub_batches_processed = 0

                batch_train_accuracy = float(batch_train_true_positives) / float(args.batch_size)

                print(f"Training loss of batch is {batch_train_loss}")
                print(f"Accuracy of batch is {batch_train_accuracy}")

                batch_train_loss = 0.0
                batch_train_true_positives = 0.0


        epoch_train_accuracy = float(epoch_train_true_positives) / float(NUM_TRAIN_SAMPLES)

        print(f"Training loss of epoch {epoch} is {epoch_train_loss}")
        print(f"Accuracy of epoch {epoch} is {epoch_train_accuracy}")

        with torch.no_grad():

            res_classificator_model.eval()
            res_encoder_model.eval()

            for batch in data_loader_test:

                img_batch = batch['image'].to(args.device)

                patch_batch = get_patch_tensor_from_image_batch(img_batch)
                patches_encoded = res_encoder_model.forward(patch_batch)

                patches_encoded = patches_encoded.view(img_batch.shape[0], 7,7,-1)
                patches_encoded = patches_encoded.permute(0,3,1,2)

                classes = batch['cls'].to(args.device)

                y_one_hot = torch.zeros(img_batch.shape[0], args.num_classes).to(args.device)
                y_one_hot = y_one_hot.scatter_(1, classes.unsqueeze(dim=1), 1)

                labels = batch['class_name']

                pred = res_classificator_model.forward(patches_encoded)
                loss = torch.sum(-y_one_hot * torch.log(pred))
                epoch_test_losses.append(loss.detach().to('cpu').numpy())
                epoch_test_loss += loss.detach().to('cpu').numpy()

                epoch_test_true_positives += torch.sum(pred.argmax(dim=1) == classes)

            epoch_test_accuracy = float(epoch_test_true_positives) / float(NUM_TEST_SAMPLES)

            print(f"Test loss of epoch {epoch} is {epoch_test_loss}")
            print(f"Test accuracy of epoch {epoch} is {epoch_test_accuracy}")

        res_classificator_model.train()
        res_encoder_model.train()


        torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "last_res_ecoder_weights.pt"))
        torch.save(res_classificator_model.state_dict(), os.path.join(models_store_path, "last_res_classificator_weights.pt"))

        if best_epoch_test_loss > epoch_test_loss:

            best_epoch_test_loss = epoch_test_loss
            torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "best_res_ecoder_weights.pt"))
            torch.save(res_classificator_model.state_dict(), os.path.join(models_store_path, "best_res_classificator_weights.pt"))


        stats = dict(
            epoch = epoch,
            train_acc = epoch_train_accuracy,
            train_loss = epoch_train_loss,
            test_acc = epoch_test_accuracy,
            test_loss = epoch_test_loss
        )

        print(f"Writing dict {stats} into file {stats_csv_path}")
        write_csv_stats(stats_csv_path, stats)

