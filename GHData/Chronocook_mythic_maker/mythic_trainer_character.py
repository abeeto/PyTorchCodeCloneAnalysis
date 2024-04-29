"""
This script trains a character level RNN (LSTM) model

Usage:
python mythic_trainer_character.py --text_file ./DATA/foo.txt --print_every 100 --epochs 2000


@author: Brad Beechler (brad.e.beechler@gmail.com)
# Last Modification: 09/20/2017 (Brad Beechler)
"""

from uplog import log
import os
import time
import random
from tqdm import tqdm
import torch
import mythic_common as common
import mythic_writer_character as writer
import mythic_model_character as model

# This is hamfisted but these setting are used so much it's convienient
settings = common.TrainerSettings()


def random_training_set():
    inp = model.torch.LongTensor(settings.batch_size, settings.chunk_size)
    target = model.torch.LongTensor(settings.batch_size, settings.chunk_size)
    for bi in range(settings.batch_size):
        start_index = random.randint(0, settings.text_length - settings.chunk_size - 1)
        end_index = start_index + settings.chunk_size + 1
        chunk = settings.text_string[start_index:end_index]
        inp[bi] = common.char_tensor(chunk[:-1])
        target[bi] = common.char_tensor(chunk[1:])
    inp = model.Variable(inp)
    target = model.Variable(target)
    if settings.cuda is not None:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


# def ordered_training_set(start_index):
#     if start_index > (settings.text_length - settings.chunk_size):
#         log.out.warning("Requested index would blow bounds in text array, setting to random.")
#         start_index = random.randint(0, settings.text_length - settings.chunk_size)
#     inp = model.torch.LongTensor(settings.batch_size, settings.chunk_size)
#     target = model.torch.LongTensor(settings.batch_size, settings.chunk_size)
#     for bi in range(settings.batch_size):
#         # start_index = random.randint(0, settings.text_length - settings.chunk_size)
#         end_index = start_index + settings.chunk_size + 1
#         chunk = settings.text_string[start_index:end_index]
#         inp[bi] = common.char_tensor(chunk[:-1])
#         target[bi] = common.char_tensor(chunk[1:])
#     inp = model.Variable(inp)
#     target = model.Variable(target)
#     if settings.cuda is not None:
#         inp = inp.cuda()
#         target = target.cuda()
#     return inp, target, end_index


def train(input_pattern, target):
    hidden = net.init_hidden(settings.batch_size, cuda=settings.cuda)
    net.zero_grad()
    this_loss = 0
    for c in range(settings.chunk_size):
        output, hidden = net(input_pattern[:, c], hidden)
        this_loss += criterion(output.view(settings.batch_size, -1), target[:, c])
    this_loss.backward()
    net_optimizer.step()
    return this_loss.data[0] / settings.chunk_size


def save(save_filename):
    model.torch.save(net, save_filename)
    log.out.info("Saved as:" + save_filename)


if __name__ == '__main__':
    """
    Main driver.
    """
    # Set the logging level to normal and start run
    start_time = time.time()
    common.report_sys_info()
    if settings.debug:
        log.out.setLevel('DEBUG')
    else:
        log.out.setLevel('INFO')
    if settings.cuda is not None:
        log.out.info("Using CUDA on device: " + str(settings.cuda))
        torch.cuda.set_device(settings.cuda)
    else:
        log.out.info("Using CPU")
    if settings.model_file is None:
        settings.model_file = os.path.splitext(os.path.basename(settings.text_file))[0] + '.pt'

    settings.report()

    log.out.info("Read text data from: " + settings.text_file)
    log.out.info("Found " + str(settings.text_length) + " characters.")

    # Initialize the net
    net = model.CharRNN(
        common.num_characters,
        settings.hidden_size,
        common.num_characters,
        model=settings.model,
        n_layers=settings.layers,
        dropout=settings.dropout,
        cuda=settings.cuda
    )
    # Set the optimizer
    current_learning_rate = settings.learning_rate
    net_optimizer = model.torch.optim.Adam(net.parameters(), lr=current_learning_rate)
    # Set the loss function (criterion)
    criterion = model.nn.CrossEntropyLoss()

    sample_prediction_size = 2 * settings.chunk_size
    all_losses = []
    # Hardcode loss inspection parameters, because hardcode is hardcore
    loss_inspection_window = 100
    loss_drop_percent_threshold = 0.75
    loss_drop_grace_iterations = 5  # Will give the model n windows before dropping the learning rate again
    # Initialize running variables
    loss_accum = 0
    loss_drop_grace_fails = 0
    loss_average_last = None
    try:
        for epoch in tqdm(range(1, settings.epochs + 1)):
            loss = train(*random_training_set())
            loss_accum += loss
            if epoch % loss_inspection_window == 0:
                all_losses.append(round(loss, 4))
                loss_average_current = loss_accum / settings.print_every
                # If your average loss hasn't dropped much half the learning rate
                if loss_average_last is not None:
                    loss_drop_percent = 100.0 * (loss_average_last - loss_average_current) / loss_average_last
                    log.out.info("Loss drop percentage over window: " + str(loss_drop_percent))
                    if loss_drop_percent < loss_drop_percent_threshold:
                        loss_drop_grace_fails += 1
                        if loss_drop_grace_fails >= loss_drop_grace_iterations:
                            current_learning_rate = current_learning_rate / 2.0
                            loss_drop_percent_threshold = loss_drop_percent_threshold / 2.0
                            loss_drop_grace_fails = 0
                            log.out.info("Learning rate hasn't dropped much in  " + str(loss_drop_grace_iterations) +
                                         " time windows.")
                            log.out.info("Halving the learning rate to: " + str(current_learning_rate))
                            log.out.info("Halving the loss drop threshold to: " + str(loss_drop_percent_threshold))
                            net_optimizer = model.torch.optim.Adam(net.parameters(), lr=current_learning_rate)
                loss_accum = 0
                loss_average_last = loss_average_current

            if epoch % settings.print_every == 0:
                # Report progress
                percent_done = epoch / settings.epochs * 100
                log.out.info('[%s (%d%%) %.4f]' % (common.time_since(start_time), percent_done, loss))
                log.out.info("\n" + writer.generate(net, 'Wh', sample_prediction_size, cuda=settings.cuda))
        log.out.info("Loss history:" + "\n" + ",".join(map(str, all_losses)))
        log.out.info("Saving model.")
        save(settings.model_file)

    except KeyboardInterrupt:
        log.out.info("Saving model before quit.")
        save(settings.model_file)

    # Shut down and clean up
    total_time = round((time.time() - start_time), 0)
    log.out.info("Execution time: " + str(total_time) + " sec")
    log.out.info("All Done!")
    log.stopLog()
