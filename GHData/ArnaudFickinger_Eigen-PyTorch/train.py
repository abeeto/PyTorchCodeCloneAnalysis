from torch.utils.data import DataLoader
from torchvision import transforms
import os
import time
import logging
import statistics
from options import Options
from dataset import Dataset
from utils import *
from loss import *
from model import Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

opt = Options().parse()

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(opt.log_dir, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("opt = %s", opt)

def main():
    test_rmse = []
    test_logrmse = []
    test_eigen=[]
    train_rmse = []
    train_logrmse = []
    train_eigen = []
    tfms = transforms.Compose([
        ResizeImgAndDepth((opt.input_width, opt.input_height)),
        RandomHorizontalFlip(),
        ImgAndDepthToTensor()
    ])
    train_dataset = Dataset(data_dir=opt.data_dir, tfms=tfms)
    train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    test_dataset = Dataset(data_dir=opt.data_dir, tfms=tfms, train = False)
    test_dataset_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    model = Model()
    batch_, labels_ = next(iter(test_dataset_loader))
    test_samples = batch_[:3]
    test_samples_depth = labels_[:3]
    batch_, labels_ = next(iter(train_dataset_loader))
    train_samples = batch_[:3]
    train_samples_depth = labels_[:3]

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr_G)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr_G)

    total_steps = 0
    start = time.time()
    for e in range(opt.epochs):
        model.train()
        tmp_rmse = []
        tmp_logrmse = []
        tmp_eigen = []
        for batch, labels in train_dataset_loader:
            optimizer.zero_grad()

            batch = batch.to(device)
            labels = labels.to(device)

            preds = model(batch)
            eigen, rmse, logrmse = all_loss(preds, labels)

            eigen.backward()
            optimizer.step()

            total_steps += 1

            tmp_eigen.append(eigen.item())
            tmp_logrmse.append(logrmse.item())
            tmp_rmse.append(rmse.item())

            if total_steps % opt.display_every == 0:
                print ('Epoch [{}/{}], Step {}, Loss: {:.4f}'.format(e, opt.epochs, total_steps, eigen.item()))
                logging.info('Epoch [{}/{}], Step {}, Loss: {:.4f}'.format(e, opt.epochs, total_steps, eigen.item()))
        train_rmse.append(statistics.mean(tmp_rmse))
        train_logrmse.append(statistics.mean(tmp_logrmse))
        train_eigen.append(statistics.mean(tmp_eigen))
        if e%opt.checkpoint_every==0:
            save_name = os.path.join(opt.checkpoint_dir, 'eigen_{}_{}.pth'.format(opt.session, e))
            torch.save({'epoch': e+1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        },
                       save_name)
            end = time.time()
            print('save model: {}'.format(save_name))
            print('time elapsed: %fs' % (end - start))
        plot_result(test_samples, test_samples_depth, preds,
                    os.path.join(opt.results_dir, 'eigen_{}_{}_train.png'.format(opt.session, e)), test_rmse,
                    test_logrmse, test_eigen)
        model.eval()
        with torch.no_grad():
            tmp_rmse=[]
            tmp_logrmse=[]
            tmp_eigen=[]
            for batch, labels in test_dataset_loader:
                batch = batch.to(device)
                labels = labels.to(device)
                preds = model(batch)
                tmp_eigen.append(scale_invariant_loss(preds,labels).item())
                tmp_logrmse.append(logrmse_loss(preds, labels).item())
                tmp_rmse.append(rmse_loss(preds, labels).item())
            test_rmse.append(statistics.mean(tmp_rmse))
            test_logrmse.append(statistics.mean(tmp_logrmse))
            test_eigen.append(statistics.mean(tmp_eigen))
            logging.info('Epoch [{}/{}], Test_Loss: {:.4f}'.format(e, opt.epochs, total_steps, test_eigen[-1]))
            preds = []
            for i in range(3):
                sample = test_samples[i].to(device).unsqueeze(0)
                preds.append(model(sample))
            plot_result(test_samples, test_samples_depth, preds, os.path.join(opt.results_dir, 'eigen_{}_{}_test.png'.format(opt.session, e)), test_rmse, test_logrmse, test_eigen)
            # .data.cpu().numpy().transpose...





if __name__ == "__main__":
    main()