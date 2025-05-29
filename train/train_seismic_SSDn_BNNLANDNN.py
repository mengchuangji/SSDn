import argparse
import cv2
from torch.utils.data import DataLoader
from utils.build import build
from utils.io_ import log
from utils.option import parse, recursive_log
from validate_.validate_seis import validate_seis
import os



def main(opt):
    train_loaders = []
    for train_dataset_opt in opt['train_datasets']:
        TrainDataset = getattr(__import__('dataset'), train_dataset_opt['type'])
        train_set = build(TrainDataset, train_dataset_opt['args'])
        train_loader = DataLoader(train_set, batch_size=train_dataset_opt['batch_size'], shuffle=True,
                                  num_workers=0, drop_last=True)
        #num_workers = train_dataset_opt['batch_size']
        train_loaders.append(train_loader)

    validation_loaders = []
    for validation_dataset_opt in opt['validation_datasets']:
        ValidationDataset = getattr(__import__('dataset'), validation_dataset_opt['type'])
        validation_set = build(ValidationDataset, validation_dataset_opt['args'])
        validation_loader = DataLoader(validation_set, batch_size=1)
        validation_loaders.append(validation_loader)

    print('dataloader done!')

    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()
    if 'resume_from' in opt:
        model.load_model(opt['resume_from'])

    def train_step(data):
        model.train_step(data)

        if model.iter % opt['print_every'] == 0:
            model.log()

        if model.iter % opt['save_every'] == 0:
            model.save_net()
            model.save_model()

        if model.iter % opt['validate_every'] == 0:
            message = 'iter: %d, ' % model.iter
            for validation_loader in validation_loaders:
                # psnr = validate_sidd(model, validation_loader)
                psnr,ssim,snr = validate_seis(model, validation_loader,data_range=2) #mcj
                message += '%s: psnr: %6.2f, snr: %6.2f, ssim: %.4f' % (validation_loader.dataset.__class__.__name__, psnr,snr,ssim)
            log(opt['log_file'], message + '\n')

        if model.iter == opt['num_iters']:
            model.save_net()
            exit()

    while True:
        for data in train_loaders[0]:
            data = {'L': data['L'].cuda(), 'H': data['H'].cuda()} if 'H' in data else {'L': data['L'].cuda()}
            train_step(data)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    parser = argparse.ArgumentParser(description="Train the denoiser")
    parser.add_argument("--config", type=str, default='../option/three_stage_SASS.json')
    argspar = parser.parse_args()

    opt = parse(argspar.config)
    opt['print_every'] = 200
    opt['validate_every'] = 200
    opt['save_every'] = 10000
    opt['BNN_iters']= 10000
    opt['LAN_iters'] =10000
    opt['UNet_iters'] =10000
    opt['num_iters'] =30000
    opt['resume_from']='../train/logs/XJ/BNN_LAN_DN_29/model_iter_00020000.pth'

    opt['networks'][0]['args']['blindspot'] = 29
    opt['networks'][1]['args']['receptive_feild'] = 3
    opt['std_width'] = 7
    opt['beta_lower_upper'] = [{"lower":1, "upper":5}]  # [0.3,0.5][0.05,0.2]



    opt['log_dir'] = 'logs/XJ/BNN_LAN_DN_29_l005u02'
    opt['log_file']='logs/XJ/BNN_LAN_DN_29_l005u02/log.out'
    opt['train_datasets'][0]['args']['path']= '/home/shendi_mcj/datasets/seismic/fielddata/train/prep/SEGY_s256_o128'
    opt['validation_datasets'][0]['args']['path'] = '/home/shendi_mcj/datasets/seismic/fielddata/train/prep/SEGY_s256_o128'

    # opt['train_datasets'][0]['args']['path']= 'D:\datasets\seismic\marmousi\\f35_s256_o128'
    # opt['validation_datasets'][0]['args']['path'] = 'D:\datasets\seismic\marmousi\\f35_s256_o128'

    # opt['log_dir'] = 'logs/XJ/4S_invDN_l1u5'
    # opt['log_file']='logs/XJ/4S_invDN_l1u5/log.out'
    # opt['train_datasets'][0]['args']['path']= 'D:\datasets\seismic\XJ\SEGY_s256_o128'
    # opt['validation_datasets'][0]['args']['path'] = 'D:\datasets\seismic\XJ\SEGY_s256_o128'

    recursive_log(opt['log_file'], opt)

    main(opt)