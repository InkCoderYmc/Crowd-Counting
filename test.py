'''
Original souce code: https://github.com/ZhihengCV/Bayesian-Crowd-Counting
'''
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm as CM
from matplotlib import pyplot as plt
import torch
import os
import time
import numpy as np
from datasets.crowd import Crowd
import argparse
from models import M_SFANet_UCF_QNRF
from models import M_SegNet_UCF_QNRF
from PIL import Image
import cv2


args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='UCF',
                        help='training data directory')
    parser.add_argument('--save-dir', default='weights',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    #add cuda disable
    #parser.add_argument('--disable-cuda', action='store_ture', help='Disable CUDA')
    args = parser.parse_args()
    return args


def resize(density_map, image):
    density_map = 255*density_map/np.max(density_map)
    density_map = density_map[0][0]
    image = image[0]
    print(density_map.shape)
    result_img = np.zeros((density_map.shape[0]*2, density_map.shape[1]*2))
    for i in range(result_img.shape[0]):
        for j in range(result_img.shape[1]):
            result_img[i][j] = density_map[int(i / 2)][int(j / 2)] / 4
    result_img = result_img.astype(np.uint8, copy=False)
    return result_img


def vis_densitymap(o, den, cc, img_path):
    fig = plt.figure()
    columns = 2
    rows = 1
#     X = np.transpose(o, (1, 2, 0))
    X = o
    summ = int(np.sum(den))

    den = resize(den, o)

    for i in range(1, columns*rows + 1):
        # image plot
        if i == 1:
            img = X
            fig.add_subplot(rows, columns, i)
            plt.gca().set_axis_off()
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1,
                                left=0, hspace=0, wspace=0)
            plt.imshow(img)

        # Density plot
        if i == 2:
            img = den
            fig.add_subplot(rows, columns, i)
            plt.gca().set_axis_off()
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1,
                                left=0, hspace=0, wspace=0)
            plt.text(1, 80, 'M-SegNet* Est: '+str(summ),
                     fontsize=7, weight="bold", color='w')
            plt.imshow(img, cmap=CM.jet)

    filename = img_path.split('/')[-1]
    filename = filename.replace('.jpg', '_heatpmap.png')
    print('Save at', filename)
    plt.savefig(filename, transparent=True,
                bbox_inches='tight', pad_inches=0.0, dpi=200)


def vis_den(o, den, img_path):
    fig = plt.figure()
    columns = 1
    rows = 1
#     X = np.transpose(o, (1, 2, 0))
    X = o
    summ = int(np.sum(den))

    den = resize(den, o)

    for i in range(1, columns*rows + 1):
        # Density plot
        if i == 1:
            img = den
            fig.add_subplot(rows, columns, i)
            plt.gca().set_axis_off()
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1,
                                left=0, hspace=0, wspace=0)
            plt.text(1, 80, 'Crowd Count Number:'+str(summ),
                     fontsize=7, weight="bold", color='w')
            plt.imshow(img, cmap=CM.jet)

    filename = img_path.split('/')[-1]
    filename = filename.replace('.jpg', '_heatpmap.png')
    filename = 'images/'+filename
    print('Save at', filename)
    plt.savefig(filename, transparent=True,
                bbox_inches='tight', pad_inches=0.0, dpi=200)



if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'val'),
                     512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=0, pin_memory=False)
    model = M_SegNet_UCF_QNRF.Model()
    #device = torch.device('cpu')
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(
        args.save_dir, 'best_M-SegNet__UCF_QNRF.pth'), device))
    epoch_minus = []
    preds = []
    gts = []
    model.eval()
    print(len(dataloader))
    sum_est = 0

    time_start = time.time()
    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            preds.append(torch.sum(outputs).item())
            gts.append(count[0].item())
            #print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            sum_est += int(np.sum(outputs.cpu().detach().numpy()))
            print('Est: {}'.format(int(np.sum(outputs.cpu().detach().numpy()))))
            epoch_minus.append(temp_minu)

    
    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)

    met = []
    for i in range(len(preds)):
        met.append(100 * np.abs(preds[i] - gts[i]) / gts[i])

    idxs = []
    for k in range(100):
        idxs.append(k)
    for i in range(len(met)):
        idxs.append(np.argmin(met))
        if len(idxs) == 5:
            break
        met[np.argmin(met)] += 100000000

        processed_dir = os.path.join(args.data_dir, 'val')
        model.eval()
        c = 0

    # print(idxs)

    for inputs, count, name in dataloader:
        img_path = os.path.join(processed_dir, name[0]) + '.jpg'
        if c in set(idxs):
            inputs = inputs.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)

                img = Image.open(img_path).convert('RGB')
                height, width = img.size[1], img.size[0]
                height = round(height / 16) * 16
                width = round(width / 16) * 16
                img = cv2.resize(
                    np.array(img), (width, height), cv2.INTER_CUBIC)

                print('Do VIS')
                '''
                vis_densitymap(img, outputs.cpu().detach().numpy(),
                               int(count.item()), img_path)
                '''
                vis_den(img, outputs.cpu().detach().numpy(), img_path)
                c += 1

        else:
            c += 1

    time_end = time.time()
    print('Sum of Est: {}, totally cost: {}'.format(
        sum_est, time_end-time_start))