import imageio
import numpy as np
import torch
import os
from dataloader.dfs_test_dataset import DFSDataset_test
from argparse import ArgumentParser
from utils.evaluator import Evaluator
from model.network import U_Net, U_Net_MSDF, Dual_U_Net, Dual_UNet_MMFI, Dual_UNet_MSDF,Dual_UNet_MSDF_MMFI
import tifffile
from tqdm import tqdm
evaluator_1 = Evaluator(6)
evaluator_2 = Evaluator(6)
evaluator_3 = Evaluator(6)
evaluator_4 = Evaluator(6)

def get_rgb(args):
    if args.dataset == 'US3D':
        return np.array([
            [255, 0, 0],
            [204, 255, 0],
            [0, 255, 102],
            [0, 102, 255],
            [204, 0, 255],
            [255, 255, 255]])

    if args.dataset == 'ISPRS':
        return (np.array([
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
            [255, 0, 0]
        ]))


def decode_segmap(label_mask, args):
    label_colours = get_rgb(args)
    if len(label_mask.shape) == 3:
        label_mask = np.squeeze(label_mask, axis=0)
    h, w = label_mask.shape[0], label_mask.shape[1]
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, args.classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((h, w, 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb.astype(np.uint8)


def main(args):
    test_loader = torch.utils.data.DataLoader(
        DFSDataset_test(args),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    if args.model == 'UNet':
        model = U_Net(args.in_ch, args.classes)
        args.resume = '/private/Graduation_Projects/MFMDNet/train_result/US3D/UNet/'
    if args.model == 'UNet_MSFD':
        model = U_Net_MSDF(args.in_ch, args.classes)
        args.resume = '/private/Graduation_Projects/MFMDNet/train_result/US3D/UNet_MSFD/'
    if args.model == 'DualUNet':
        model = Dual_U_Net(args.in_ch, args.d_ch, args.classes)
        args.resume = '/private/Graduation_Projects/MFMDNet/train_result/US3D/DualUNet/'
    if args.model == 'DualUNet_MCEA':
        model = Dual_UNet_MMFI(args.in_ch, args.d_ch, args.classes)
        args.resume = '/private/Graduation_Projects/MFMDNet/train_result/US3D/DualUNet_MCEA/'
    if args.model == 'DualUNet_MSFD':
        model = Dual_UNet_MSDF(args.in_ch, args.d_ch, args.classes)
        args.resume = '/private/Graduation_Projects/MFMDNet/train_result/US3D/DualUNet_MSFD/'
    if args.model == 'DualUNet_MCEA_MSFD':
        model = Dual_UNet_MSDF_MMFI(args.in_ch, args.d_ch, args.classes)
        args.resume = '/private/Graduation_Projects/MFMDNet/train_result/US3D/DualUNet_MCEA_MSFD/'

    if args.gpu:
        model = model.cuda()
    model = torch.nn.DataParallel(model)
    weight_name = 'model_' + str(args.test_epoch) + '.pth'
    model_weight_file = os.path.join(args.resume, weight_name)
    if not os.path.isfile(model_weight_file):
        print('Pre-trained model file does not exist. Please check ../pretrained/decoder folder')
        exit(-1)
    else:
        print("Loading {}...".format(model_weight_file))
    model_param = torch.load(model_weight_file)
    model.load_state_dict(model_param)

    # set to evaluation mode
    model.eval()
    tbar = tqdm(test_loader, ncols=80)
    for i, (image, depth, target, image_name) in enumerate(tbar):
        # start_time = time.time()
        image = image.transpose(1, 3)
        image = image.transpose(2, 3)
        image = image.type(torch.FloatTensor)

        depth = depth.unsqueeze(3)
        depth = depth.transpose(1, 3)
        depth = depth.transpose(2, 3)
        depth = depth.type(torch.FloatTensor)
        # print(input.shape)
        if args.gpu == True:
            image = image.cuda()
            depth = depth.cuda()
            target = target.cuda()

        image_var = torch.autograd.Variable(image)
        depth_var = torch.autograd.Variable(depth)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        with torch.no_grad():
            if args.model in ['UNet' or 'UNet_MSFD']:
                output = model(image_var)
            elif args.model in ['DualUNet', 'DualUNet_MCEA', 'DualUNet_MSFD', 'DualUNet_MCEA_MSFD']:
                output = model(image_var, depth_var)

        if args.model in ['UNet', 'DualUNet', 'DualUNet_MCEA']:
            pred_1 = output.data.cpu().numpy()
            target = target_var.data.cpu().numpy()
            pred_1 = np.argmax(pred_1, axis=1)  # 0-4
            evaluator_1.add_batch(target, pred_1)
            evaluator_1.add_batch(target, pred_1)
        else:
            pred_1 = output[0].data.cpu().numpy()
            target = target_var.data.cpu().numpy()
            pred_1 = np.argmax(pred_1, axis=1)  # 0-4
            evaluator_1.add_batch(target, pred_1)

            pred_2 = output[1].data.cpu().numpy()
            pred_2 = np.argmax(pred_2, axis=1)  # 0-4
            evaluator_2.add_batch(target, pred_2)

            pred_3 = output[2].data.cpu().numpy()
            pred_3 = np.argmax(pred_3, axis=1)  # 0-4
            evaluator_3.add_batch(target, pred_3)

            pred_4 = output[3].data.cpu().numpy()
            pred_4 = np.argmax(pred_4, axis=1)  # 0-4
            evaluator_4.add_batch(target, pred_4)
        xy1_early, xy1_late = output[4].data.cpu().numpy(), output[5].data.cpu().numpy()
        xy2_early, xy2_late = output[6].data.cpu().numpy(), output[7].data.cpu().numpy()
        xy3_early, xy3_late = output[8].data.cpu().numpy(), output[9].data.cpu().numpy()
        xy4_early, xy4_late = output[10].data.cpu().numpy(), output[11].data.cpu().numpy()
        xy5_early, xy5_late = output[12].data.cpu().numpy(), output[13].data.cpu().numpy()

        if args.get_gray_output:
            gray_output_path = os.path.join(args.SaveDir, args.dataset, 'gray_pred')
            if not os.path.exists(gray_output_path):
                os.makedirs(gray_output_path)
            for i in range(pred_4.shape[0]):
                pred_name = image_name[i]
                tifffile.imwrite(os.path.join(gray_output_path, pred_name), pred_4[i])

        if args.get_mid_output:
            mid_output_path = os.path.join(args.SaveDir, args.dataset, 'mid')
            if not os.path.exists(mid_output_path):
                os.makedirs(mid_output_path)
            xy1_early_name = image_name[0].replace(".tif", "_xy1_early.tif")
            xy2_early_name = image_name[0].replace(".tif", "_xy2_early.tif")
            xy3_early_name = image_name[0].replace(".tif", "_xy3_early.tif")
            xy4_early_name = image_name[0].replace(".tif", "_xy4_early.tif")
            xy5_early_name = image_name[0].replace(".tif", "_xy5_early.tif")

            xy1_late_name = image_name[0].replace(".tif", "_xy1_late.tif")
            xy2_late_name = image_name[0].replace(".tif", "_xy2_late.tif")
            xy3_late_name = image_name[0].replace(".tif", "_xy3_late.tif")
            xy4_late_name = image_name[0].replace(".tif", "_xy4_late.tif")
            xy5_late_name = image_name[0].replace(".tif", "_xy5_late.tif")

            tifffile.imwrite(os.path.join(mid_output_path, xy1_early_name), xy1_early)
            tifffile.imwrite(os.path.join(mid_output_path, xy2_early_name), xy2_early)
            tifffile.imwrite(os.path.join(mid_output_path, xy3_early_name), xy3_early)
            tifffile.imwrite(os.path.join(mid_output_path, xy4_early_name), xy4_early)
            tifffile.imwrite(os.path.join(mid_output_path, xy5_early_name), xy5_early)

            tifffile.imwrite(os.path.join(mid_output_path, xy1_late_name), xy1_late)
            tifffile.imwrite(os.path.join(mid_output_path, xy2_late_name), xy2_late)
            tifffile.imwrite(os.path.join(mid_output_path, xy3_late_name), xy3_late)
            tifffile.imwrite(os.path.join(mid_output_path, xy4_late_name), xy4_late)
            tifffile.imwrite(os.path.join(mid_output_path, xy5_late_name), xy5_late)

        if args.get_rgb_output:
            rgb_output_path = os.path.join(args.SaveDir, args.dataset, 'rgb_pred')
            if not os.path.exists(rgb_output_path):
                os.makedirs(rgb_output_path)
            # print the result of the model with RGB mode
            # for i in range(pred_1.shape[0]):
            #     pred_name = image_name[i].replace(".tif", "_1.png")
            #     image = decode_segmap(pred_4[i], args)
            #     tifffile.imwrite(os.path.join(rgb_output_path, pred_name), image)
            # for i in range(pred_2.shape[0]):
            #     pred_name = image_name[i].replace(".tif", "_2.png")
            #     image = decode_segmap(pred_4[i], args)
            #     tifffile.imwrite(os.path.join(rgb_output_path, pred_name), image)
            # for i in range(pred_3.shape[0]):
            #     pred_name = image_name[i].replace(".tif", "_3.png")
            #     image = decode_segmap(pred_4[i], args)
            #     tifffile.imwrite(os.path.join(rgb_output_path, pred_name), image)
            for i in range(pred_4.shape[0]):
                pred_name = image_name[i].replace(".tif", "_4.png")
                image = decode_segmap(pred_4[i], args)
                tifffile.imwrite(os.path.join(rgb_output_path, pred_name), image)

    if args.model in ['UNet', 'DualUNet', 'DualUNet_MCEA']:
        Acc_1 = evaluator_1.Pixel_Accuracy()
        Acc_class_1 = evaluator_1.Pixel_Accuracy_Class()
        mIoU_1 = evaluator_1.Mean_Intersection_over_Union()
        FWIoU_1 = evaluator_1.Frequency_Weighted_Intersection_over_Union()
        kappa_1 = evaluator_1.Kappa()
        print(
            'Acc_1: %.4f, Acc_class_1: %.4f, mIoU_1: %.4f, FWIoU_1: %.4f, Kappa_1: %.4f ' %
            (Acc_1, Acc_class_1, mIoU_1, FWIoU_1, kappa_1))

    else:
        Acc_1 = evaluator_1.Pixel_Accuracy()
        Acc_class_1 = evaluator_1.Pixel_Accuracy_Class()
        mIoU_1 = evaluator_1.Mean_Intersection_over_Union()
        FWIoU_1 = evaluator_1.Frequency_Weighted_Intersection_over_Union()
        kappa_1 = evaluator_1.Kappa()

        Acc_2 = evaluator_2.Pixel_Accuracy()
        Acc_class_2 = evaluator_2.Pixel_Accuracy_Class()
        mIoU_2 = evaluator_2.Mean_Intersection_over_Union()
        FWIoU_2 = evaluator_2.Frequency_Weighted_Intersection_over_Union()
        kappa_2 = evaluator_2.Kappa()

        Acc_3 = evaluator_3.Pixel_Accuracy()
        Acc_class_3 = evaluator_3.Pixel_Accuracy_Class()
        mIoU_3 = evaluator_3.Mean_Intersection_over_Union()
        FWIoU_3 = evaluator_3.Frequency_Weighted_Intersection_over_Union()
        kappa_3 = evaluator_3.Kappa()

        Acc_4 = evaluator_4.Pixel_Accuracy()
        Acc_class_4 = evaluator_4.Pixel_Accuracy_Class()
        mIoU_4 = evaluator_4.Mean_Intersection_over_Union()
        FWIoU_4 = evaluator_4.Frequency_Weighted_Intersection_over_Union()
        kappa_4 = evaluator_4.Kappa()
        if args.get_confusion_matrix:
            confusion_matrix_path = os.path.join(args.SaveDir, args.dataset, 'confusion_matrix')
            if not os.path.exists(confusion_matrix_path):
                os.makedirs(confusion_matrix_path)
            confusion_matrix_1 = evaluator_1.get_confusion_matrix()
            imageio.imwrite(os.path.join(confusion_matrix_path, 'confusion_matrix_1.tif'), confusion_matrix_1)
            confusion_matrix_2 = evaluator_2.get_confusion_matrix()
            imageio.imwrite(os.path.join(confusion_matrix_path, 'confusion_matrix_2.tif'), confusion_matrix_2)
            confusion_matrix_3 = evaluator_3.get_confusion_matrix()
            imageio.imwrite(os.path.join(confusion_matrix_path, 'confusion_matrix_3.tif'), confusion_matrix_3)
            confusion_matrix_4 = evaluator_4.get_confusion_matrix()
            imageio.imwrite(os.path.join(confusion_matrix_path, 'confusion_matrix_4.tif'), confusion_matrix_4)
        print(
            'Acc_1: %.4f, Acc_class_1: %.4f, mIoU_1: %.4f, FWIoU_1: %.4f, Kappa_1: %.4f ' %
            (Acc_1, Acc_class_1, mIoU_1, FWIoU_1, kappa_1))
        print(
            'Acc_2: %.4f, Acc_class_2: %.4f, mIoU_2: %.4f, FWIoU_2: %.4f, Kappa_2: %.4f ' %
            (Acc_2, Acc_class_2, mIoU_2, FWIoU_2, kappa_2))
        print(
            'Acc_3: %.4f, Acc_class_3: %.4f, mIoU_3: %.4f, FWIoU_3: %.4f, Kappa_3: %.4f  ' %
            (Acc_3, Acc_class_3, mIoU_3, FWIoU_3, kappa_3))
        print(
            'Acc_4: %.4f, Acc_class_4: %.4f, mIoU_4: %.4f, FWIoU_4: %.4f, Kappa_4: %.4f  ' %
            (Acc_4, Acc_class_4, mIoU_4, FWIoU_4, kappa_4))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default='DualUNet_MSFD', type=str)
    parser.add_argument('--in_ch', default=3, type=int)
    parser.add_argument('--d_ch', default=1, type=int)
    parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--dataset', default='US3D', type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--classes', default=5, type=int)
    parser.add_argument('--test_epoch', default='best', type=str)
    parser.add_argument('--SaveDir', default='../output')
    parser.add_argument('--get_rgb_output', type=bool, default=False)
    parser.add_argument('--get_mid_output', type=bool, default=False)
    parser.add_argument('--get_gray_output', type=bool, default=False)
    parser.add_argument('--get_confusion_matrix', type=bool, default=False)

    args = parser.parse_args()
    main(args)
