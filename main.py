from utils.summaries import TensorboardSummary
from model.network import Dual_UNet_MSDF_MMFI
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
import torch.optim.lr_scheduler
from dataloader.dataset import Dataset_US3D, Dataset_ISPRS_Vaihingen, Dataset_ISPRS_Potsdam
import loss_functions.lovasz_loss as L
from utils.evaluator import Evaluator
import warnings
warnings.filterwarnings('ignore')
import os
from tqdm import tqdm
from utils.saver import Saver
from apex import amp

def val(args, val_loader, model, epoch):
    model.eval()
    epoch_loss = []
    val_loss = 0.0
    tbar = tqdm(val_loader, ncols=80)

    for i, (image, depth, target) in enumerate(tbar):
        
        image = image.transpose(1, 3)
        image = image.transpose(2, 3)
        image = image.type(torch.FloatTensor)

        depth = depth.unsqueeze(3)
        depth = depth.transpose(1, 3)
        depth = depth.transpose(2, 3)
        depth = depth.type(torch.FloatTensor)
        
        if args.onGPU == True:
            image = image.cuda()
            depth = depth.cuda()
            target = target.cuda()

        image_var = torch.autograd.Variable(image)
        depth_var = torch.autograd.Variable(depth)
        target_var = torch.autograd.Variable(target)

        
        output = model(image_var, depth_var)
        
        loss1 = L.lovasz_softmax(output[0], target_var)
        loss2 = L.lovasz_softmax(output[1], target_var)
        loss3 = L.lovasz_softmax(output[2], target_var)
        loss4 = L.lovasz_softmax(output[3], target_var)
        loss = loss1 + loss2 + loss3 + loss4
        epoch_loss.append(loss.item())
        

        # compute the confusion matrix
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

        # print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.item(), time_taken))
        val_loss += loss.item()
        tbar.set_description('Validing loss: %.3f' % (val_loss / (i + 1)))

    summary.visualize_image(writer, target, pred_1, pred_2, pred_3, pred_4, epoch)
    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    Acc_1 = evaluator_1.Pixel_Accuracy()
    Acc_class_1 = evaluator_1.Pixel_Accuracy_Class()
    mIoU_1 = evaluator_1.Mean_Intersection_over_Union()
    FWIoU_1 = evaluator_1.Frequency_Weighted_Intersection_over_Union()

    Acc_2 = evaluator_2.Pixel_Accuracy()
    Acc_class_2 = evaluator_2.Pixel_Accuracy_Class()
    mIoU_2 = evaluator_2.Mean_Intersection_over_Union()
    FWIoU_2 = evaluator_2.Frequency_Weighted_Intersection_over_Union()

    Acc_3 = evaluator_3.Pixel_Accuracy()
    Acc_class_3 = evaluator_3.Pixel_Accuracy_Class()
    mIoU_3 = evaluator_3.Mean_Intersection_over_Union()
    FWIoU_3 = evaluator_3.Frequency_Weighted_Intersection_over_Union()

    Acc_4 = evaluator_4.Pixel_Accuracy()
    Acc_class_4 = evaluator_4.Pixel_Accuracy_Class()
    mIoU_4 = evaluator_4.Mean_Intersection_over_Union()
    FWIoU_4 = evaluator_4.Frequency_Weighted_Intersection_over_Union()

    writer.add_scalar('valid/loss_epoch', average_epoch_loss_val, epoch)
    writer.add_scalar('valid/Acc_1', Acc_1, epoch)
    writer.add_scalar('valid/Acc_2', Acc_2, epoch)
    writer.add_scalar('valid/Acc_3', Acc_3, epoch)
    writer.add_scalar('valid/Acc_4', Acc_4, epoch)
    writer.add_scalar('valid/Acc_class_1', Acc_class_1, epoch)
    writer.add_scalar('valid/Acc_class_2', Acc_class_2, epoch)
    writer.add_scalar('valid/Acc_class_3', Acc_class_3, epoch)
    writer.add_scalar('valid/Acc_class_4', Acc_class_4, epoch)
    writer.add_scalar('valid/mIoU_1', mIoU_1, epoch)
    writer.add_scalar('valid/mIoU_2', mIoU_2, epoch)
    writer.add_scalar('valid/mIoU_3', mIoU_3, epoch)
    writer.add_scalar('valid/mIoU_4', mIoU_4, epoch)
    writer.add_scalar('valid/FWIoU_1', FWIoU_1, epoch)
    writer.add_scalar('valid/FWIoU_2', FWIoU_2, epoch)
    writer.add_scalar('valid/FWIoU_3', FWIoU_3, epoch)
    writer.add_scalar('valid/FWIoU_4', FWIoU_4, epoch)

    print('[Validing]  Epoch [%d/%d], Loss: %.4f' %(epoch, args.max_epochs, average_epoch_loss_val))
    print(
        'Acc_1: %.4f, Acc_class_1: %.4f, mIoU_1: %.4f, FWIoU_1: %.4f ' %
        (Acc_1, Acc_class_1, mIoU_1, FWIoU_1))
    print(
        'Acc_2: %.4f, Acc_class_2: %.4f, mIoU_2: %.4f, FWIoU_2: %.4f ' %
        (Acc_2, Acc_class_2, mIoU_2, FWIoU_2))
    print(
        'Acc_3: %.4f, Acc_class_3: %.4f, mIoU_3: %.4f, FWIoU_3: %.4f ' %
        (Acc_3, Acc_class_3, mIoU_3, FWIoU_3))
    print(
        'Acc_4: %.4f, Acc_class_4: %.4f, mIoU_4: %.4f, FWIoU_4: %.4f ' %
        (Acc_4, Acc_class_4, mIoU_4, FWIoU_4))

    return average_epoch_loss_val, Acc_4, Acc_class_4, mIoU_4, FWIoU_4


def train(args, train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()
    
    train_loss = 0.0
    epoch_loss = []
    tbar = tqdm(train_loader, ncols=80)
    total_batches = len(train_loader)
    for i, (image, depth, target) in enumerate(tbar):
        # start_time = time.time()
        image = image.transpose(1, 3)
        image = image.transpose(2, 3)
        image = image.type(torch.FloatTensor)

        depth = depth.unsqueeze(3)
        depth = depth.transpose(1, 3)
        depth = depth.transpose(2, 3)
        depth = depth.type(torch.FloatTensor)
        
        if args.onGPU == True:
            image = image.cuda()
            depth = depth.cuda()
            target = target.cuda()

        image_var = torch.autograd.Variable(image)
        depth_var = torch.autograd.Variable(depth)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(image_var, depth_var)
        # set the grad to zero
        optimizer.zero_grad()
        # target_var_long = target_var.type(torch.LongTensor)
        # target_var_long = target_var_long.cuda()
        # loss = criteria(output, target_var_long)
        loss1 = L.lovasz_softmax(output[0], target_var)
        loss2 = L.lovasz_softmax(output[1], target_var)
        loss3 = L.lovasz_softmax(output[2], target_var)
        loss4 = L.lovasz_softmax(output[3], target_var)
        loss = loss1+loss2+loss3+loss4

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        writer.add_scalar('Train/loss_iter', loss.item(), i+epoch*total_batches)
        # time_taken = time.time() - start_time

        # pred_1 = output[0].data.cpu().numpy()
        # target = target_var.data.cpu().numpy()
        # pred_1 = np.argmax(pred_1, axis=1)  # 0-4
        # evaluator_1.add_batch(target, pred_1)
        #
        # pred_2 = output[1].data.cpu().numpy()
        # pred_2 = np.argmax(pred_2, axis=1)  # 0-4
        # evaluator_2.add_batch(target, pred_2)
        #
        # pred_3 = output[2].data.cpu().numpy()
        # pred_3 = np.argmax(pred_3, axis=1)  # 0-4
        # evaluator_3.add_batch(target, pred_3)
        #
        pred_4 = output[3].data.cpu().numpy()
        target = target_var.data.cpu().numpy()
        pred_4 = np.argmax(pred_4, axis=1)  # 0-4
        evaluator_4.add_batch(target, pred_4)

        # print('[%d/%d] loss: %.3f time:%.2f' % (i, total_batches, loss.item(), time_taken))
        train_loss += loss.item()
        tbar.set_description('Training loss: %.3f' % (train_loss / (i + 1)))
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    writer.add_scalar('Train/loss_epoch', average_epoch_loss_train, epoch)

    # Acc_1 = evaluator_1.Pixel_Accuracy()
    # Acc_class_1 = evaluator_1.Pixel_Accuracy_Class()
    # mIoU_1 = evaluator_1.Mean_Intersection_over_Union()
    # FWIoU_1 = evaluator_1.Frequency_Weighted_Intersection_over_Union()
    #
    # Acc_2 = evaluator_2.Pixel_Accuracy()
    # Acc_class_2 = evaluator_2.Pixel_Accuracy_Class()
    # mIoU_2 = evaluator_2.Mean_Intersection_over_Union()
    # FWIoU_2 = evaluator_2.Frequency_Weighted_Intersection_over_Union()
    #
    # Acc_3 = evaluator_3.Pixel_Accuracy()
    # Acc_class_3 = evaluator_3.Pixel_Accuracy_Class()
    # mIoU_3 = evaluator_3.Mean_Intersection_over_Union()
    # FWIoU_3 = evaluator_3.Frequency_Weighted_Intersection_over_Union()

    Acc_4 = evaluator_4.Pixel_Accuracy()
    Acc_class_4 = evaluator_4.Pixel_Accuracy_Class()
    mIoU_4 = evaluator_4.Mean_Intersection_over_Union()
    FWIoU_4 = evaluator_4.Frequency_Weighted_Intersection_over_Union()
    # print(
    #     'Epoch [%d/%d], Loss: %.4f, [Training] Acc_1: %.4f, Acc_class_1: %.4f, mIoU_1: %.4f, FWIoU_1: %.4f ' %
    #     (epoch, args.max_epochs, average_epoch_loss_train, Acc_1, Acc_class_1, mIoU_1, FWIoU_1))
    # print(
    #     'Acc_2: %.4f, Acc_class_2: %.4f, mIoU_2: %.4f, FWIoU_2: %.4f ' %
    #     (Acc_2, Acc_class_2, mIoU_2, FWIoU_2))
    # print(
    #     'Acc_3: %.4f, Acc_class_3: %.4f, mIoU_3: %.4f, FWIoU_3: %.4f ' %
    #     (Acc_3, Acc_class_3, mIoU_3, FWIoU_3))
    print(
        'Epoch [%d/%d], Loss: %.4f, [Training] Acc_4: %.4f, Acc_class_4: %.4f, mIoU_4: %.4f, FWIoU_4: %.4f ' %
        (epoch, args.max_epochs, average_epoch_loss_train, Acc_4, Acc_class_4, mIoU_4, FWIoU_4))
    return average_epoch_loss_train, Acc_4, Acc_class_4, mIoU_4, FWIoU_4

def netParams(model):
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


def trainValidateSegmentation(args):

    model = Dual_UNet_MSDF_MMFI(3, 1, args.classes)
    args.savedir = args.savedir + '/' + args.dataset + '/'

    if args.onGPU:
        model = model.cuda()

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    if args.dataset == 'US3D':
        train_loader = torch.utils.data.DataLoader(
            Dataset_US3D(transform=None, train=True),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            Dataset_US3D(transform=None, train=False),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    if args.dataset == 'ISPRS_Vaihingen':
        train_loader = torch.utils.data.DataLoader(
            Dataset_ISPRS_Vaihingen(transform=None, train=True),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            Dataset_ISPRS_Vaihingen(transform=None, train=False),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    if args.dataset == 'ISPRS_Potsdam':
        train_loader = torch.utils.data.DataLoader(
            Dataset_ISPRS_Potsdam(transform=None, train=True),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            Dataset_ISPRS_Potsdam(transform=None, train=False),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    if args.onGPU:
        cudnn.benchmark = True

    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resumeLoc):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resumeLoc)
            start_epoch = checkpoint['epoch']
            # args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val'))
    logger.flush()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, 0.9, weight_decay=5e-8)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = torch.nn.DataParallel(model)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.5)
    for epoch in range(start_epoch, args.max_epochs):
        scheduler.step(epoch)
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))

        evaluator_1.reset()
        evaluator_2.reset()
        evaluator_3.reset()
        evaluator_4.reset()
        lossTr, Acc_tr, Acc_class_tr, mIoU_tr, FWIoU_tr = train(args, train_loader, model, optimizer, epoch)
        evaluator_1.reset()
        evaluator_2.reset()
        evaluator_3.reset()
        evaluator_4.reset()
        with torch.no_grad():
            lossVal, Acc_val, Acc_class_val, mIoU_val, FWIoU_val = val(args, val_loader, model, epoch)

        if mIoU_val > saver.best_mIoU:
            is_best = True
        else:
            is_best = False
        saver.save_checkpoint(epoch, model, optimizer, lossTr, lossVal, mIoU_tr, mIoU_val, lr, Acc_tr, Acc_val, Acc_class_tr, Acc_class_val, FWIoU_tr, FWIoU_val, is_best)

        if args.save_model_epoch != 0 and ((epoch+1) % args.save_model_epoch==0):
            saver.save_model_epoch(epoch, model)

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, lossVal, mIoU_tr, mIoU_val, lr))
        logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--max_epochs', type=int, default=150, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=6, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=50, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='/root/Desktop/code_rewrite/MFMDNet/train_result', help='directory to save the results')
    parser.add_argument('--visualizeNet', type=bool, default=True, help='If you want to visualize the model structure')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Use this flag to load last checkpoint for training') 
    parser.add_argument('--classes', type=int, default=6, help='No of classes in the dataset. 5 for dfs and 6 for class+background')
    parser.add_argument('--logFile', default='trainValLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--dataset', default="ISPRS_Vaihingen", help='US3D or ISPRS_Vaihingen or ISPRS_Potsdam.')
    parser.add_argument('--save_model_epoch', type=int, default=50, help='Save parameters of the model every * epoches.')

    args = parser.parse_args()
    evaluator_1 = Evaluator(args.classes)
    evaluator_2 = Evaluator(args.classes)
    evaluator_3 = Evaluator(args.classes)
    evaluator_4 = Evaluator(args.classes)
    summary = TensorboardSummary(args)
    writer = summary.create_summary()
    saver = Saver(args, 0)
    trainValidateSegmentation(args)
