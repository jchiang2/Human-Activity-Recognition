import torch
import torch.nn as nn
import visdom
from config.set_params import params as sp
from tqdm import tqdm
from modeling.model import HARmodel
from data.preprocessing import HARdataset



def train(train_loader,
          model,
          criterion,
          optimizer,
          num_classes,
          start_epoch,
          max_epoch,
          logger
    ):

    model.train()
    start_epoch = start_epoch

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    print("Start training...")
    for epoch in range(start_epoch, max_epoch):
        scheduler.step()
        for iteration, (steps, targets, _) in enumerate(train_loader):
            # if use_cuda:
            #     steps = steps.cuda()
            #     targets = targets.cuda()

            output = model(steps)

            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if iteration % 20 == 0:
            print("iter: {iter}, loss: {loss}".format(iter=iteration,
                                                          loss=loss))

        if epoch % 2 == 0:
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                }, "model_{}.pth".format(epoch))


def eval(val_loader, model, criterion, num_classes, state, logger):
    model.eval()
    print("Evaluating...")

    loss = 0
    confusion = np.zeros((num_classes, num_classes))
    for iteration, (steps, targets, _) in enumerate(tqdm(val_loader)):
        # if use_cuda:
        #     steps = steps.cuda()
        #     targets = targets.cuda()

        output = model(steps)

        rows = targets
        cols = torch.max(output, dim=1)

        confusion[rows, cols] += 1

        loss += criterion(output, targets)

    loss = loss / iteration
    acc = np.trace(confusion) / np.sum(confusion)
    print("Avg Loss: {loss}, Acc: {acc}".format(loss=(loss),
                                                acc=acc))

def save_checkpoint(state, filename):
    torch.save(state, filename)

def main():
    model = HARmodel(1, 5)

    # if use_cuda:
    #     model = model.cuda()
    params = sp().params

    root = params["root"]
    workers = params["workers"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    lr = params["lr"]
    momentum = params["momentum"]
    weight_decay = params["weight_decay"]
    num_classes = params["num_classes"]

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    start_epoch = 0
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         start_epoch = checkpoint["epoch"]
    #         model.load_state_dict(checkpoint["state_dict"])
    #         optimizer.load_state_dict(checkpoint["optimizer"])
    #         print("=> loaded checkpoint '{}' (epoch {})".format(
    #             args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    dataset = HARdataset(root)
    train_sampler, val_sampler = dataset.split_ind(val_split=0.2, shuffle=True)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               num_workers=workers,
                                               pin_memory=True,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=workers,
                                             pin_memory=True,
                                             sampler=val_sampler)
    # logger = visdom.Visdom()
    logger = None

    train(train_loader=train_loader,
          model=model,
          criterion=criterion,
          optimizer=optimizer,
          num_classes=num_classes,
          start_epoch=start_epoch,
          max_epoch=epochs,
          logger=logger,
          )

if __name__ == "__main__":
    main()