import torch
import numpy as np
import os
from config.set_params import params
from .inference import eval

def train(train_loader,
          val_loader,
          model,
          criterion,
          optimizer,
          params,
          logger,
    ):
    """Main method to run training"""
    model.train()
    start_epoch = params["start_epoch"]
    max_epoch = params["epochs"]

    # Learning rate decay scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    print("Start training...")
    for epoch in range(start_epoch, max_epoch):
        scheduler.step()
        for iteration, (steps, targets, _) in enumerate(train_loader):
            if params["use_cuda"]:
                steps = steps.cuda()
                targets = targets.cuda()

            output = model(steps)

            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print("Epoch: {ep}/{max}, iter: {iter}, loss: {loss}".format(
                                                            ep=epoch,
                                                            max=max_epoch,
                                                            iter=iteration,
                                                            loss=loss)
                                                            )
        # Log average loss to visdom after epoch
        logger.line([loss.cpu().detach().numpy() / iteration],
                         [epoch],
                         opts=dict(title="Training_loss"),
                         win='1',
                         name='loss',
                         update='append'
                )
        # Evaluate model
        if epoch % 2 == 0:
            val_loss, val_acc = eval(val_loader,
                                     model,
                                     criterion,
                                     params,
                                     epoch,
                                     logger
                )
            # Log validation loss to visdom
            logger.line([val_loss.cpu().detach().numpy()],
                     [epoch],
                     opts=dict(title="Val_loss"),
                     win='2',
                     name='val_loss',
                     update='append'
                )
            # Log validation accuracy to visdom
            logger.line([val_acc],
                         [epoch],
                         opts=dict(title="Val_acc"),
                         win='3',
                         name='acc',
                         update='append'
                )
            print("Evaluation results: Loss: {}, Acc: {}".format(val_loss,
                                                                 val_acc)
            )
            print("================================")

            model.train()
            
        # Save checkpoint
        if epoch % 25 == 0:
            if not os.path.exists("Checkpoints"):
                os.makedirs("Checkpoints")
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                }, "Checkpoints/model_{}.pth".format(str(epoch).zfill(3)))

def save_checkpoint(state, filename):
    """Saves current model."""
    torch.save(state, filename)