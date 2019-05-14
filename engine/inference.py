import numpy as np
from tqdm import tqdm

def eval(val_loader, model, criterion, params, epoch, logger):
    """Main method to evaluate model."""
    model.eval()
    print("================================")
    print("Evaluating...")

    loss = 0
    num_classes = params["num_classes"]

    confusion = np.zeros((num_classes, num_classes))
    for iteration, (steps, targets, _) in enumerate(tqdm(val_loader)):
        if params["use_cuda"]:
            steps = steps.cuda()
            targets = targets.cuda()

        output = model(steps)

        rows = targets.cpu().numpy()
        cols = output.max(1)[1].cpu().numpy()

        confusion[rows, cols] += 1

        loss += criterion(output, targets)

    loss = loss / iteration
    acc = np.trace(confusion) / np.sum(confusion)

    # Plot confusion matrix in visdom
    logger.heatmap(confusion, win='4', opts=dict(
        title="Confusion_Matrix_epoch_{}".format(epoch), 
        columnnames=["A","B","C","D","E"],
        rownames=["A","B","C","D","E"])
    )

    return loss, acc