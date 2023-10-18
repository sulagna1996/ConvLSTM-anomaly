import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from EncoderDecoderConvLSTM import EncoderDecoderConvLSTM
from custom_loader import MeltpoolDataset
from tqdm import tqdm

def create_video(x, y_hat, y):
    # predictions with input for illustration purposes
    #y_hat = y_hat.unsqueeze(1)
    #y_hat = y_hat.unsqueeze(2)
    # print(y_hat.unsqueeze(2).cpu().shape)
    preds = torch.cat([x.cpu(), y_hat.cpu()], dim=1)[0].unsqueeze(1)

    # entire input and ground truth
    #y = y.unsqueeze(1)
    #y = y.unsqueeze(2)
    y_plot = torch.cat([x.cpu(), y.cpu()], dim=1)[0].unsqueeze(1)

    # error (l2 norm) plot between pred and ground truth
    difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
    zeros = torch.zeros(difference.shape)
    # print(difference.shape)
    difference_plot = torch.cat([zeros.cpu().unsqueeze(0), difference.unsqueeze(0).cpu()], dim=1)[0].unsqueeze(1)
    # print(difference_plot.shape)
    # concat all images
    final_image = torch.cat([preds, y_plot, difference_plot], dim=0)

    # make them into a single grid image file
    #import ipdb; ipdb.set_trace()
    grid = torchvision.utils.make_grid(final_image, nrow=n_steps_past + n_steps_ahead)

    return grid


def forward(x):
    x = x.to(device='cuda')
    output = model(x, future_seq=n_steps_ahead)
    return output


if __name__ == '__main__':

    n_steps_past = 10
    n_steps_ahead = 20
    path = 'MP_Train2\\'
    batch_size = 6
    train_data = MeltpoolDataset(
        root_dir=path + 'Train',
        seq_len=n_steps_past + n_steps_ahead,
        time_stride=1)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True)

    test_data = MeltpoolDataset(
        root_dir=path + 'Test',
        seq_len=n_steps_past + n_steps_ahead,
        time_stride=1)
    test_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True)
    model = EncoderDecoderConvLSTM(nf=64, in_chan=1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98))
    criterion.cuda()
    model.cuda()
    model.train()
    epochs = 5


    pbar1 = tqdm(total = epochs, desc='Epoch...', position=0)
    global_step = 0
    loss_list_plot_test = []
    step_list_plot_test = []
    loss_list_plot_train = []

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            model.train()
            x, y = batch[:, 0:n_steps_past, :, :, :], batch[:, n_steps_past:, :, :, :]
            x = x.permute(0, 1, 4, 2, 3)
            y = y.squeeze()
            y_hat = forward(x).squeeze()  # is squeeze neccessary?
            #import ipdb; ipdb.set_trace()
            x = x.squeeze()
            loss = criterion(y_hat, y.cuda())
            loss_list_plot_train.append(loss.item())
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update parameters
            optimizer.step()
            if global_step % 250 == 0:
                final_image = create_video(x, y_hat.cpu(), y)
                torchvision.utils.save_image(final_image, 'Recons/' + str(global_step)+'.png')
                plt.close()
                with torch.no_grad():
                    model.eval()
                    test_loss_iter = 0
                    for batch_idx, batch in enumerate(test_loader):
                        x, y = batch[:, 0:n_steps_past, :, :, :], batch[:, n_steps_past:, :, :, :]
                        x = x.permute(0, 1, 4, 2, 3)
                        y = y.squeeze()
                        y_hat = forward(x).squeeze()  # is squeeze neccessary?
                        # import ipdb; ipdb.set_trace()
                        x = x.squeeze()
                        loss = criterion(y_hat, y.cuda())
                        test_loss_iter+=loss.item()
                    loss_list_plot_test.append(test_loss_iter/len(test_loader))
                    step_list_plot_test.append(global_step)

            global_step += 1
        pbar1.update()
    torch.save(model.state_dict(),'trained_params_1020w_size.pt')
    plt.figure(1)
    plt.plot(loss_list_plot_train, label='Training Loss')
    plt.plot(step_list_plot_test, loss_list_plot_test, label='Testing Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Mean Squared Error')
    plt.legend(loc=1, prop={'size':9})
    plt.show()