
import os


import numpy as np

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset

mnist_train = MNIST(
    'data/MNIST',
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    train=True
)

mnist_test = MNIST(
    'data/MNIST',
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    train = False
)


# Then we synthetically add high noise using `torch`:

# In[4]:


from torch import randn

def add_noise(img):
    return img + randn(img.size())*0.4

class SyntheticNoiseDataset(Dataset):
    def __init__(self, data, mode='train'):
        self.mode = mode
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index][0]
        return add_noise(img), img


noisy_mnist_train = SyntheticNoiseDataset(mnist_train, 'train')
noisy_mnist_test = SyntheticNoiseDataset(mnist_test, 'test')


noisy, clean = noisy_mnist_train[0]



from mask import Masker
masker = Masker(width=4, mode='interpolate')

from models.babyunet import BabyUnet
model = BabyUnet()


from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

loss_function = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)


# We lazily convert the torch tensors to NumPy arrays and concatenate them into dask arrays containing all the data. We do this for the training (noisy) data, the ground truth, and the model output.
#
# There's a bit of reshaping because torch data comes with extra dimensions that we want to squeeze out, to only get a `(nsamples, size_y, size_x)` volume.
#
# Finally, because of a [performance issue with `dask.array.stack`](https://github.com/dask/dask/issues/5913), we convert the input data to a non-lazy numpy array for the moment.


from dask import array as da, delayed

def get_noisy_test_image(arr, block_id):
    j = block_id[0]
    return (
        noisy_mnist_test[j][0].detach().numpy().reshape((1, 28, 28))
    )


def get_clean_test_image(arr, block_id):
    j = block_id[0]
    return (
        noisy_mnist_test[j][1].detach().numpy().reshape((1, 28, 28))
    )

n = len(noisy_mnist_test)

noisy_test_dask = da.map_blocks(
            get_noisy_test_image,
            np.arange(1),
            chunks=((1,) * n, (28,), (28,)),
            dtype=np.float32,
        )
noisy_test = noisy_test_dask

clean_test_dask = da.map_blocks(
            get_clean_test_image,
            np.arange(1),
            chunks=((1,) * n, (28,), (28,)),
            dtype=np.float32,
        )
clean_test = clean_test_dask

import torch

def test_numpy_to_result_numpy(dask_array):
    """Convert test NumPy array to model output and back to NumPy."""
    out = model(
        torch.Tensor(np.array(dask_array)[:, np.newaxis])
    ).detach().numpy()[:, 0]
    return out

# build the results dask array
model_output_dask = da.map_blocks(
        test_numpy_to_result_numpy,
        noisy_test_dask,
        dtype=np.float32,
        )


import napari
from napari.utils import resize_dask_cache
resize_dask_cache(0)


with napari.gui_qt():
    viewer = napari.Viewer()
    _ = viewer.add_image(clean_test)  # returns layer, we don't care
    _ = viewer.add_image(noisy_test)  # returns layer, we don't care
    model_layer = viewer.add_image(
        model_output_dask,
        colormap='magma',
    )  # this layer though, we're gonna play with
    viewer.grid_view()


    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvas
    from matplotlib.figure import Figure

    NUM_ITER = 500


    # build the plot, but don't display it yet
    # — we'll add it to the napari viewer later
    with plt.style.context('dark_background'):
        loss_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        loss_axes = loss_canvas.figure.subplots()
        lines = loss_axes.plot([], [])  # make empty plot
        loss_axes.set_xlim(0, NUM_ITER)
        loss_axes.set_xlabel('batch number')
        loss_axes.set_ylabel('loss')
        loss_canvas.figure.tight_layout()
        loss_line = lines[0]


    # Napari's threading utilities, created by Talley Lambert, allow *yielding* of values during a thread's execution, and connecting those yielded values to callbacks. Below, we create callbacks to update the loss plot and the displayed model output:



    # when getting a new loss, update the plot
    def update_plot(loss):
        x, y = loss_line.get_data()
        new_y = np.append(y, loss)
        new_x = np.arange(len(new_y))
        loss_line.set_data(new_x, new_y)
        loss_axes.set_ylim(
            np.min(new_y) * (-0.05), np.max(new_y) * (1.05)
        )
        loss_canvas.draw()


    # and update the model output layer
    def update_viewer(loss):
        model_layer.refresh()
        model_layer._set_view_slice()
        viewer.help = f'loss: {loss}'


    # We then do two things to a standard PyTorch training loop:
    # 
    # - we wrap it in a function with the `@thread_worker` decorator
    # - inside the function, we yield the loss value after each training batch

    from napari.qt import thread_worker


    # define a function to train the model in a new thread,
    # connecting the yielded loss values to our update functions
    @thread_worker(connect={'yielded': [update_viewer, update_plot]})
    def train(model, data_loader, n_iter):

        for i, batch in zip(range(n_iter), data_loader):
            noisy_images, clean_images = batch

            net_input, mask = masker.mask(noisy_images, i)
            net_output = model(net_input)

            loss = loss_function(net_output*mask, noisy_images*mask)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            yield round(loss.item(), 4)


    # Finally, we create the PyTorch DataLoader, add the loss plot to our viewer, and start training! You should be able to see the model output refine over time, while simultaneously browsing through the whole test dataset.

    # In[ ]:


    # finally, add the plot to the viewer, and start training!
    data_loader = DataLoader(noisy_mnist_train, batch_size=32, shuffle=True)

    viewer.window.add_dock_widget(loss_canvas)
    worker = train(model, data_loader, NUM_ITER)


