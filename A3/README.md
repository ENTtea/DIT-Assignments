# Play with GANs
This repository is Jiahao Zhang's implementation of Assignment_03 of DIP.
![None](pic/UI.png)

## Requirements

To install requirements:

```bash
python -m pip install -r requirements.txt
```

## Running
To run Pix2Pix with GANs, navigate to the target folder `pix2pix with GANs` and run `train.py`:

```bash
cd pix2pix with GANs
python train.py
```

To run automatic editing with dragGANs, 
1. Clone the original implementation of the DragGAN paper:
```bash
git clone <repository-url>
cd <repository-name>
```
2. Install the required libraries and download the pre-trained models as specified in requirements.txt and scripts/download_model.py
```bash
pip install -r requirements.txt
python scripts/download_model.py
```
3. Add the Python file auto_drag_gradio.py from folder automatic picture editing to the cloned repository.
4. Run the auto_drag_gradio.py file:
```bash
python auto_drag_gradio.py
```
5. If you encounter an error when clicking the four automatic editing buttons for the first time, click them again. This should generate the target editing points.
6. Once the target points are visible, click the Start button in the interface to begin dragging the image.

## Results

### Pix2Pix with GANs
Partial results of training sets and validation sets

train_results:
<p align="center">
  <img src="https://github.com/ENTtea/DIT-Assignments/blob/main/A3/pic/train1.png" alt="Image 11" width="500" style="display: inline-block;"/>
</p>
<p align="center">
  <img src="https://github.com/ENTtea/DIT-Assignments/blob/main/A3/pic/train2.png" alt="Image 11" width="500" style="display: inline-block;"/>
</p>
valid_results:
<p align="center">
  <img src="https://github.com/ENTtea/DIT-Assignments/blob/main/A3/pic/val1.png" alt="Image 11" width="500" style="display: inline-block;"/>
</p>
<p align="center">
  <img src="https://github.com/ENTtea/DIT-Assignments/blob/main/A3/pic/val2.png" alt="Image 11" width="500" style="display: inline-block;"/>
</p>

### Automatic editing with DragGANs

smile:

<p align="center">
  <img src="https://github.com/ENTtea/DIT-Assignments/blob/main/A3/pic/smile_s.png" alt="Image 11" width="300" style="display: inline-block;"/>
  <img src="https://github.com/ENTtea/DIT-Assignments/blob/main/A3/pic/smile_t.png" alt="Image 12" width="300" style="display: inline-block;"/>
</p>

thin face:

<p align="center">
  <img src="https://github.com/ENTtea/DIT-Assignments/blob/main/A3/pic/thin_s.png" alt="Image 11" width="300" style="display: inline-block;"/>
  <img src="https://github.com/ENTtea/DIT-Assignments/blob/main/A3/pic/thin_t.png" alt="Image 12" width="300" style="display: inline-block;"/>
</p>

closed eye:

<p align="center">
  <img src="https://github.com/ENTtea/DIT-Assignments/blob/main/A3/pic/closed_s.png" alt="Image 11" width="300" style="display: inline-block;"/>
  <img src="https://github.com/ENTtea/DIT-Assignments/blob/main/A3/pic/closed_t.png" alt="Image 12" width="300" style="display: inline-block;"/>
</p>

big eye:

<p align="center">
  <img src="https://github.com/ENTtea/DIT-Assignments/blob/main/A3/pic/big_s.png" alt="Image 11" width="300" style="display: inline-block;"/>
  <img src="https://github.com/ENTtea/DIT-Assignments/blob/main/A3/pic/big_t.png" alt="Image 12" width="300" style="display: inline-block;"/>
</p>


## Acknowledgement

📋 Thanks for the algorithms of image editing proposed by [Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf).

📋 Thanks for the algorithms of the pix2pix proposed by [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038).
