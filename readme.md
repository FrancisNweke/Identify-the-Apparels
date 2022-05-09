# Identify-the-Apparels
Fashion MNIST is a drop-in replacement for the very well known, machine learning hello world - MNIST dataset which can be checked out at ‘Identify the digits’ practice problem. Instead of digits, the images show a type of apparel e.g. T-shirt, trousers, bag, etc. The dataset used in this problem was created by Zalando Research. 
More details can be found at this link: https://github.com/zalandoresearch/fashion-mnist

We have total 70,000 images (28 x 28), out of which 60,000 are part of train images with the label of the type of apparel (total classes: 10) and rest 10,000 images are unlabelled (known as test images).The task is to identify the type of apparel for all test images. Given below is the code description for each of the apparel class/label.

### Labels
Each training and test example is assigned to one of the following labels:

| Label | Description |
|-------|-------------|
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

## Project Structure

```
.
├── custom_models   
│   └── convNet.py 
├── utilities   
│   └── convnet.h5
│   └── convNet.png
│   └── processdata.py
│   └── save_load_model.py
│   └── submit-prediction.csv
│   └── submit-prediction-template.csv
├── main.py
├── README.md
├── requirements.txt
```

## Usage

```
python3 main.py
```

## Acknowledgement
Many thanks to Analytics Vidhya. See link: https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/

## License
This project is licensed under the terms of the [MIT license](https://choosealicense.com/licenses/mit/).
