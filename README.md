# Neural Network Project
## Learning to play Connect4 using a Recurrent Neural Network
Recurrent Neural Networks are widely used today for various tasks: text classification, text generation, audio or video processing, time-series forecasting and many others.\
The main advantage of Recurrent Networks is the ability of the recurrent neurons - fundamental components of the network - to be able to intercept patterns within data sequences with a precise ordering. Indeed, the applications mentioned above are characterised by sequential data.\
This project stems precisely from the realisation that most games are made up of sequences of actions performed by players. These actions follow a precise order and, therefore, it is assumed that there is a pattern that determines future actions.\
This observation gave rise to the idea of applying a Neural Network with a hybrid convolutional-recurrent architecture to games, in particular the popular board game Connect 4. After an initial phase of dataset realisation, various experiments were conducted to evaluate the network's performance: the results show a certain degree of learning on the part of the network, despite the severe computational limitations of the available hardware.
The complete **results** are found in the `main.ipynb` file.\
Detailed code descriptions and instructions can be found in the files named after the `.py` files, inside the `tools` folder: feel free to read (and use) every file in the repository.
