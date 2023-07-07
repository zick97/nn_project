#### Human Player
The choice is simply the user's input.

#### Random Player
The choice is picked randomly from the list of available columns.

#### Simple Player
The strategy is very simple: it puts its tokens randomly until it finds that one line is almost filled up with three tokens of the same player plus one empty space. It then places the token inside the empty space, giving priority to winning moves.\
The main issue about the `SimplePlayer()` class is that, when he recognizes any row, column or diagonal that is almost full (e.g.: it sums up to 3) it chooses the column to fill the empty space without worrying if this move would end in a defeat. For example, if there is an almost full line at row 3, from column 0 to 3, it starts putting its tokens in column 4 until the empty space gets filled by any token.\
This is basically a serious lack of foresight skills and, although it can be fixed with a bit of hard-coding, I decided to try to make the RNN avoid this problem by training it on a lsrge dataset.

#### RNN Player
The choice is determined by the model's predicted probabilities. As you can see in the code, if the highest probability index, that is, the column where to put the token, is NOT in the list of available columns, the algorithm takes the second highest probability index and so on.\
The RNN makes predictions on an entire batch (since Keras' models are optimized for *batch prediction*), then we take just the last sequence of the last batch.\
Note that we trained the network on sequences of fixed length (`mean_duration`) but during each game we have to wait `mean_duration` moves before the sequence gets filled up. In fact, we start with a sequence made of empty grids, then we fill it with one updated grid at a time and, when the number of moves eventually becomes greater than `mean_duration`, we start shifting this sequence, discarding the first moves.\
\
This method can be definitely improved: during the first few moves the network does not really know what to do since it was trained on the last `mean_duration` moves of simulated games. This issue is partially alleviated by the fact that, in the training set, many games last less then the average duration, therefore many sequences contain multiple empty grids. However, as we can see in the `main.ipynb` file, the shorter possible games last around 7 moves, so the training games are, at least, 7 full grids long.\
The effects of this problem can be noticed in the RNN's behavior when playing against the user: during the first moves of the game, the network puts the tokens close to each other, but it can not detect a winning line. Furthermore, it seems to be giving priorirty to the interruption of its opponents' moves rather than to its winning moves.
\
\
__*Preparing the Dataset*__\
To make the Network learn a robust winning strategy, we want to train it using only the winning moves: the `df_to_tensor()` function takes care of exactly that.\
Briefly, this function manipulates the Dataframe as follows:
- for each row (player's move) it concatenates the grid's columns along the first axis, obtaining a *grid* feature that is just a 1D array with $height_{board} \times width_{board}$ items,
- it computes the mean duration ($\mu$) of the games (`int` number) and then it takes only the last $\mu$ moves from each game.\
Some games may have a shorter number of moves, therefore this function also expands them by using an auxiliary `padding()` function, which appends the required number of padding rows at the beginning of the sequence.
- transforms the dataset into a Tensorflow's tensor, batching it into equally-sized batches, each of them corresponding to the last $\mu$ moves of a game.
*Padding Row*: a row with an empty grid associated to the $0^{th}$ column as a choice. In this case, we suppose that it's fine to place the token in the first column of an empty grid.\
\
Finally, the `quickdraw_dataset()` function caches, shuffles and batches the resulting dataset into batches of size 32.
