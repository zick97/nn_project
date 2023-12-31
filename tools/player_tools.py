import numpy as np
import pandas as pd
import tensorflow as tf
#---------------------------------------------------------------------------------------------
class Player(object):
    def __init__(self, p=1, name='Player1'):
        self.player = p                     # number ID to produce the grid markers
        self.name = name                    # the name is used only for displaying purposes and to avoid misunderstandings
        if self.player == 1:
            self.marker = 1
        else:
            self.marker = -1
        self.target = int(4 * self.marker)  # +4 for player 1 and -4 for player 2

#---------------------------------------------------------------------------------------------       
class HumanPlayer(Player):
    def __init__(self, p=1, name='Human'):
        Player.__init__(self, p, name)
        self.player_type = 'Human'

    def move(self, Board):
        self.choice = int(input('Digit a column number to place your token.'))
        
class RandomPlayer(Player):
    def __init__(self, p=1, name='Random'):
        Player.__init__(self, p, name)
        self.player_type = 'RandomAI'

    def move(self, Board):
        # columns with available moves
        valid_columns = [i for i, v in enumerate(Board.column_n_pos) if v != 0]
        self.choice = np.random.choice(valid_columns)

class SimplePlayer(Player):
    def __init__(self, p=1, name='Simple'):
        Player.__init__(self, p, name)
        self.player_type = 'SimpleAI'

    def move(self, Board):
        # columns with available moves
        valid_columns = [i for i, v in enumerate(Board.column_n_pos) if v != 0]
        # first of all, it guesses at random 
        # then, using the following algorithm, it checks for rows, columns or diagonals that
        # may lead to victory or defeat and fills the column where there is still a 0
        rows_number = (Board.width - 3) * Board.height  # number of checking rows
        cols_number = Board.width * (Board.height - 3)  # number of checking columns
        # below we write, respectively, the checking rows, columns and diagonals vectors stored
        # inside the board class
        rows = Board.vectors[:rows_number]
        cols = Board.vectors[rows_number:(rows_number + cols_number)]
        diags = Board.vectors[(rows_number + cols_number):]
        # now we want to identify the column of each element inside these vectors
        # to do so, we use an equation explained in the markdown file
        if self.target == 4:
            target = 3
        else:
            target = -3

        column = -1 # initialize the column to a non-valid value
        for i, vector in enumerate([rows, cols, diags]):
            targets = np.array([sum(v) for v in vector])           # sum of each vector (row, col, or diag) 
            vector_index = np.where(targets == -(target))          # list of indexes where the vector can be completed
            if len(vector_index[0]) > 0:
                index = np.where(vector[vector_index[0][0]] == 0)  # index of the 0 inside the vector
                if i == 0:
                    column = int(vector_index[0][0] / Board.height) + index[0][0]
                if i == 1:
                    column = vector_index[0][0] % Board.width
                if i == 2:
                    column = int(int(vector_index[0][0] / 2) / (Board.height - 3)) + index[0][0]
                break
            # the same thing can happen for a winning move: by putting it after the defeat scenario,
            # we give priority to winning moves
            vector_index = np.where(targets == target)
            if len(vector_index[0]) > 0:
                index = np.where(vector[vector_index[0][0]] == 0)
                if i == 0:
                    column = int(vector_index[0][0] / Board.height) + index[0][0]
                if i == 1:
                    column = vector_index[0][0] % Board.width
                if i == 2:
                    column = int(int(vector_index[0][0] / 2) / (Board.height - 3)) + index[0][0]
                break

        if column in valid_columns:
            self.choice = column
        else:
            self.choice = np.random.choice(valid_columns)

class RNNPlayer(Player):
    def __init__(self, p=1, name='RNN', model=tf.keras.Sequential()):
        Player.__init__(self, p, name)
        self.player_type = 'RecurrentAI'
        self.model = model
    
    def move(self, Board, batch):  # note that we use the entire batch to make the predictions
        valid_columns = [i for i, v in enumerate(Board.column_n_pos) if v != 0]
        # column probabilities on the current sequence
        probas = self.model.predict(batch, verbose=0)[-1:][0]
        # sort in descending order the probas array and get the index list
        choices = np.argsort(-probas)
        # take the first element of the list that is also in the valid_columns list
        self.choice = choices[np.isin(choices, valid_columns)][0] 

from tqdm import tqdm
#---------------------------------------------------------------------------------------------       
pd.options.mode.chained_assignment = None  # default='warn'
def padding(dataset):
    df = pd.DataFrame()
    # getting the mean duration of each game
    mean_duration = int(dataset['grid'].groupby(dataset.index).count().mean())
    # creating the padding row: it's just an empty board with a random choice
    padding_row = dataset[:1]
    padding_row['grid'] = [np.zeros(shape=dataset['grid'].iloc[0].shape)]
    padding_row['choice'] = np.int64(0)
    # now, for each sequence shorter than mean_duration, we append (mean_duration - len(sequence)) 
    # padding rows at the beginning of the sequence
    def append_rows(sequence):
        difference = abs(mean_duration - len(sequence))
        if difference > 0:
            # repeating padding rows
            padding_rows = pd.DataFrame(np.repeat(padding_row.values, difference, axis=0),
                                        index=[sequence.index[0]]*difference, columns=sequence.columns)
            return pd.concat([padding_rows, sequence])
        else:
            return sequence
    df = dataset.groupby(dataset.index).apply(append_rows).droplevel(0)
    return df, mean_duration

#---------------------------------------------------------------------------------------------       
import tensorflow as tf
def df_to_tensor(dataset):
    df = dataset.copy()
    df.drop(columns=['player', 'move'], inplace=True)
    # stacking all the columns to get just one feature to train the RNN with 
    df['grid'] = df[df.columns[1:]].apply(lambda x: np.concatenate(x), axis=1)
    # dropping all the single columns 
    df.drop(columns=df.columns[1:-1], inplace=True)
    # padding inside the dataframe
    df, mean_duration = padding(df)
    # we are going to use the last mean_duration moves for each game
    df = df.groupby(df.index).tail(mean_duration)
    # with the following procedure, we build the tensorflow datasets stacking together all the column vectors
    # for the features dataset and all the final moves for the target dataset
    features = tf.data.Dataset.from_tensor_slices(list(df['grid']))
    features = features.batch(mean_duration)
    # taking the last move (the winning one) as target
    targets = tf.data.Dataset.from_tensor_slices(list(df['choice'].groupby(df.index).tail(1)))
    df = tf.data.Dataset.zip((features, targets))
    return df

#---------------------------------------------------------------------------------------------
def quickdraw_dataset(dataset, batch_size=32, shuffle_buffer_size=None, cache=False):
    if cache:
        # cache the dataset
        dataset = dataset.cache()
    if shuffle_buffer_size:
        # shuffle the dataset's batches
        dataset = dataset.shuffle(shuffle_buffer_size)
    # batch the games into batches of length 32, meaning that each batch will contain 32 games
    dataset = dataset.batch(batch_size) 
    return dataset