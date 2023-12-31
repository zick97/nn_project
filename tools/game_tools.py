import numpy as np
import pandas as pd
from tools.player_tools import *

#---------------------------------------------------------------------------------------------        
class Board(object):
    def __init__(self, grid_size=(6,7)):
        # winner checking: 1 for player1, -1 for player 2 and 0 for "tie"
        self.winner = 0

        # grid parameters
        self.height = grid_size[0]
        self.width = grid_size[1]
        self.n_positions = self.height * self.width # number of initial positions

        # how many positions are left for each column
        self.column_n_pos = np.zeros(self.width, dtype=np.int16) + self.height
        # how many total positions are left
        self.n_pos_left = self.n_positions

        # initializing functions
        self.initialize_grids(grid_size)
        self.initialize_vectors()

    # function for initializing the grids used to keep track of the various positions
    def initialize_grids(self, grid_size):
        self.grid = np.zeros(shape=grid_size)                   # main grid
        self.valid_grid = np.zeros_like(self.grid, dtype=bool)  # grid for valid moves
        # initialize the whole first row as valid
        self.valid_grid[-1] = True
        # grid containing, for each column, the number of the column itself
        self.column_grid = np.array([[i for i in range(self.width)] for j in range(self.height)])
    
    # function for initializing vectors that will be used later for the update process
    # these vectors contain basically all the needed information about the grid state
    def initialize_vectors(self):
        self.vectors = []           # used to check if the winning condition is satisfied
        self.valid_vectors = []     # used to let the player make a valid move during the game
        self.column_vectors = []    # used to gather the information about the columns' state during the game

        # to check for a winner, the next loop goes through the whole grid looking for a filled "Connect4"
        # the view() method is only used to generate a new view of the arrays using the same data
        for j in range(self.width - 3): # exclude the last 3 columns
            for i, sub_row in enumerate(self.grid):
                # view the next 4 cells 
                grid_view = self.grid[i][j:j+4].view()   
                valid_view = self.valid_grid[i][j:j+4].view()
                column_view = self.column_grid[i][j:j+4].view()
        
                self.vectors.append(grid_view)
                self.valid_vectors.append(valid_view)
                self.column_vectors.append(column_view)

        # the column views are very similar to the previous ones, with the exception that we are using the 
        # transpose of the previous grid
        t_grid = self.grid.transpose()
        t_valid_grid = self.valid_grid.transpose()
        t_column_grid = self.column_grid.transpose()

        for j in range(self.height - 3): # exclude the last 3 rows
            for i, sub_col in enumerate(t_grid):
                # view the next 4 cells 
                grid_view = t_grid[i][j:j+4].view()   
                valid_view = t_valid_grid[i][j:j+4].view()
                column_view = t_column_grid[i][j:j+4].view()
        
                self.vectors.append(grid_view)
                self.valid_vectors.append(valid_view)
                self.column_vectors.append(column_view)

        # diagonal views are a bit different: we firstly divide the grids into smaller 4x4 sub-grids
        # then, using numpy's diagonal and flip methods, we take both the diagonal and the anti-diagonal
        # of these sub-grids
        for j in range(self.width - 3):
            for i in range(self.height - 3):
                sub_grid = self.grid[i:i+4, j:j+4]
                sub_valid_grid = self.valid_grid[i:i+4, j:j+4]
                sub_column_grid = self.column_grid[i:i+4, j:j+4]

                grid_diag = np.diagonal(sub_grid).view()
                grid_adiag = np.flipud(sub_grid).diagonal().view()
                valid_grid_diag = np.diagonal(sub_valid_grid).view()
                valid_grid_adiag = np.flipud(sub_valid_grid).diagonal().view()
                column_grid_diag = np.diagonal(sub_column_grid).view()
                column_grid_adiag = np.flipud(sub_column_grid).diagonal().view()

                self.vectors.append(grid_diag)
                self.vectors.append(grid_adiag)
                self.valid_vectors.append(valid_grid_diag)
                self.valid_vectors.append(valid_grid_adiag)
                self.column_vectors.append(column_grid_diag)
                self.column_vectors.append(column_grid_adiag)
                
    #---------------------------------------------------------------------------------------------
    # function used to update the grid state, based on a Player's choice (the column to place the token in)
    def update_grid(self, Player):
        choice = Player.choice
        # row corresponding to the column choice 
        row = self.column_n_pos[choice] - 1 # since the enumeration starts from 0

        # first check: is there place for an additional token in that column?
        valid_columns = [i for i, v in enumerate(self.column_n_pos) if v != 0]
        if choice not in valid_columns:
            print(f'Error: this column number {choice} has no more valid positions!')
            return -1
        
        # update values in the main grid and the boolean one, using the player's marker (-1 or +1)
        self.grid[row,choice] = Player.marker
        self.valid_grid[row,choice] = False
        # the cell on top of the token becomes available
        if (row != 0):  # the last top row 
            self.valid_grid[row-1,choice] = True

        # update the number of moves left in the choice column
        if self.column_n_pos[choice] > 0:
            self.column_n_pos[choice] -= 1

        elif self.column_n_pos[choice] == 0:
            print(f'Error: the number of valid positions in column number {choice} is already 0!')
            return -1
        
        # update the total number of moves left
        if self.n_pos_left <= 0:
            print('Error : there are no positions left!')
            return -1
        else :
            self.n_pos_left -= 1

    
    def check_winner(self, Player):
        win = False
        for vector in self.vectors:
            if sum(vector) == Player.target:
                self.winner = Player.marker
                win = True
                break
        return win
    
    def display_grid(self):
        display = np.zeros_like(self.grid).astype(int).astype(str)
        for x, row in enumerate(display):
            for y, _ in enumerate(row):
                if(self.grid[x][y] == -1):
                    display[x][y] = 'O'
                elif(self.grid[x][y] == 1):
                     display[x][y] = 'X'
                elif(self.grid[x][y] == 0):
                     display[x][y] = ' '
        count = 0
        pattern = '   '
        for i in range(display.shape[1]):
            pattern += '+-----'
        pattern += '+'
        for row in display:
            txt = str(count)
            for cell in row:
                txt += '  |  ' + cell
            txt += '  |'
            print(txt)
            print(pattern)
            count += 1

        txt = '      '
        for i in range(display.shape[1]):
            txt += str(i) + '     '
        print(txt)

#---------------------------------------------------------------------------------------------       
import math
from IPython.display import clear_output

class Game(object):
    def __init__(self, game_type=None, verbose=False, pause=False, player1=None, player2=None, 
                 model=tf.keras.Sequential()):
        self.verbose = verbose              # shows the grid during the game
        self.pause = pause                  # used when playing against AI to visualize its moves      
        self.flag = False                   # used to break the loop at the end of the game
        self.Board = Board()                # generate the board and all the useful vectors
        
        self.game_record = {'player' : [], 
                            'choice' : [],  # dictionary used to save the game
                            'grid' : []}

        if game_type=='user-user':
            p1 = HumanPlayer(name=player1, p=1)
            p2 = HumanPlayer(name=player2, p=-1)

        elif game_type=='user-random':
            p1 = HumanPlayer(name=player1, p=1)
            p2 = RandomPlayer(name=player2, p=-1)

        elif game_type=='user-simple':
            p1 = HumanPlayer(name=player1, p=1)
            p2 = SimplePlayer(name=player2, p=-1)
        
        elif game_type=='user-rnn':
            p1 = HumanPlayer(name=player1, p=1)
            p2 = RNNPlayer(name=player2, p=-1, model=model)

        elif game_type=='random-random':
            p1 = RandomPlayer(name=player1, p=1)
            p2 = RandomPlayer(name=player2, p=-1)

        elif game_type=='simple-random':
            p1 = SimplePlayer(name=player1, p=1)
            p2 = RandomPlayer(name=player2, p=-1)
        
        elif game_type=='simple-simple':
            p1 = SimplePlayer(name=player1, p=1)
            p2 = SimplePlayer(name=player2, p=-1)

        elif game_type=='rnn-random':
            p1 = RNNPlayer(name=player1, p=1, model=model)
            p2 = RandomPlayer(name=player2, p=-1)

        elif game_type=='rnn-simple':
            p1 = RNNPlayer(name=player1, p=1, model=model)
            p2 = SimplePlayer(name=player2, p=-1)

        if p1.name == None:
            p1.name = f'{p1.player_type}_1'
        if p2.name == None:
            p2.name = f'{p2.player_type}_2'
            
        # the starting player is always chosen at random 
        self.player_list = [p1, p2]
        self.player_types = [p1.player_type, p2.player_type]
        np.random.shuffle(self.player_list)

        if self.verbose:
            print(f'Game Type: {game_type}')
            print(f'Player #1: {self.player_list[0].name}, {self.player_list[0].player_type}')
            print(f'Player #2: {self.player_list[1].name}, {self.player_list[1].player_type}')
    
    # this function runs through a game of Connect4 by first asking the player for a move, then updating the grid and,
    # finally, checking for a winner (eventually, it stops when reaching the total number of possible moves)
    def play_game(self, record=True, mean_duration=21):
        self.move_counter = 0
        # initialize the batch to make the prediction on
        batch_np = np.zeros(shape=(32, mean_duration, self.Board.height * self.Board.width))
        # the sequence is the board states sequence that gets updated during the game
        sequence = batch_np[0]
        for _ in range(math.ceil(self.Board.n_positions / 2)):   # main loop, one iteration per move by both players
            for player in self.player_list:
                self.move_counter += 1
                # generate the batch tensor only if one of the player is a neural network
                if 'RecurrentAI' in self.player_types:
                    board = np.concatenate(self.Board.grid.transpose()) # 42 elements from the original grid
                    sequence = np.vstack([sequence, board])[1:]         # we update the sequence with the last move,
                                                                        # removing the first one
                    batch_np[-1] = sequence                             # update the last sequence
                    batch = tf.convert_to_tensor(batch_np)

                # we save the grid before the grid update, so that the choice in the final dataset will
                # correspond to the choice made on that particular grid
                if record:
                    self.game_record['grid'].append((self.Board.grid.transpose()).tolist())
                    self.game_record['player'].append(player.name)
                # the board state (from the Board.update_grid() function) returns -1 if the player
                # makes an invalid move
                board_state = -1
                while board_state == -1:
                    if self.verbose:
                        print(f'Turn #{self.move_counter}: rock to {player.name}!')
                        print('+----------------------------------------------------------------+')
                        self.Board.display_grid()
                        clear_output(wait=True)
                        if (self.pause) & (player.player_type != 'Human') : input()
                    if player.player_type == 'RecurrentAI':
                        player.move(self.Board, batch)
                    else: player.move(self.Board)
                    board_state = self.Board.update_grid(player)
                if record:
                    self.game_record['choice'].append(player.choice)
                self.win = self.Board.check_winner(player)      # check if the game has a winner
                if self.win:
                    if self.verbose:
                        print(f'The WINNER is {player.name}, {player.player_type}!!!')
                        print('+----------------------------------------------------------------+')
                        self.Board.display_grid()
                        if (self.pause) & (player.player_type != 'Human') : input()
                    self.Board.winner = player.player
                    break
            # check if there are no positions left
            if (self.win) | (self.Board.n_pos_left == 0):
                break
        
    def save_game(self):
        df = pd.DataFrame(self.game_record)
        columns = [f'col_{i}' for i in range(self.Board.width)]
        df[columns] = pd.DataFrame(df['grid'].tolist(), index=df.index)
        df.drop(columns='grid', inplace=True)
        return df
    
#---------------------------------------------------------------------------------------------
from tqdm import tqdm
# simulating N AI-AI games
def simulation(n=100, game_type='random-random', model=tf.keras.Sequential(), 
               mean_duration=21, save_json=False, name=None):
    dataset = pd.DataFrame()
    i = 0
    if n <= 1000:
        for i in tqdm(range(n), desc='Simulating', bar_format='{l_bar}{bar:15}{r_bar}{bar:-10b}'):
            game = Game(game_type=game_type, model=model)  # initialize the game environment
            game.play_game(mean_duration=mean_duration)    # play the game
            # repeat if the game ends in a draw
            while not game.win: 
                game = Game(game_type=game_type, model=model)
                game.play_game(mean_duration=mean_duration)                   
            game = game.save_game()        
            game['move'] = game.index 
            game.index = [i] * len(game)
            dataset = pd.concat([dataset, game])
    else:
        j = 0
        for _ in tqdm(range(int(n / 1000)), desc='Simulating', bar_format='{l_bar}{bar:15}{r_bar}{bar:-10b}'):
            df = pd.DataFrame()
            for i in range(1000):
                game = Game(game_type=game_type, model=model)
                game.play_game(mean_duration=mean_duration)
                while not game.win:
                    game = Game(game_type=game_type, model=model)
                    game.play_game(mean_duration=mean_duration)       
                game = game.save_game()
                game['move'] = game.index
                game.index = [i + j] * len(game)
                df = pd.concat([df, game])
            j += i + 1
            dataset = pd.concat([dataset, df])
    # I decided to save the dataframe into .json format in order to preserve the dtype inside the dataframe,
    # since most of the values are arrays that get converted to strings using the default pd.to_csv() function
    if save_json:
        print('Saving: ...')
        # rearrange the columns order
        if name == None:
            dataset.reset_index().to_json(f'simulations/simulation_{game_type}_{n}.json')
        else:
            dataset.reset_index().to_json(f'{name}.json')
        #dataset.to_csv(f'simulations/simulation_{game_type}_{n}.csv')
    return dataset[['player', 'move', 'choice', 'col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6']]

#---------------------------------------------------------------------------------------------
def read_json(path):
    dataset = pd.read_json(path)
    index = dataset['index']
    dataset.drop(columns=['index'], inplace=True)
    dataset.index = index
    return dataset

#---------------------------------------------------------------------------------------------
def save_model(model, path=''): # path example: 'folder/name' - without extension!
    # serialize model to JSON
    model_json = model.to_json()
    with open(f'{path}.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f'{path}.h5')
    print('Model saved succesfully.')
 
from keras.models import model_from_json
def load_model(path):
    # load json and create model
    json_file = open(f'{path}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f'{path}.h5')
    print('Model succesfully loaded.')
    return loaded_model