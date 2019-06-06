import argparse

def str2bool(v):
    if v.lower()   in ('yes', 'true', 't', 'y', '1', 'ndiyo'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'hapana'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getArgs():
    parser = argparse.ArgumentParser(description = 'CS 5860 -- Homework 1 -- Thomas Conley',add_help=True)
    parser.add_argument('--verbose',
        help='verbose', 
        type=str2bool, nargs='?',
        default=False,
        ) 
    parser.add_argument('--epochs', 
        help='Epochs or iterations.',
        type=int,
        default=50,
        )
    parser.add_argument('--model',
        help = 'word2vec or glove',
        type=str,
        default = 'glove',
        )

    args = parser.parse_args()
    args.model = args.model.lower() 
    return args

def getFileUrl(filepath,cachepath='.'):
    from keras.utils.data_utils import get_file
    from os import path

    _, filename = path.split(filepath)

    cachefile = path.join(cachepath,filename)
    if not path.exists(cachefile):
        # _, filepart = path.split(filepath)
        fpath = get_file(
            filename,
            origin=filepath,
            # default is relative to .keras directory in ~
            cache_subdir= cachepath,
            )
    return cachefile

class colors:
    green = '\033[92m'
    red = '\033[91m'
    close = '\033[0m'

def tprint(string, value, threshold):
    import sys
    end='\n'
    file = sys.stdout
    if value > threshold:
        print(colors.red + repr(string) + colors.close, file=file, end=end, flush=True)
    else:
        print(colors.green + repr(string) + colors.close, file=file, end=end, flush=True)

def eprint(string,end='\n'):
    print(colors.green + repr(string) + colors.close, file=sys.stderr, end=end, flush=True)

def plot(dir,tag,epochs):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cbook as cbook
    import csv
    import sys
    from os import path

    filename = path.join(dir,'hello-{}-{}.dat'.format(tag,epochs))

    CCONTROL = 'red'
    CSHUFFLE = 'black'
    lines = [] 
    # acc,loss,val_acc,val_loss
    lines.append(['accuracy (training)','-',])
    lines.append(['loss (training)',':',])
    lines.append(['accuracy (validation)','-',])
    lines.append(['loss (validation)',':',])
    # line=0
    files = [filename]

    # for each column that you want to plot
    for file in files:
        for col in [1]: #
            x = []
            y = []
            with open(file,'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                next(plots) # skip header
                for i,row in enumerate(plots):
                    if i != 0:
                        x.append(i)
                        y.append(float(row[col]))

            plt.plot(x,y, linewidth=2)
            # line = line+1

    axes = plt.gca()
    # plt.grid()
    # axes.set_ylim([0,10])
    # axes.set_xlim([0,1000])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title(filename)
    # plt.legend(loc='center right')

    plt.savefig(filename.replace('.dat','.png'))
    # plt.show()

def showMatrix(vocabulary, embedding_matrix):
    tabhead = '{:>4s} -> {:5s} -> {}'.format('i','word','embedding_matrix[i]')
    tabline = '{:>4s} -> {:5s} -> {}'.format('-','----','-'*len('embedding_matrix[i]'))
    print(); print(tabhead); print(tabline)
    for i,word in enumerate(vocabulary):
        print('{:4d} -> {:5s} -> {}'.format(i,word,embedding_matrix[i]))
    print()
