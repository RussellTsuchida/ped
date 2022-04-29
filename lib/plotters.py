import matplotlib 
import matplotlib.pyplot as plt
import gc

def matplotlib_gc():
    plt.cla()
    plt.clf()
    plt.close('all')
    gc.collect()

def matplotlib_config():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['axes.labelsize'] = 30
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 20
    matplotlib.rcParams['ytick.labelsize'] = 20
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"


