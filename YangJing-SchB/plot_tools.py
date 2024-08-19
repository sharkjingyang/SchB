import matplotlib.pyplot as plt
import numpy

def plot_loss_all(llh_record,v_pen_record):
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    axs[0].plot(llh_record, label='likelihood')
    axs[0].set_yscale("log")

    axs[1].plot(v_pen_record, label='v_penalty')
    axs[1].set_yscale("log")

    axs[2].plot(numpy.array(llh_record)*10+numpy.array(v_pen_record), label='totol loss')
    axs[2].set_yscale("log")

    plt.legend()
    plt.show()
    
def plot_loss(llh_record,log_flag=True):

    plt.plot(llh_record, label='likelihood')
    if log_flag:
        plt.yscale("log")
    plt.legend()
    plt.show()