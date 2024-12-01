import pandas as pd
import matplotlib.pyplot as plt

def plot_unstructured_accuracy_sparcity(data: pd.DataFrame) -> None:
    # Visualise the results to choose the best sparsity level to reduce maximum loss and maintain maximum accuracy
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plt.grid(None)
    ax2 = ax1.twinx()
    plt.grid(None)
    plt.title(
        'Validation accuracy & loss as a function of k% Sparsity\nfor a `VGG16` dataset trained on `horses_or_humans')
    ax1.plot(data['sparsity'].values,
             data['val_accuracy'].values,
             '#008fd5', linestyle='-', label='Pruning Accuracy')
    ax2.plot(data['sparsity'].values,
             data['val_loss'].values,
             '#fc4f30', linestyle='-', label='Pruning Loss')

    ax1.set_ylabel('Accuracy (%)', color='#008fd5')
    ax2.set_ylabel('Loss (sparse categorical crossentropy)', color='#fc4f30')
    ax1.set_xlabel('k% Sparsity')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2);
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), shadow=True, ncol=2);
    plt.savefig('Horses_or_Humans_sparsity_comparisons.png')