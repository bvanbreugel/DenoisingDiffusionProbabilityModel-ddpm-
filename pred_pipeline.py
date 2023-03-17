from MainCondition import main
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from pytorch_resnet_cifar10.trainer import main



def generate_images(K=5):
    # generate training and test images
    for k in range(0,K):
        main(state = 'generate', seed=k)

def train_downstream(K=5, include_ensemble=False):

    # Training experiment
    # train resnet on training set per model
    preds = []
    targets = []
    training_naiveS = []

    for k in range(0, K):
        main(state = 'train', seed=0, train_dir = str(k))
        res = main(state = 'eval', seed=0, train_dir = str(k), val_dirs = ['real'])           
        
        preds.append(res['real']['preds'])
        targets.append(res['real']['targets'])
        if targets[-1]!=targets[0]:
            raise ValueError('targets not equal across runs, somewhere there is randomness')
        training_naiveS.append(accuracy_score(targets[k], np.argmax(preds[k],axis=1)))
    
    
    if include_ensemble:
        training_ensemble = []
        for k in range(0, K):
            preds_ens = [preds[k]]
            for seed in range(1, K):
                main(state = 'train', seed=k, train_dir = str(k), ensemble=True)
                preds_ens.append(main(state = 'eval', seed=seed, train_dir = str(k), val_dirs = ['real'])['real']['preds'])
            training_ensemble.append(accuracy_score(targets[0], np.argmax(np.mean(preds_ens,axis=0),axis=1)))
        training_ensemble_mean = np.mean(training_ensemble)
        training_ensemble_std = np.std(training_ensemble)
            
    training_dge = accuracy_score(targets[0], np.argmax(np.mean(preds,axis=0),axis=1))
    training_naiveS_mean = np.mean(training_naiveS)
    training_naiveS_std = np.std(training_naiveS)


    train_df = pd.DataFrame({'Naive (S)': f'{training_naiveS_mean}\pm{training_naiveS_std}', 
                             'DGE_5':training_dge})
    if include_ensemble:
        train_df['Ensemble'] = f'{training_ensemble_mean}\pm{training_ensemble_std}'

    print(train_df)


    # 
    results_matrix = np.zeros((K,K))
    for k in range(K):
        res = main(state = 'eval', seed=0, train_dir = str(k), val_dirs = [str(i) for i in range(K)])         
        for l in range(K):
            results_matrix[k,l] = accuracy_score(targets[l], np.argmax(res[l]['preds'],axis=1))
    
    print(results_matrix)
    eval_naiveS_mean = np.mean(np.diag(results_matrix)-training_naiveS)
    eval_naiveS_std = np.std(np.diag(results_matrix)-training_naiveS)
    eval_dge = [np.mean([results_matrix[i] for i in range(K) if i!=k]) for k in range(K)]
    eval_dge_mean = np.mean(eval_dge)
    eval_dge_std = np.std(eval_dge)

        

if __name__ == '__main__':
    generate_images()
