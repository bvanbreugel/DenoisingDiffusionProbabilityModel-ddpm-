from MainCondition import main
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from pytorch_resnet_cifar10.trainer import main as resnet_main



def generate_images(K=5):
    # generate training and test images
    for k in range(0,K):
        main(state = 'generate', seed=k)

def train_downstream(K=5, include_naiveE=True, include_naiveC=True, off=5, train_dir_root='SampledImgsV2_'):

    # Training experiment
    # train resnet on training set per model
    preds = []
    targets = []
    training_naiveS = []

    
    for k in range(K):
        resnet_main(seed=0, train_dir = str(k+off), train_dir_root=train_dir_root)
        res = resnet_main(evaluate=True, seed=0, train_dir = str(k+off), val_dirs = ['real'], train_dir_root=train_dir_root)           
        
        preds.append(res['real']['preds'])
        targets.append(res['real']['targets'])
        if np.any(targets[-1] != targets[0]):
            raise ValueError('targets not equal across runs, somewhere there is randomness')
        training_naiveS.append(accuracy_score(targets[k], np.argmax(preds[k],axis=1)))
    
    
    if include_naiveE:
        training_naiveE = []
        training_naiveE_preds = []
        for k in range(0, K):
            preds_naiveE = [preds[k]]
            for seed in range(1, K):
                resnet_main(seed=seed, train_dir = str(k+off), train_dir_root=train_dir_root)
                preds_naiveE.append(resnet_main(evaluate = True, seed=seed, train_dir = str(k+off), val_dirs = ['real'],train_dir_root=train_dir_root)['real']['preds'])
            training_naiveE.append(accuracy_score(targets[0], np.argmax(np.mean(preds_naiveE,axis=0),axis=1)))
            training_naiveE_preds.append(preds_naiveE)
        training_naiveE_mean = np.mean(training_naiveE)
        training_naiveE_std = np.std(training_naiveE)

    if include_naiveC:
        resnet_main(seed=0, train_dir = [str(i) for i in np.arange(K,dtype=int)+off], train_dir_root=train_dir_root)
        res = resnet_main(evaluate=True, seed=0, train_dir = [str(i) for i in np.arange(K, dtype=int)+off], val_dirs = ['real'], train_dir_root=train_dir_root)
        training_naiveC = accuracy_score(targets[0], np.argmax(res['real']['preds'],axis=1))

    training_naiveS = np.array(training_naiveS)
    training_naiveS_mean = np.mean(training_naiveS)
    training_naiveS_std = np.std(training_naiveS)

    if include_naiveE:
        training_dge = [accuracy_score(targets[0], np.argmax(np.mean([training_naiveE_preds[k][seed] for k in range(K)]))) for seed in range(K)]
        training_dge_mean = np.mean(training_dge)
        training_dge_std = np.std(training_dge)
        train_df = pd.DataFrame({'Naive (S)': [f'{training_naiveS_mean}\pm{training_naiveS_std}'],
                                'Naive (E)': [f'{training_naiveE_mean}\pm{training_naiveE_std}'],
                                'DGE_5':[f'{training_dge_mean}\pm{training_dge_std}']})
    else:
        training_dge = accuracy_score(targets[0], np.argmax(np.mean(preds,axis=0),axis=1))
        train_df = pd.DataFrame({'Naive (S)': [f'{training_naiveS_mean}\pm{training_naiveS_std}'], 
                                'DGE_5':[training_dge]})
    

    
    if include_naiveC:
        train_df['naiveC'] = [training_naiveC]

    print(train_df)
    train_df.to_csv('train_results.csv')
    

    # 
    results_matrix = np.zeros((K,K))
    for k in range(K):
        res = resnet_main(evaluate = True, seed=0, train_dir = str(k+off), val_dirs = [str(i+off) for i in range(K)], train_dir_root=train_dir_root)         
        for l in range(K):
            results_matrix[k,l] = accuracy_score(res[str(l+off)]['targets'], np.argmax(res[str(l+off)]['preds'],axis=1))
    
    print(results_matrix)
    print(training_naiveS)
    eval_naiveS_mean = np.mean((np.diag(results_matrix)-training_naiveS)**2)
    eval_naiveS_std = np.std((np.diag(results_matrix)-training_naiveS)**2)
    eval_dge = np.array([np.mean([results_matrix[i] for i in range(K) if i!=k]) for k in range(K)])
    eval_dge_mean = np.mean((eval_dge- training_naiveS)**2)
    eval_dge_std = np.std((eval_dge - training_naiveS)**2)

    eval_df = pd.DataFrame({'Naive': [f'{eval_naiveS_mean}\pm{eval_naiveS_std}'],
                            'DGE_5':[f'{eval_dge_mean}\pm{eval_dge_std}']})
    print(eval_df)

    # save to disk
    eval_df.to_csv('eval_results.csv')
    return train_df, eval_df
        

if __name__ == '__main__':
    train_downstream(off=5)
