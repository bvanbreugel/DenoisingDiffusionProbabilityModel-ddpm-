from DiffusionFreeGuidence.TrainCondition import train, eval, generate
import os

def main(model_config=None, **kwargs):
    modelConfig = {
        "state": "gen", # or eval
        "epoch": 70,
        "batch_size": 80,
        "batch_size_generation": 128,
        "T": 100,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./CheckpointsCondition",
        "training_load_weight": None,
        "sampled_dir": "./SampledImgs",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8,
        "seed": 'range', 
        "dataset": "CIFAR10",
        "samples_per_class": 6000,
        "p_train": 5/6,

    }

    
    if model_config is not None:
        modelConfig = model_config
    
    modelConfig["test_load_weight"] = f"ckpt_{modelConfig['epoch']-1}_.pt"
                     
    if modelConfig["T"] == 100:
        modelConfig['sampled_dir']+= 'V2_'
        modelConfig['save_dir']+= 'V2_'    
    
    if kwargs is not None:
        for key, value in kwargs.items():
            modelConfig[key] = value
    
    if modelConfig['seed'] == 'range':
        seeds = range(5,10)
    else:
        seeds = [modelConfig['seed']]

    save_dir = modelConfig['save_dir']
    sampled_dir = modelConfig['sampled_dir']
    for seed in seeds:
        modelConfig['seed'] = seed
        if modelConfig['dataset'] == 'CIFAR10':
            modelConfig['save_dir'] = save_dir + str(modelConfig['seed']) + '/'
            modelConfig['sampled_dir'] = sampled_dir+ str(modelConfig['seed']) + '/'
            modelConfig['num_classes'] = 10
        elif modelConfig['dataset'] == 'CIFAR100':
            modelConfig['save_dir'] = save_dir + str(modelConfig['seed']) + 'CIFAR100/'
            modelConfig['sampled_dir'] = sampled_dir + str(modelConfig['seed']) + 'CIFAR100/'
            modelConfig['num_classes'] = 100

        os.makedirs(modelConfig['save_dir'], exist_ok=True)
        os.makedirs(modelConfig['sampled_dir'], exist_ok=True)

        if modelConfig["state"] == "train":
            train(modelConfig)
        elif modelConfig["state"] == "eval":
            eval(modelConfig)
        elif modelConfig["state"] == "gen":
            generate(modelConfig)
        elif modelConfig["state"] == "traingen":
            train(modelConfig)
            generate(modelConfig)


if __name__ == '__main__':
    main()
