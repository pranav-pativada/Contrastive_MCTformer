import argparse
import subprocess

if __name__ == '__main__':
    
    args = argparse.ArgumentParser()
    args.add_argument('--weights', type=str, default='1*1')
    
    # Strip Weights
    weights = args.weights.strip()
    weight_list = []
    for weight in weights:
        clr_weight, cls_weight = weight.strip('*')
        weight_list.append((clr_weight, cls_weight))

    # Cache process args
    process_path = 'automate.sh'
    cache = None
    with open(process_path, 'r') as f:
        cache = f.readlines() 

    for pair in weight_list: 
        for arg_index in range(len(cache)):
            if "cls_weight" in cache[arg_index]:
                cache[arg_index] = "cls_weight" + str(pair[0]) + " \\"
            if "clr_weight" in cache[arg_index]:
                cache[arg_index] = "clr_weight" + str(pair[1]) + " \\"
                 