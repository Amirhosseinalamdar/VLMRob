import json
import pandas as pd
from typing import List

def load_coco_entities(path: str, all_entities: bool = True) -> List[str]:
    # COCO Vocabulary

    with open(path, 'r') as infile:
        entities = json.load(infile)       # List [category1, category2, ...]
    
    if all_entities:
        entities = [entity.lower().strip() for entity in entities]
    else:
        entities = [entity.lower().strip() for entity in entities if len(entity.split()) == 1]
    entities.sort()  # sort

    return entities

def load_open_image_entities(path: str, all_entities: bool = True) -> List[str]:
    # Open Image Vocabulary

    open_images = pd.read_csv(path)                     # 601x2, i.e., [LabelName, DisplayName]
    open_image_entities = list(open_images.DisplayName) # list
    
    for i in range(len(open_image_entities)):
        entity = open_image_entities[i].lower().strip()
        if entity[-1] == ')':
            entity = entity[:entity.find('(')].strip()
        open_image_entities[i] = entity

    if all_entities:
        entities = [entity for entity in open_image_entities]
    else:
        entities = [entity for entity in open_image_entities if len(entity.split()) == 1]
    entities.sort()  # sort

    return entities

def load_vinvl_vgoi_entities(path: str, all_entities: bool = True) -> List[str]:

    with open(path, 'r') as infile: 
        vgoi_entities = json.load(infile) # dictionary = {str: int}

    if all_entities:
        entities = [entity.lower().strip() for entity in vgoi_entities]
    else:
        entities = [entity.lower().strip() for entity in vgoi_entities if len(entity.split()) == 1]
    entities.sort()  # sort

    return entities

def load_entities_text(name_of_entities: str, path_of_entities: str, all_entities: bool = True) -> List[str]:
    
    if name_of_entities == 'coco_entities':
        return load_coco_entities(path_of_entities, all_entities)

    if name_of_entities == 'open_image_entities':
        return load_open_image_entities(path_of_entities, all_entities)

    if name_of_entities == 'vinvl_vgoi_entities':
        return load_vinvl_vgoi_entities(path_of_entities, all_entities)

    print('The entities text fails to load!')

def count_object_in_gt(annotations, entities):

    results = {}
    for annotation in annotations:
        captions = annotation['captions']
        prediction = annotation['prediction']
        gt_objects = []
        pred_objects = []
        for caption in captions:
            caption = caption.lower()
            for entity in entities:
                if entity in caption:
                    gt_objects.append(entity)

        prediction = prediction.lower()
        for entity in entities:
            if entity in prediction:
                pred_objects.append(entity)
        
        gt_objects = list(set(gt_objects))
        pred_objects = list(set(pred_objects))

        results[annotation['image_name']] = {}
        results[annotation['image_name']]['split'] = annotation['split']
        results[annotation['image_name']]['gt_objects'] = gt_objects
        results[annotation['image_name']]['pred_objects'] = pred_objects

    return results


if __name__ == '__main__':
    
    
    # entities_coco = load_entities_text('coco_entities', './coco_categories.json')
    # entities_oi = load_entities_text('open_image_entities', './oidv7-class-descriptions-boxable.csv')
    entities_vgoi = load_entities_text('vinvl_vgoi_entities', './vgcocooiobjects_v1_class2ind.json')
    entities = entities_vgoi
    entities.sort()

    path_of_predictions = './overall_generated_captions.json'

    with open(path_of_predictions, 'r') as infile:
        annotations = json.load(infile)
    objects = count_object_in_gt(annotations, entities)

    sum_pred = [0, 0, 0]
    acc_pred = [0, 0, 0]
    for i in range(len(annotations)):
        image_id = f'{i}.jpg'
        split = objects[image_id]['split']
        gt_objects = objects[image_id]['gt_objects']
        pred_objects = objects[image_id]['pred_objects']
        
        if split == 'in_domain':
            sum_pred[0] += len(pred_objects)
            for o in pred_objects:
                if o in gt_objects:
                    acc_pred[0] += 1
        elif split == 'near_domain':
            sum_pred[1] += len(pred_objects)
            for o in pred_objects:
                if o in gt_objects:
                    acc_pred[1] += 1
        elif split == 'out_domain':
            sum_pred[2] += len(pred_objects)
            for o in pred_objects:
                if o in gt_objects:
                    acc_pred[2] += 1

    in_acc_pred = acc_pred[0] / sum_pred[0]
    near_acc_pred = acc_pred[1] / sum_pred[1]
    out_acc_pred = acc_pred[2] / sum_pred[2]

    print(f'pred accuracy in in domain: {100 * in_acc_pred:.4}%')
    print(f'pred accuracy in near domain: {100 * near_acc_pred:.4}%')
    print(f'pred accuracy in out domain: {100 * out_acc_pred:.4}%')