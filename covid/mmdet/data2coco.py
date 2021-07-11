from tqdm.auto import tqdm 

def get_coco_dict(df): 
    coco = {
        'annotations': [], 
        'images': [], 
        'categories': [
            {'supercategory': 'opacity', 'name': 'opacity', 'id': 1},     
        ]
    }
    annot_id = 0
    for i, row in tqdm(df.iterrows()): 
        # Hack for None filenames
        if row.coco_file_name == None: 
            continue
        image_info = {
            'file_name': row.coco_file_name, 
            'id': i, 
            'width': row.img_width, 
            'height': row.img_height, 
        }
        # SKIP Boxes without annotations
        if len(row.boxes) == 0: 
            continue
        for j, bbox in enumerate(row.boxes): 
            x_min, y_min, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            annot = {
                'id': annot_id, 
                'image_id': i, 
                'ignore': 0, 
                'area': bbox['width'] * bbox['height'], 
                'bbox': [x_min, y_min, width, height], 
                'is_crowd': 0, 
                'category_id': 1, 
            }
            annot_id += 1
            coco['annotations'].append(annot)
        coco['images'].append(image_info)
    return coco