from keras import backend as K

def yolo_loss(args, point_num, point_classes,grid_length):
    
    # (None, W, H, point_num, 3+point_classes)
    model_output, true_points = args
    groundtruth_tensor = K.variable(true_points)
    grid_length_tensor = K.constant(grid_length)

    # visiblity
    pred_vis = model_output[...,0]
    true_vis = groundtruth_tensor[...,0]
    visiblity_loss = K.binary_crossentropy(true_vis,pred_vis)
    visiblity_loss = K.sum(visiblity_loss)

    # x and y
    pred_position = model_output[...,1:3] * grid_length_tensor
    true_position = groundtruth_tensor[...,1:3]
    vis_mask = K.repeat_elements(  K.expand_dims(true_vis,axis=-1), 2, axis=-1)
    dist_loss = K.sum(K.square(  (pred_position - true_position)*vis_mask ), axis=-1)
    dist_loss = K.sum(dist_loss)

    # classify
    pred_classprob = model_output[...,3:]
    true_classprob = groundtruth_tensor[...,3:]
    classify_loss = K.categorical_crossentropy(true_classprob, pred_classprob) * true_vis
    classify_loss = K.sum(classify_loss)

    total_loss = classify_loss + dist_loss + visiblity_loss
    return total_loss