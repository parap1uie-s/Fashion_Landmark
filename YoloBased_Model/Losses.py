def yolo_loss(args,
              point_num,
              point_classes):
    
    # (None, W, H, point_num, 3+point_classes)
    model_output, true_points = args
    groundtruth_tensor = K.variable(true_points)

    # visiblity
    pred_vis = model_output[:,:,:,:,0]
    true_vis = groundtruth_tensor[:,:,:,:,0]

    # x and y
    pred_position = model_output[:,:,:,:,1:3]
    true_position = groundtruth_tensor[:,:,:,:,1:3]

    # classify
    pred_classprob = model_output[:,:,:,:,3:]
    true_classprob = groundtruth_tensor[:,:,:,:,3:]

    # 

    
    return total_loss