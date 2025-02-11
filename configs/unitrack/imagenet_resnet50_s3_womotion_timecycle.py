_base_ = [
    '../mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic_custom_single_video_test.py'
]

tracker_cfg = dict(
    common=dict(
        exp_name="imagenet_resnet50_s3_womotion_timecycle",
        model_type="imagenet50",
        remove_layers=['layer4'],
        im_mean=[0.485, 0.456, 0.406],
        im_std=[0.229, 0.224, 0.225],
        nopadding=False,
        resume='./checkpoints/UniTrack/timecycle.pth',
        down_factor=8,
        infer2D=True,
        workers=4,
        gpu_id=1,
        device='cuda'
    ),
    mots=dict(
        obid='COSTA_st',
        save_videos=True,
        save_images=True,
        test=False,
        track_buffer=300,
        nms_thres=0.4,
        conf_thres=0.5,
        iou_thres=0.5,
        prop_flag=False,
        max_mask_area=300,
        dup_iou_thres=0.15,
        confirm_iou_thres=0.7,
        first_stage_thres=0.7,
        feat_size=[4,10],
        use_kalman=True,
        asso_with_motion=False,
        motion_lambda=1, 
        motion_gated=False,
    ),
    frame_rate=5, # for pvsg
)
