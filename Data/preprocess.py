import numpy as np
from sionna.rt import load_scene, Transmitter, PlanarArray
import cv2
import os
import sys

def get_scene(xml_file):
  scene = load_scene(xml_file)

  scene.tx_array = PlanarArray(
    num_rows = 4,
    num_cols = 4,
    vertical_spacing = 0.5,
    horizontal_spacing = 0.5,
    pattern = "iso",
    polarization="V"
  )
  scene.rx_array = PlanarArray(
    num_rows = 1,
    num_cols = 1,
    vertical_spacing = 0.5,
    horizontal_spacing = 0.5,
    pattern = "iso",
    polarization="V"
  )
  accepted_mat = ["itu_concrete", "itu_very_dry_ground"]
  for obj_name in scene.objects:
    obj = scene.get(obj_name)
    if scene.get(obj_name).radio_material.name not in accepted_mat:
      obj.radio_material = "itu_concrete"

  return scene

def get_city_map(root):
  image = cv2.imread(f"{root}Boundary2dCity.png")
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray[gray<118] = 0
  gray[gray>=118] = 255
  # Find contours in the image
  contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Get the bounding box of the largest contour
  x, y, w, h = cv2.boundingRect(contours[0])

  # Crop the image using the bounding box
  cropped_image = gray[y:y+h, x:x+w]

  return cropped_image

def augment_image(image):
  flipped_vertical = cv2.flip(image, 0)
  flipped_horizontal = cv2.flip(image, 1)
  flipped_both = cv2.flip(image, -1)

  return image, flipped_vertical, flipped_horizontal, flipped_both

def crop_image_with_tx(root, bs, tx_image, building_mask, power, point, w, crop_dim=np.array((512, 512)), stride=80):
    os.makedirs(ROOT+f"/cropped/tx_map", exist_ok=True)
    os.makedirs(ROOT+f"/cropped/power_map", exist_ok=True)
    os.makedirs(ROOT+f"/cropped/city_map", exist_ok=True)

    height, width = tx_image.shape[:2]
    crop_height, crop_width = crop_dim
    stride_x, stride_y = stride, stride

    shift = w//2
    point[1] = height - point[1]
    min_sq, max_sq = point-shift, point+shift
    count = 0

    # Iterate over the image with the specified stride
    for y in range(0, height - crop_height + 1, stride_y):
        for x in range(0, width - crop_width + 1, stride_x):
            # Check if the point is present in the current cropped region
            if x <= min_sq[0] < x + crop_width and x <= max_sq[0] < x + crop_width and y <= min_sq[1] < y + crop_height and y <= max_sq[1] < y + crop_height:
                
                crop_tx = augment_image(
                    cv2.resize(tx_image[y:y + crop_height, x:x + crop_width], (256, 256))
                )
                crop_bld = augment_image(
                    cv2.resize(building_mask[y:y + crop_height, x:x + crop_width], (256, 256))
                )
                crop_p = augment_image(
                    cv2.resize(power[y:y + crop_height, x:x + crop_width], (256, 256))
                )

                cv2.imwrite(root+f"cropped/tx_map/{bs}_{count}_0.png", crop_tx[0])
                cv2.imwrite(root+f"cropped/tx_map/{bs}_{count}_1.png", crop_tx[1])
                cv2.imwrite(root+f"cropped/tx_map/{bs}_{count}_2.png", crop_tx[2])
                cv2.imwrite(root+f"cropped/tx_map/{bs}_{count}_3.png", crop_tx[3])

                cv2.imwrite(root+f"cropped/city_map/{bs}_{count}_0.png", crop_bld[0])
                cv2.imwrite(root+f"cropped/city_map/{bs}_{count}_1.png", crop_bld[1])
                cv2.imwrite(root+f"cropped/city_map/{bs}_{count}_2.png", crop_bld[2])
                cv2.imwrite(root+f"cropped/city_map/{bs}_{count}_3.png", crop_bld[3])

                cv2.imwrite(root+f"cropped/power_map/{bs}_{count}_0.png", crop_p[0])
                cv2.imwrite(root+f"cropped/power_map/{bs}_{count}_1.png", crop_p[1])
                cv2.imwrite(root+f"cropped/power_map/{bs}_{count}_2.png", crop_p[2])
                cv2.imwrite(root+f"cropped/power_map/{bs}_{count}_3.png", crop_p[3])

                count += 1
    return True


if __name__ == "__main__":
    ROOT = "data/"
    TX =  np.load(f"{ROOT}tx_positions.npy")
    CROP_DIM = np.array((512, 512))
    STRIDE = 80
    TX_WIDTH = 8
    FLOOR = -200
    scene = get_scene(ROOT + "USC_3D/USC.xml")

    cm_dim = 900

    city_map = cv2.imread(f"{ROOT}USC_city_map.png")[:, :, 0]

    start, end = int(sys.argv[1]), int(sys.argv[2]) 
    batch_size = end-start

    if end > len(TX):
        print(f"Please enter an end less than number of TX points ({len(TX)})")
    elif start < 0:
        print(f"start should be greater than 0")
    elif batch_size <= 0:
        print(f"end should be greater than start")
    else:
        for i in range(start, end):
            tx = Transmitter(f"tx{i}", TX[i]+[0, 0, 2], [0.0, 0.0, 0.0])
            scene.add(tx)

        cm = scene.coverage_map(
            cm_cell_size=(5.0, 5.0),
            diffraction=True, scattering=True, edge_diffraction=True, max_depth = 1000, num_samples = 8.99*(10**6)
        )
        for i in range(start, end):
            tx_cm = 10.*np.log10(cm[i%batch_size].numpy())
            tx_cm[tx_cm==(-np.inf)] = -255
            tx_cm[tx_cm > 0] = 0
            tx_cm = cv2.resize(tx_cm, city_map.shape)

            tx_cm = cv2.flip(tx_cm, 0)
            tx_cm[tx_cm < FLOOR] = FLOOR
            tx_cm += 255

            tx_cm[city_map < 55] = 0 # for building

            if tx_cm.min() != 0: continue

            tx_pos= np.array(TX[i])+(cm_dim//2) + (50)
            tx_pos = (tx_pos).astype(np.int16)

            tx_map = np.zeros((cm_dim, cm_dim))
            shift = TX_WIDTH//2
            tx_map[cm_dim - tx_pos[1]-shift:cm_dim - tx_pos[1] +shift, tx_pos[0]-shift:tx_pos[0]+shift] = 255

            uncropped =tx_cm.copy()
            bld_filt = tx_cm == 0
            tx_filt = tx_map == 255
            uncropped[bld_filt] = city_map[bld_filt]
            uncropped[tx_filt] = 255
            uncropped = np.clip(uncropped.astype(np.uint8), 0, 255)

            os.makedirs(ROOT+f"/uncropped", exist_ok=True)
            cv2.imwrite(ROOT+f"/uncropped/{i}.png", uncropped)
            crop_image_with_tx(ROOT, f"{i}", tx_map, city_map, tx_cm, tx_pos[:2], w = TX_WIDTH, crop_dim=CROP_DIM, stride=STRIDE)

    print(f"Completed for tx {start} - {end}")