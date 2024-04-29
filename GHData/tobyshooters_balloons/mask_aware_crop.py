import albumentations as albu
import albumentations.augmentations.functional as F

class MaskAwareRandomSizedCrop(albu.DualTransform):
    def __init__(self, min_max_height, height, width, always_apply=False, p=1.0):
        super(MaskAwareRandomSizedCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.min_max_height = min_max_height

    def apply(self, img, min_x, max_x, min_y, max_y, **params):
        crop = img[min_y:max_y, min_x:max_x]
        resized_crop = aF.resize(crop, self.height, self.width)
        return resized_crop

    def get_params_dependent_on_targets(self, params):
        """ Image and mask are (H, W, 3), and (H, W) np.array's in (0, 255) """

        if np.sum(params["mask"]) == 0:
            H, W = params["mask"].shape
            return { "min_x": 0, "max_x": W, "min_y": 0, "max_y": H }

        # Get bbox around mask
        xs = np.any(params["mask"], axis=0)
        ys = np.any(params["mask"], axis=1)
        mask_min_x, mask_max_x = np.where(xs)[0][[0, -1]]
        mask_min_y, mask_max_y = np.where(ys)[0][[0, -1]]

        # Crop size from min_max_height and bbox
        # Crop should be, if possible, at least the size of bbox
        bbox_sz = max(mask_max_x - mask_min_x, mask_max_y - mask_min_y)
        lower_bound = min(max(bbox_sz, self.min_max_height[0]), self.min_max_height[1])
        crop_size =  np.random.randint(lower_bound, self.min_max_height[1] + 1)

        # Choose center of crop, close to bbox
        # Crop center should avoid going over the border
        H, W = params["mask"].shape[:2]
        dd = crop_size // 2

        if mask_max_x < dd:
            cx = dd
        elif mask_min_x > W - dd:
            cx = W - dd
        else:
            cx = np.random.randint(max(mask_min_x, dd), min(mask_max_x, W - dd) + 1)

        if mask_max_y < dd:
            cy = dd
        elif mask_min_y > H - dd:
            cy = H - dd
        else:
            cy = np.random.randint(max(mask_min_y, dd), min(mask_max_y, H - dd) + 1)

        # Get a crop region
        params = { "min_x": cx - dd, "max_x": cx + dd, "min_y": cy - dd, "max_y": cy + dd }

        return params

    @property
    def targets_as_params(self):
        return ["mask"]

