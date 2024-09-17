~~~python
def regroup_reds_dataset(train_path, val_path):
    val_folders = glob.glob(os.path.join(val_path, '*'))
    for folder in val_folders:
        new_folder_idx = int(folder.split('\\')[-1]) + 240
        print(new_folder_idx)
        os.system(f'cp -r {folder} 
                  {os.path.join(train_path, str(new_folder_idx))}')
~~~

获得帧序列：

~~~python
clip_name, frame_name = key.split('/')
# ensure not exceeding the borders
start_frame_idx = center_frame_idx - self.num_half_frames
end_frame_idx = center_frame_idx + self.num_half_frames
# each clip has 100 frames starting from 0 to 99
while (start_frame_idx < 0) or (end_frame_idx > 99):
    center_frame_idx = random.randint(0, 99)
    start_frame_idx = (
        center_frame_idx - self.num_half_frames)
    end_frame_idx = center_frame_idx + self.num_half_frames
    frame_name = f'{center_frame_idx:08d}'
    neighbor_list = list(
        range(center_frame_idx - self.num_half_frames,
              center_frame_idx + self.num_half_frames + 1))
    # random reverse
    if self.random_reverse and random.random() < 0.5:
        neighbor_list.reverse()
#get GT frame
img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
img_bytes = self.file_client.get(img_gt_path, 'gt')
img_gt = imfrombytes(img_bytes, float32=True)
#get neighboring LQ frames
img_lqs = []
for neighbor in neighbor_list:
    img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
img_bytes = self.file_client.get(img_lq_path, 'lq')
img_lq = imfrombytes(img_bytes, float32=True)
img_lqs.append(img_lq)
~~~

裁剪LQ帧和GT帧：

~~~python
h_lq, w_lq, _ = img_lqs[0].shape
h_gt, w_gt, _ = img_gts[0].shape
lq_patch_size = gt_patch_size // scale
# randomly choose top and left coordinates for lq patch
top = random.randint(0, h_lq - lq_patch_size)
left = random.randint(0, w_lq - lq_patch_size)
# crop lq patch
img_lqs = [
    v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
    for v in img_lqs]
# crop corresponding gt patch
top_gt, left_gt = int(top * scale), int(left * scale)
img_gts = [
    v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
    for v in img_gts]
return img_gts, img_lqs
~~~

进行随机翻转和旋转：

~~~python
def augment(imgs):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    imgs = [_augment(img) for img in imgs]
    return imgs
~~~

残差块：

~~~python
class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale
~~~

特征提取模块

~~~python
self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
~~~

对齐模块网络组成

~~~python
self.offset_conv1 = nn.ModuleDict()
self.offset_conv2 = nn.ModuleDict()
self.offset_conv3 = nn.ModuleDict()
self.dcn_pack = nn.ModuleDict()
self.feat_conv = nn.ModuleDict()

for i in range(3, 0, -1):
    level = f'l{i}'
    self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
    if i == 3:
        self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
    else:
        self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.dcn_pack[level] = DCNv2Pack(num_feat,num_feat,3,padding=1,
                                         deformable_groups=deformable_groups)
        if i < 3:
            self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
self.cas_dcnpack = DCNv2Pack(num_feat,num_feat,3,padding=1,
                             deformable_groups=deformable_groups)
self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
~~~

融合模块正向传播

~~~python
feat = self.lrelu(self.feat_fusion(aligned_feat))

attn = self.lrelu(self.spatial_attn1(aligned_feat))
attn_max = self.max_pool(attn)
attn_avg = self.avg_pool(attn)
attn = self.lrelu(
    self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
# levels
attn_level = self.lrelu(self.spatial_attn_l1(attn))
attn_max = self.max_pool(attn_level)
attn_avg = self.avg_pool(attn_level)
attn_level = self.lrelu(
    self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
attn_level = self.upsample(attn_level)

attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
attn = self.lrelu(self.spatial_attn4(attn))
attn = self.upsample(attn)
attn = self.spatial_attn5(attn)
attn_add = self.spatial_attn_add2(
    self.lrelu(self.spatial_attn_add1(attn)))
attn = torch.sigmoid(attn)

# * 2 makes (attn * 2) to be close to 1.
feat = feat * attn * 2 + attn_add
return feat
~~~

