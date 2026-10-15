"""
bratlab_dataset.py - 2채널 입력 버전
stations_position.txt 컬럼: [row1, col1, row2, col2, row3, col3]
"""
import os, random
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

class BRATLabPaths:
    def __init__(self, root, freq=4750):
        self.root=root; self.freq=freq
        self.rm_dir=os.path.join(root,f'receivedpower_{freq}MHz_mat')
        self.bp_dir=os.path.join(root,'buildings_position')
        self.sta_file=os.path.join(root,'stations_position.txt')
    def rm_path(self,n): return os.path.join(self.rm_dir,f'TOTAL_REC_{self.freq}MHz_{n}.mat')
    def bp_path(self,n): return os.path.join(self.bp_dir,f'buildings_{n}.mat')

def load_stations(sta_file):
    with open(sta_file) as f: lines=f.readlines()
    return np.array([[int(x) for x in l.strip().split()] for l in lines if l.strip()])

def make_building_mask(bp, target_h, target_w):
    if bp.shape==(target_h,target_w): return (bp>127).astype(np.float32)
    img=Image.fromarray(bp).resize((target_w,target_h),Image.NEAREST)
    return (np.array(img)>127).astype(np.float32)

def make_tx_map(shape, tx_positions, sigma=4.0):
    H,W=shape; tx_map=np.zeros((H,W),dtype=np.float32)
    for (r,c) in tx_positions:
        tx_map[min(max(int(r),0),H-1), min(max(int(c),0),W-1)]=1.0
    tx_map=gaussian_filter(tx_map,sigma=sigma)
    tx_map/=(tx_map.max()+1e-8)
    return tx_map

def make_gt(radio_map, min_dbm=-90.0, max_dbm=0.0):
    gt=np.zeros_like(radio_map,dtype=np.float32)
    mask=radio_map!=0
    gt[mask]=np.clip((radio_map[mask]-min_dbm)/(max_dbm-min_dbm),0.0,1.0)
    return gt

class BRATLabDataset(Dataset):
    def __init__(self, root, freq=4750, img_size=256, augment=False,
                 min_dbm=-90.0, max_dbm=0.0, n_samples=2000):
        self.paths=BRATLabPaths(root,freq)
        self.stations=load_stations(self.paths.sta_file)
        self.img_size=img_size; self.augment=augment
        self.min_dbm=min_dbm; self.max_dbm=max_dbm
        self.indices=[]; skipped=[]
        for n in range(1,n_samples+1):
            if not (os.path.exists(self.paths.rm_path(n)) and os.path.exists(self.paths.bp_path(n))): continue
            try:
                mat_rm=sio.loadmat(self.paths.rm_path(n))
                rm_key=next(k for k in mat_rm if not k.startswith('__'))
                rm=mat_rm[rm_key].astype(np.float64); H,W=rm.shape
                if np.isnan(rm).any() or W<50 or H<50: skipped.append(n); continue
            except: skipped.append(n); continue
            self.indices.append(n)
        if skipped: print(f"[BRATLabDataset] 비정상 샘플 {len(skipped)}개 제외")
        print(f"[BRATLabDataset] {len(self.indices)}개 유효 샘플 | freq={freq}MHz | img_size={img_size} | augment={augment}")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        n=self.indices[idx]
        mat_rm=sio.loadmat(self.paths.rm_path(n))
        rm_key=next(k for k in mat_rm if not k.startswith('__'))
        rm=mat_rm[rm_key].astype(np.float64)
        mat_bp=sio.loadmat(self.paths.bp_path(n))
        bp_key=next(k for k in mat_bp if not k.startswith('__'))
        bp=mat_bp[bp_key]
        sta=self.stations[n-1]; H,W=rm.shape
        # stations: [row1, col1, row2, col2, row3, col3]
        tx_positions=[
            (int(sta[0]), int(sta[1])),
            (int(sta[2]), int(sta[3])),
            (int(sta[4]), int(sta[5])),
        ]
        ch0=make_building_mask(bp,H,W)
        ch1=make_tx_map((H,W),tx_positions)
        x=torch.from_numpy(np.stack([ch0,ch1],axis=0)).float()
        y=torch.from_numpy(make_gt(rm,self.min_dbm,self.max_dbm)[np.newaxis]).float()
        x=TF.resize(x,[self.img_size,self.img_size],interpolation=TF.InterpolationMode.BILINEAR,antialias=True)
        y=TF.resize(y,[self.img_size,self.img_size],interpolation=TF.InterpolationMode.BILINEAR,antialias=True)
        if self.augment:
            if random.random()>0.5: x,y=TF.hflip(x),TF.hflip(y)
            if random.random()>0.5: x,y=TF.vflip(x),TF.vflip(y)
            k=random.choice([0,1,2,3])
            if k>0: x=torch.rot90(x,k,dims=[1,2]); y=torch.rot90(y,k,dims=[1,2])
        return x,y,n

# def get_dataloaders(root, freq=4750, img_size=256, batch_size=8,
#                     num_workers=4, split=(0.7,0.1,0.2), seed=42):
#     full_ds=BRATLabDataset(root,freq=freq,img_size=img_size,augment=False)
#     n=len(full_ds); n_tr=int(n*split[0]); n_va=int(n*split[1]); n_te=n-n_tr-n_va
#     g=torch.Generator().manual_seed(seed)
#     tr_ds,va_ds,te_ds=random_split(full_ds,[n_tr,n_va,n_te],generator=g) # random_split으로 만들어진 train/val/test은 같은 full_ds를 공유하므로, 이 방식이면 train뿐만 아니라 validation/test에도 augmentation이 적용될 수 있다. 따라서, 고정된 테스트셋 399개 샘플에서 평가했다고 하기 어려워진다.
#     tr_ds.dataset.augment=True
#     kw=dict(num_workers=num_workers,pin_memory=True)
#     print(f"[DataLoader] Train:{n_tr} | Val:{n_va} | Test:{n_te}")
#     return (DataLoader(tr_ds,batch_size,shuffle=True,**kw),
#             DataLoader(va_ds,batch_size,shuffle=False,**kw),
#             DataLoader(te_ds,1,shuffle=False,**kw))


def get_dataloaders(root, freq=4750, img_size=256, batch_size=8,
                    num_workers=4, split=(0.7, 0.1, 0.2), seed=42):
    from torch.utils.data import Subset, DataLoader
    import torch

    # split index 생성을 위한 기준 dataset
    base = BRATLabDataset(root, freq=freq, img_size=img_size, augment=False)

    n = len(base)
    n_tr = int(n * split[0])
    n_va = int(n * split[1])
    n_te = n - n_tr - n_va

    # 기존 random_split과 같은 방식
    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=split_generator).tolist()

    tr_idx = indices[:n_tr]
    va_idx = indices[n_tr:n_tr + n_va]
    te_idx = indices[n_tr + n_va:]

    # split별 dataset 객체 분리
    tr_base = BRATLabDataset(root, freq=freq, img_size=img_size, augment=True)
    va_base = BRATLabDataset(root, freq=freq, img_size=img_size, augment=False)
    te_base = BRATLabDataset(root, freq=freq, img_size=img_size, augment=False)

    tr_ds = Subset(tr_base, tr_idx)
    va_ds = Subset(va_base, va_idx)
    te_ds = Subset(te_base, te_idx)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    loader_generator = torch.Generator().manual_seed(seed + 1000)
    kw = dict(
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    print(f"[DataLoader-Fixed] Train:{n_tr} | Val:{n_va} | Test:{n_te}")
    print("[DataLoader-Fixed] augment: train=True, val=False, test=False")

    return (
        DataLoader(
            tr_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
            **kw,
        ),
        DataLoader(va_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(te_ds, batch_size=1, shuffle=False, **kw),
    )


if __name__=="__main__":
    import matplotlib.pyplot as plt
    ROOT="/home/james4510/OutdoorRadioMapPrediction/data"
    ds=BRATLabDataset(ROOT,freq=4750,img_size=256,n_samples=10)
    x,y,n=ds[0]
    sta=ds.stations[n-1]; H_orig,W_orig=328,369
    scale_h=256/H_orig; scale_w=256/W_orig
    tx_scaled=[
        (int(sta[0]*scale_h), int(sta[1]*scale_w)),
        (int(sta[2]*scale_h), int(sta[3]*scale_w)),
        (int(sta[4]*scale_h), int(sta[5]*scale_w)),
    ]
    print(f"\n샘플 번호  : {n}")
    print(f"Input shape: {x.shape}  [{x.min():.3f}, {x.max():.3f}]")
    print(f"GT shape   : {y.shape}  [{y.min():.3f}, {y.max():.3f}]")
    print(f"Tx 3개 위치: {tx_scaled}")
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    panels=[(x[0],'gray','Ch0: Building Mask'),(x[1],'hot','Ch1: Tx Position (3 Tx)'),(y[0],'jet','GT: Radio Map')]
    for ax,(d,cm,t) in zip(axes,panels):
        im=ax.imshow(d.numpy(),cmap=cm,origin='upper')
        ax.set_title(t,fontsize=10,fontweight='bold')
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        for (r,c) in tx_scaled: ax.plot(c,r,'w*',markersize=12,markeredgecolor='black',markeredgewidth=0.5)
        ax.axis('off')
    plt.suptitle(f'BRAT-Lab Sample #{n} | 2ch Input',fontsize=13,fontweight='bold')
    plt.tight_layout()
    plt.savefig('dataset_verify.png',dpi=150,bbox_inches='tight')
    print("dataset_verify.png 저장 완료")
