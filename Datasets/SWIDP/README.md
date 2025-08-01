## [SWIDP: Integrated Workflow for Dataset Construction](../SWIDP/)  

SWIDP provides a fully modular pipeline for constructing large-scale surface-wave dispersion curve datasets:  

1. **Collect and standardize geological models**  
   Aggregate 2D/3D geological–geomorphological models from public databases and literature, and unify data format through cleaning, parameter conversion, and normalization.  

2. **Build 1D velocity models**  
   Extract representative 1D shear-wave velocity profiles, de-duplicate, optimize thin layers, interpolate to uniform thickness, and complete missing parameters (e.g., \(v_p\), density).  

3. **Augment for geological diversity**  
   Apply perturbation-based and generative-model-based enhancements to improve geological diversity and boundary complexity coverage.  

4. **Forward modeling of dispersion curves**  
   Efficiently compute large-scale surface-wave dispersion curves (using an optimized **Disba**-based solver with parallelization) across diverse period ranges.  

```mermaid
flowchart LR
    A[Collect & Standardize Geological Models] --> B[Build 1D Velocity Models]
    B --> C[Augment for Geological Diversity]
    C --> D[Forward Modeling of Dispersion Curves]
```

---

### Example 1: Building the **OpenSWI-shallow** Dataset

```python
import numpy as np
import sys
sys.path.append("OpenSWI/Datasets/OpenSWI/")
from SWIDP.process_1d_shallow import augment_workflow
from SWIDP.dispersion import (
    generate_mixed_samples, calculate_dispersion,
    transform_vp_to_vs, transform_vs_to_vel_model
)
from p_tqdm import p_map

# Step 1: Load 1D velocity model n x (depth, vp)
depth_vp = np.loadtxt(
    "./OpenSWI/Datasets/OpenSWI/Datasets-Construction/OpenSWI-shallow/0.2-10s-Aug/vp_demo.txt"
)
depth = depth_vp[:, 0]
vp = depth_vp[:, 1]

# Step 2: Convert vp to vs
vs = transform_vp_to_vs(vp)

# Step 3: Augment vs models
augment_nums = 100
vs_augmented = augment_workflow(
    vs, depth,
    perturb_num=augment_nums,
    vs_perturbation=0.05,           # relative ratio
    thickness_perturbation=0.1,     # relative ratio
    vel_threshold=0.05,             # km/s
    thickness_threshold=0.01,       # km
    min_layers_num=3,
    smooth_vel=False,
    smooth_nodes=10
)

# Step 4: Build full velocity models (depth, vp, vs, rho)
vel_model_augmented = p_map(
    transform_vs_to_vel_model,
    list(vs_augmented),
    [depth] * len(vs_augmented)
)

# Step 5: Generate dispersion curves [t, phase velocity, group velocity]
t = generate_mixed_samples(
    num_samples=100, start=0.2, end=10,
    uniform_num=50, log_num=20, random_num=30
)
disp = p_map(
    calculate_dispersion,
    vel_model_augmented,
    [t] * len(vel_model_augmented)
)
```

Details of this example can be found at

| Dataset Type    | Description                         | Link                                                                                      |
|-----------------|-------------------------------------|-------------------------------------------------------------------------------------------|
| **OpenSWI-shallow Example** | Example notebook for the OpenSWI dataset | [OpenSWI-shallow-example.ipynb](../Datasets-Construction/OpenSWI-shallow/0.2-10s-Aug/00_OpenSWI-shallow-example.ipynb) |
| **Flat**        | Flat velocity model construction    | [OpenFWI-FlatVel-A.ipynb](../Datasets-Construction/OpenSWI-shallow/0.2-10s-Aug/01_1_OpenFWI-FlatVel-A.ipynb) |
| **Flat-Fault**  | Flat velocity model with fault      | [OpenFWI-FlatFault-A.ipynb](../Datasets-Construction/OpenSWI-shallow/0.2-10s-Aug/01_2_OpenFWI-FlatFault-A.ipynb) |
| **Fold**        | Fold velocity model construction    | [OpenFWI-CurveVel-A.ipynb](../Datasets-Construction/OpenSWI-shallow/0.2-10s-Aug/01_3_OpenFWI-CurveVel-A.ipynb) |
| **Fold-Fault**  | Fold velocity model with fault      | [OpenFWI-CurveFault-A.ipynb](../Datasets-Construction/OpenSWI-shallow/0.2-10s-Aug/01_4_OpenFWI-CurveFault-A.ipynb) |
| **Field**       | Field data modeling                 | [OpenFWI-Style-A.ipynb](../Datasets-Construction/OpenSWI-shallow/0.2-10s-Aug/01_5_OpenFWI-Style-A.ipynb) |


---

### Example 2: Building the **OpenSWI-deep** Dataset

```python
import numpy as np
import sys
sys.path.append("../../../")
from SWIDP.process_1d_deep import *
from SWIDP.dispersion import generate_mixed_samples,calculate_dispersion,transform_vs_to_vel_model
from p_tqdm import p_map

# step1: get 1d velocity model (vp model or vs)
depth_vs = np.loadtxt("../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/vs_demo.txt")
depth = depth_vs[:,0]
vs = depth_vs[:,1]

# step2-1: remove the thin sandwidth layer
vs = combine_thin_sandwich(vs,
                            depth,
                            thickness_threshold=12, # km
                            uniform_thickness=1,    # km (thickness_threshold/uniform_thickness) = max_check_layers
                            gradient_threshold=0.05, # km/s (gradient_threshold)
                            return_idx=False
                            )

# step2-2: smooth the vs model (selectable)
vs = smooth_vs_by_node_interp(vs,
                            depth,
                            n_nodes=20,
                            method="pchip"
                            )

# step3: find moho index
moho_idx = find_moho_depth(vs,
                           depth,
                           [5,90], # range of moho index
                           gradient_search=False,
                           gradient_threshold=0.01)

# step4: augment the vs model
perturb_nums = 100
vs_augmented = p_map(augment_crust_moho_mantle,
                    [vs]*perturb_nums,
                    list(depth.reshape(1,-1))*perturb_nums,
                    [moho_idx]*perturb_nums,
                    [[-0.1,0.1]]*perturb_nums, # relative ratio
                    [[3,8]]*perturb_nums,     # nodes for crust
                    [[8,15]]*perturb_nums,   # nodes for mantle
                    [3]*perturb_nums,          # km 
                    [2]*perturb_nums,     # km
                    [False]*perturb_nums,
                    np.random.randint(0,1000000,perturb_nums)
                    )

# step5: transform the vs model to vp model
vel_models = p_map(transform_vs_to_vel_model,list(vs_augmented),[depth]*len(vs_augmented))

# step6: calculate the dispersion curve
t = generate_mixed_samples(num_samples=300,start=1,end=100,uniform_num=100,log_num=100,random_num=100)
t = np.ones((len(vel_models),len(t)))*t
disp_data = p_map(calculate_dispersion, vel_models, list(t))
disp_data = np.array(disp_data)
vel_models = np.array(vel_models)

```
Details of this example can be found at:

| Dataset Name               | Reference                                                                                   | Link                                                                                       |
|----------------------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| OpenSWI-deep-example        | -                                                                                           | [OpenSWI-deep-example.ipynb](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/00_OpenSWI-deep-example.ipynb) |
| LITHO1.0                   | Pasyanos et al., 2014                                                                        | [LITHO1.0](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/12_LITHO1.ipynb) |
| USTClitho1.0               | Xin et al., 2018                                                                             | [USTClitho1.0](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/11_USTCLitho1.ipynb) |
| Central-and-Western US     | Shen et al., 2013                                                                            | [Central-and-Western US](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/13_Central_and_Western_US_Shen2013.ipynb) |
| Continental China          | Shen et al., 2016                                                                            | [Continental China](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/14_Continental_China_Shen2016.ipynb) |
| US Upper-Mantle            | Xie et al., 2018                                                                              | [US Upper-Mantle](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/03_US-upper-mantle.ipynb) |
| EUCrust                    | Lu et al., 2018                                                                               | [EUCrust](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/05_EUCrust.ipynb) |
| Alaska                     | Berg et al., 2020                                                                             | [Alaska](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/04_Alaska.ipynb) |
| CSEM-Europe                | Blom et al., 2020; Fichtner et al., 2018; Çubuk-Sabuncu et al., 2017                         | [CSEM-Europe](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/02_CSEM_Europe.ipynb) |
| CSEM-Eastmed               | Blom et al., 2020; Fichtner et al., 2018                                                     | [CSEM-Eastmed](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/01_CSEM_Eastmed.ipynb) |
| CSEM-Iberian               | Fichtner et al., 2018; Fichtner et al., 2015                                                 | [CSEM-Iberian](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/09_CSEM_lberia.ipynb) |
| CSEM-South Atlantic        | Fichtner et al., 2018; Colli et al., 2013                                                   | [CSEM-South Atlantic](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/07_CSEM_North_Atlantic.ipynb) |
| CSEM-North Atlantic        | Fichtner et al., 2018; Krischer et al., 2018                                                | [CSEM-North Atlantic](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/07_CSEM_North_Atlantic.ipynb) |
| CSEM-Japan                 | Fichtner et al., 2018; Simutė et al., 2016                                                 | [CSEM-Japan](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/08_CSEM_Japan.ipynb) |
| CSEM-Australasia           | Fichtner et al., 2018; Fichtner et al., 2010                                               | [CSEM-Australasia](../Datasets-Construction/OpenSWI-deep/1s-100s-Aug/10_CSEM_Australasia.ipynb) |

----