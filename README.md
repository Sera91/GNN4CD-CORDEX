## GNN4CD-CORDEXML - branch Experiments_Valentina

This folder contains the code to reproduce the current "best runs" for ALPS.

In the following, the used configurations are summarized. All these configurations are
already specified in the config files or directly implemented in the code.

### 1. Precipitaion (pr)
**Model**:
 - GNN4CD_mod1 (with skip connections)

**High-features**:
- new orography (z-score standardization inside main.py)
- mask sea-land (0-1, used as it is)
- matrix coordinates ij (already normalized in [-1,1])

**Training parameters**
- loss: MSE_QMSE_PSD_Loss
  -> implemented as MSE + alpha*QMSE + beta*PSD, with parameters alpha=0.005 and beta=0.005
- n epochs = 150
- initial lr: 0.001
- lr scheduler: StepLR, step_size=25 epochs, gamma=0.5
- time lag: 2 days

### 2. Temperature (tasmax)
**Model**:
 - GNN4CD_mod1_GaussianNLL (with skip connections)

**High-features**:
- new orography (z-score standardization inside main.py)
- mask sea-land (0-1, used as it is)
- matrix coordinates ij (already normalized in [-1,1])

**Training parameters**
- loss: GaussianNLLLoss
- n epochs = 150
- initial lr: 0.001
- lr scheduler: StepLR, step_size=25 epochs, gamma=0.5
- time lag: 2 days
