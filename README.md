# LEAP

## Large Language Models as Interpolated and Extrapolated Event Predictors

### Conda Environment

```python
conda create -n leap python=3.9
conda activate leap
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install dgl==1.0.1+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html
pip install -r requirements.txt
```

### Event Interpolation

#### Detailed comments describing implementation steps can be found in each shell executable (.sh) file. 

#### To run interpolated object prediction as a ranking task: 

```python
cd LEAP_OP1
sh Train_LEAP_OP1.sh
```

#### To run interpolated object prediction as a generative task: 

```python
cd LEAP_OP2
sh Train_LEAP_OP2.sh
```

### Event Extrapolation

#### To run extrapolated multi-event forecasting task: 

```python
cd LEAP_MEF
sh Train_LEAP_MEF.sh
```

### References

Inspired by [SeCoGD](https://github.com/yecchen/SeCoGD), we build LEAP_OP1.  
Inspired by [Glean](https://github.com/amy-deng/glean), we build LEAP_MEF.  
