# intel-ai-seattle

## Training Tutorial 

Step 1: Install the dependencies to your Python enviornment (This repo was tested on Python3.11)

```console
pip install scikit-learn xgboost comet-ml
```

Step 2: [Sign-Up for a free Comet account](https://www.comet.com/signup?utm_source=intel_ai_dev_summit&utm_medium=github&utm_content=readme)

Step 3: Configure Your Comet Creditentials 

```console
comet login 
```

Step 4: Run the training scripts!

```console
python training_scripts/gb.py
```
```console
python training_scripts/adaboost.py
```
```console
python training_scripts/xg.py
```

## Inference Tutorial 

Step 1: Install the dependencies to your Python enviornment (This repo was tested on Python3.11)

```console
pip install torch torchvision scikit-learn openvino comet-ml
```

Step 2: [Sign-Up for a free Comet account](https://www.comet.com/signup?utm_source=intel_ai_dev_summit&utm_medium=github&utm_content=readme)

Step 3: Configure Your Comet Creditentials 

```console
comet login 
```

Step 4: Run the inference with native pytorch

```console
python inference_scripts/pt_inference.py
```

Step 4: Run the inference with OpenVino engine to see the speedup (Note need an Intel CPU for this)
```console
python inference_scripts/vino_inference.py
```
