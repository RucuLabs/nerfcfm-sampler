# nerfcfm-sampler
NeRF sampler for the NeRFCFM project.

## Important
This is a basic implementation for the NerFCFM Sampler.

### Instructions
 On a Linux Machine

 1. [Optional]Make sure to [install Nerfstudio using conda](https://docs.nerf.studio/quickstart/installation.html)

 2. Clone this repository

 3. Setup
    ```bash
    # virtual environment setup
    virtualenv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

 5. Modify final line with video data and metodo_descriptor (histograma_por_zona, vector_de_intensidades, hog, omd)

 4. python sampling.py