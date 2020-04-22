This repository includes the scripts used in the manuscript "Understanding the transition from paroxysmal to persistent atrial fibrillation from micro-anatomical re-entry in a simple model" by Ciacci et al. A good fraction of the code is written in cython therefore the user must compile these scripts by using the file setup.py.
The code is structured as follows:

1) Dashboard scripts: these files are regular python scripts (i.e., .py) where the user can set up (i.e., select the parameters), launch and save the outcomes of a certain experiment. Scripts belonging to this group are
1.a) main.py
1.b) main_MF.py
1.c) main_video.py
1.d) main_video_long.py

2) Model scripts: these files are cython scripts (i.e., pyx) containing classes and methods that are used by dashboard scripts to initiate and simulate the models presented in this study. Scripts belonging to this group are
2.a) models.pyx
2.b) models_long_experiment.pyx
2.c) models_video.pyx
2.d) models_video_long.pyx
2.e) MF_models.pyx

3) A method script: this file is a cython script (i.e., pyx) which includes various utilities methods that are used by other files.
Scripts belonging to this group are
3.a) methods.pyx

4) A setup script: this file is a regular python script (i.e., .py) that is exclusively used to compile the cython scripts. Scripts belonging to this group are
4.a) setup.py

5) Analysis scripts: these files are regular python scripts (i.e., .py) where the user can analyze the outcomes of various experiments and produce related figures. Scripts belonging to this group are
5.a) time_series_analysis.py
5.b) risk_curves.py
5.c) risk_curves_induction.py


 
