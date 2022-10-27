## setup
1. add `export PYTHONPATH=$PYTHONPATH:/path/to/promptopt` to the bottom of `~/.bash_profile` or `~/.bashrc`, then run `source ~/.bash_profile` or `source ~/.bashrc` or open a new shell
2. run `conda env create -f environment.yml`
3. run `conda activate promptopt`
4. run `cd app; python server.py` in one shell, and `cd app; streamlit run client.py` in another shell, then a browser window should open automatically

## linting
1. run `black promptopt`