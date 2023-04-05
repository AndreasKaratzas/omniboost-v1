# %%

import sys
sys.path.append('../../')

from envs.render import Render
from common.space import dnn_factory
from lib.estimator.utils import get_embeddings

# %%
embeddings = get_embeddings("../../data/demo")

# %%
names = ['LITTLE', 'GPU', 'BIG']

# %%
ui = Render(names=names, dnns=dnn_factory(), emb_dim=embeddings.shape[1])

# %%
ui.render(
    cached_workload=embeddings, 
    export_path='./', 
    epochs=999, 
    verbose=1
)

# %%
ui.close()

# %%
