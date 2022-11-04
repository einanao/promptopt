import json
import requests
import os
from collections import OrderedDict

import numpy as np
import torch

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder

import models
import datasets
import rlhf
import embed
import interrogator
import utils


embedding_model = embed.CLIP()

net_arch_kwargs = {"n_layers": 2, "layer_size": 256}
pref_model_train_config = rlhf.get_default_config()


def prefs_from_ranking(n):
    prefs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < 0.5:
                prefs.append((i, j, 0))
            else:
                prefs.append((j, i, 1))
    return prefs


def display_prompt(score, prompt):
    st.metric(label="score", value=np.round(float(score), decimals=2))
    st.code(prompt, language=None)


def display_losses(train_losses, val_losses):
    epochs = np.arange(0, len(train_losses), 1)
    data = np.stack([epochs, train_losses, val_losses], axis=1)
    cols = ["epochs", "training loss", "validation loss"]
    df = pd.DataFrame(data, columns=cols)
    st.line_chart(df, x="epochs")


def optimize_prompt(data):
    st.subheader("prompt scoring model training")
    with st.spinner("fitting prompt scoring model to ranking..."):
        prompts = data["prompts"]
        pref_data = data["prefs"]
        pref_model = models.PrefModel(
            embedding_model.n_embedding_dims, net_arch_kwargs=net_arch_kwargs
        )
        embeddings = embedding_model.embed_strings(prompts)
        embedding_dataset = datasets.EmbeddingDataset(embeddings=list(embeddings))
        pref_dataset = datasets.PrefDataset(embedding_dataset, pref_data=pref_data)
        optimizer = rlhf.RLHF(
            pref_model, pref_model_train_config, embedding_dataset, pref_dataset
        )
        _, train_losses, val_losses = optimizer.train()
        display_losses(train_losses, val_losses)
    st.subheader("prompt optimization")
    with st.spinner("searching for prompt that maximizes predicted score..."):
        candidate_scores = np.array(pref_model.score(embeddings))
        scored_candidates = list(zip(prompts, candidate_scores))
        init_prompt = max(scored_candidates, key=lambda x: x[1])[0]
        score_func = lambda x: torch.tensor(np.array(pref_model.score(x)))
        gator = interrogator.Gator(
            embedding_model=embedding_model,
            score_func=score_func,
            display=display_prompt,
        )
        best_prompt = gator.search(init_prompt)
    return best_prompt


# DEBUG
with open(os.path.join(utils.DATA_DIR, "generrated.json"), "r") as f:
    data = json.load(f)
prompts = data["prompts"][:10]
img_urls = data["img_urls"][:10]

d = OrderedDict(
    [
        ("img", img_urls),
        ("prompt", prompts),
        ("-", [None] * len(img_urls)),
        ("", [None] * len(img_urls)),
    ]
)
cols = list(d.keys())
df = pd.DataFrame(d)

with st.form(key="import-form") as form:
    imported_data = st.text_input("paste json of ranking here:")
    import_submit = st.form_submit_button("import ranking")

if import_submit:
    imported_data = json.loads(imported_data)
    df = pd.read_json(imported_data)

with st.form(key="my-form") as form:
    gb = GridOptionsBuilder.from_dataframe(df)

    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=False)
    gb.configure_side_bar()

    gridOptions = gb.build()
    gridOptions["columnDefs"][0]["rowDrag"] = True
    gridOptions["rowDragManaged"] = True
    for i in range(len(df.columns)):
        gridOptions["columnDefs"][i]["editable"] = True

    addCellRenderer = JsCode(
        """
            class BtnAddCellRenderer {
            init(params) {
                this.params = params;
                this.eGui = document.createElement('div');
                this.eGui.innerHTML = `
                 <span>
                    <style>
                    .btn_add {
                      background-color: limegreen;
                      border: none;
                      color: white;
                      text-align: center;
                      text-decoration: none;
                      display: inline-block;
                      font-size: 10px;
                      font-weight: bold;
                      height: 2.5em;
                      width: 4em;
                      cursor: pointer;
                    }

                    .btn_add :hover {
                      background-color: #05d588;
                    }
                    </style>
                    <button id='click-button'
                        class="btn_add"
                        >&CirclePlus;</button>
                 </span>
              `;
            }

            getGui() {
                return this.eGui;
            }

        };
            """
    )

    imgRenderer = JsCode(
        """
    function (params) {
            var element = document.createElement("span");
            var imageElement = document.createElement("img");

            imageElement.src = params.data.img;
            imageElement.width="80";
            imageElement.height="40";

            element.appendChild(imageElement);
            return element;
            };
        """
    )

    img_col = cols.index("img")
    gridOptions["columnDefs"][img_col]["cellRenderer"] = imgRenderer

    string_to_add_row = "\n\n function(e) { \n \
        let api = e.api; \n \
        let rowIndex = e.rowIndex + 1; \n \
        api.applyTransaction({addIndex: rowIndex, add: [{}]}); \n \
            }; \n \n"

    add_col = cols.index("")
    gridOptions["columnDefs"][add_col]["cellRenderer"] = addCellRenderer
    gridOptions["columnDefs"][add_col][
        "headerTooltip"
    ] = "Click on button to add new row"
    gridOptions["columnDefs"][add_col]["editable"] = False
    gridOptions["columnDefs"][add_col]["filter"] = True
    gridOptions["columnDefs"][add_col]["onCellClicked"] = JsCode(string_to_add_row)
    gridOptions["columnDefs"][add_col]["autoHeight"] = True
    gridOptions["columnDefs"][add_col]["wrapText"] = True

    string_to_delete = "\n\n function(e) { \n \
        let api = e.api; \n \
        let sel = api.getSelectedRows(); \n \
        api.applyTransaction({remove: sel}); \n \
        }; \n \n"

    cell_button_delete = JsCode(
        """
        class BtnCellRenderer {
            init(params) {
                this.params = params;
                this.eGui = document.createElement('div');
                this.eGui.innerHTML = `
                 <span>
                    <style>
                    .btn {
                      background-color: #F94721;
                      border: none;
                      color: white;
                      font-size: 10px;
                      font-weight: bold;
                      height: 2.5em;
                      width: 4em;
                      cursor: pointer;
                    }

                    .btn:hover {
                      background-color: #FB6747;
                    }
                    </style>
                    <button id='click-button'
                        class="btn"
                        >&#128465;</button>
                 </span>
              `;
            }

            getGui() {
                return this.eGui;
            }

        };
        """
    )

    del_col = cols.index("-")
    gridOptions["columnDefs"][del_col]["cellRenderer"] = cell_button_delete
    gridOptions["columnDefs"][del_col][
        "headerTooltip"
    ] = "Click on Button to remove row"
    gridOptions["columnDefs"][del_col]["editable"] = False
    gridOptions["columnDefs"][del_col]["onCellClicked"] = JsCode(string_to_delete)
    gridOptions["columnDefs"][del_col]["autoHeight"] = True
    gridOptions["columnDefs"][del_col]["suppressMovable"] = "true"

    st.subheader("prompt ranking")
    grid_return = AgGrid(
        df, editable=True, gridOptions=gridOptions, allow_unsafe_jscode=True
    )
    new_df = grid_return["data"]

    submit = st.form_submit_button("optimize prompt")


def get_table():
    return new_df[new_df["prompt"] != "nan"] # DEBUG


if submit:
    df = get_table()
    data = df.to_json()
    st.subheader("exported ranking (json)")
    st.code(json.dumps(data), language="json")
    prompts = df["prompt"].values.tolist()
    if len(prompts) >= 3:
        prefs = prefs_from_ranking(len(prompts))
        data = {"prompts": prompts, "prefs": prefs}
        opt_prompt = optimize_prompt(data)
    else:
        st.markdown("please rank at least 3 prompts")
