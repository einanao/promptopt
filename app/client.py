import json
import requests
import os

import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder


from promptopt import utils

# DEBUG
with open(os.path.join(utils.DATA_DIR, "generrated.json"), "r") as f:
    data = json.load(f)
prompts = data["prompts"][:100]
img_urls = data["img_urls"][:100]


df = pd.DataFrame(
    {
        "add": [None] * len(img_urls),
        "rm": [None] * len(img_urls),
        "img_url": img_urls,
        "prompt": prompts,
    }
)

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
                  width: 8em;
                  cursor: pointer;
                }

                .btn_add :hover {
                  background-color: #05d588;
                }
                </style>
                <button id='click-button'
                    class="btn_add"
                    >&CirclePlus; Add</button>
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

        imageElement.src = params.data.img_url;
        imageElement.width="40";
        imageElement.height="40";

        element.appendChild(imageElement);
        return element;
        };
    """
)

gridOptions["columnDefs"][2]["cellRenderer"] = imgRenderer

string_to_add_row = "\n\n function(e) { \n \
    let api = e.api; \n \
    let rowIndex = e.rowIndex + 1; \n \
    api.applyTransaction({addIndex: rowIndex, add: [{}]}); \n \
        }; \n \n"

gridOptions["columnDefs"][0]["cellRenderer"] = addCellRenderer
gridOptions["columnDefs"][0]["headerTooltip"] = "Click on Button to add new row"
gridOptions["columnDefs"][0]["editable"] = False
gridOptions["columnDefs"][0]["filter"] = True
gridOptions["columnDefs"][0]["onCellClicked"] = JsCode(string_to_add_row)
gridOptions["columnDefs"][0]["autoHeight"] = True
gridOptions["columnDefs"][0]["wrapText"] = True
gridOptions["columnDefs"][0]["lockPosition"] = "left"

string_to_delete = "\n\n function(e) { \n \
    let api = e.api; \n \
    let sel = api.getSelectedRows(); \n \
    api.applyTransaction({remove: sel}); \n \
    }; \n \n"

cell_button_delete = JsCode(
    """
    class BtnCellRenderer {
        init(params) {
            console.log(params.api.getSelectedRows());
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
                  width: 8em;
                  cursor: pointer;
                }

                .btn:hover {
                  background-color: #FB6747;
                }
                </style>
                <button id='click-button'
                    class="btn"
                    >&#128465; Delete</button>
             </span>
          `;
        }

        getGui() {
            return this.eGui;
        }

    };
    """
)

gridOptions["columnDefs"][1]["cellRenderer"] = cell_button_delete
gridOptions["columnDefs"][1]["headerTooltip"] = "Click on Button to remove row"
gridOptions["columnDefs"][1]["editable"] = False
gridOptions["columnDefs"][1]["onCellClicked"] = JsCode(string_to_delete)
gridOptions["columnDefs"][1]["autoHeight"] = True
gridOptions["columnDefs"][1]["suppressMovable"] = "true"

grid_return = AgGrid(
    df, editable=True, gridOptions=gridOptions, allow_unsafe_jscode=True
)
new_df = grid_return["data"]

form = st.form(key="my-form")
submit = form.form_submit_button("Submit")

if submit:
    prompts = new_df["prompt"].values.tolist()
    prompts = [p for p in prompts if p != "nan"]  # DEBUG
    if len(prompts) >= 3:
        data = {"prompts": prompts}
        r = requests.post(
            "http://localhost:8000/api/opt_from_ranking",
            data={"data": json.dumps(data)},
        )
        opt_prompt = r.json()["optimized_prompt"]
        st.markdown("Optimized prompt: %s" % opt_prompt)
    else:
        st.markdown("Please enter and rank at least 3 prompts")
