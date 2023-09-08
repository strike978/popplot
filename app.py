import pandas as pd
from scipy.cluster.hierarchy import linkage
import streamlit as st
import io
import plotly.figure_factory as ff

# Setting the layout of the page to wide and the title of the page to G25 Dendrograms
st.set_page_config(layout="wide", page_title="PopPlot", page_icon="🧬")
st.header('Pop:green[Plot]')

# Creating a file uploader to upload data as CSV or text
uploaded_file = st.file_uploader(
    "Upload a CSV or text file", type=["csv", "txt"])

# Reading the data from the file pops.csv and displaying it in the text area.
default_data = open("EUROPE.txt", "r").read()
if uploaded_file is not None:
    default_data = uploaded_file.getvalue().decode('utf-8')
data_input = st.text_area("Enter data in CSV format", value=default_data)

# Reading the data from the file pops.csv and displaying it in the text area.
if data_input:
    data = pd.read_csv(io.StringIO(data_input), header=None).iloc[:, 1:]
    populations = pd.read_csv(io.StringIO(
        data_input), header=None, usecols=[0])[0]

# if st.button("Plot"):
with st.spinner("Creating Dendrogram..."):
    labels = [i for i in populations]
    height = max(20 * len(populations), 500)
    fig = ff.create_dendrogram(
        data,
        orientation="right",
        labels=labels,
        linkagefun=lambda x: linkage(x, method="ward"),
        # color_threshold=0.07,
    )
    fig.update_layout(
        height=height,
        yaxis={'side': 'right'}
    )
    fig.update_yaxes(
        automargin=True,
        range=[0, len(populations)*10]
    )

    st.plotly_chart(fig, theme=None, use_container_width=True)
