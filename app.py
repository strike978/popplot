from scipy.cluster.hierarchy import linkage, dendrogram
import streamlit as st
import pandas as pd
import io
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.decomposition import PCA


# Initialize session state attributes
if 'textbox_content' not in st.session_state:
    st.session_state.textbox_content = ""

# Setting the layout of the page to wide and the title of the page to PopPlot
st.set_page_config(layout="wide", page_title="PopPlot", page_icon="🌎")
st.header('Pop:green[Plot]')

# Define the available data files
data_files = {
    "Modern Era": "Modern Ancestry.txt",
    "Mesolithic and Neolithic": "Mesolithic and Neolithic.txt",
    "Bronze Age": "Bronze Age.txt",
    "Iron Age": "Iron Age.txt",
    "Migration Period": "Migration Period.txt",
    "Middle Ages": "Middle Ages.txt",
}

# Define a function to read data from a file with UTF-8 encoding


def read_data_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


# Create a multiselect checkbox to choose data files
selected_files = st.multiselect("Time Period:", list(
    data_files.keys()), default=["Modern Era"])

# Read data from selected files
selected_data = []
content_after_comma_set = set()
for file in selected_files:
    data = read_data_file(data_files[file])
    for line in data:
        content_after_comma = ",".join(line.split(',')[1:])
        if content_after_comma not in content_after_comma_set:
            selected_data.append(line)
            content_after_comma_set.add(content_after_comma)

# Get the populations already in the textbox
populations_in_textbox = [line.split(',')[1] if len(line.split(
    ',')) > 1 else '' for line in st.session_state.textbox_content.strip().split('\n')]

# Create a filtered list of available populations based on content after the comma
available_populations = [pop for pop in selected_data if pop.split(
    ',')[1] not in populations_in_textbox]

group_pop_toggle = st.checkbox('Group Populations')

# Group populations based on the word after the first ":" when the toggle is enabled
grouped_populations = {}
if group_pop_toggle:
    for pop in available_populations:
        parts = pop.split(',')
        if len(parts) > 1:
            # Extract the part after the first ":"
            key = parts[0].split(':')[1]
            if key not in grouped_populations:
                grouped_populations[key] = []
            grouped_populations[key].append(pop)

# Preserve the selected index in session state
if 'selected_option_index' not in st.session_state:
    st.session_state.selected_option_index = 0

# Create a Selectbox to display populations based on the toggle
if group_pop_toggle:
    population_options = list(grouped_populations.keys())
else:
    population_options = available_populations

# Ensure the selected index is within the valid range
if st.session_state.selected_option_index is None or st.session_state.selected_option_index >= len(population_options):
    st.session_state.selected_option_index = 0

selected_option_index = st.selectbox(
    "Populations:",
    range(len(population_options)),
    format_func=lambda i: population_options[i].split(',')[0],
    key='population_selectbox',
    index=st.session_state.selected_option_index
)

# Update the session state with the selected index
st.session_state.selected_option_index = selected_option_index

# Check if grouping of populations is enabled
if group_pop_toggle:
    # Check if a valid index is selected and within the range of population_options
    if selected_option_index is not None and selected_option_index < len(population_options):
        # If valid, select the corresponding grouped populations
        selected_option = grouped_populations[population_options[selected_option_index]]
    else:
        # If not valid, set selected_option as an empty list
        selected_option = []
else:
    # If grouping is not enabled, check if a valid index is selected
    if selected_option_index is not None:
        # If valid, select the corresponding population option as a list
        selected_option = [population_options[selected_option_index]]
    else:
        # If not valid, set selected_option as an empty list
        selected_option = []

col1, col2 = st.columns(2)

with col1:
    if st.button("➕ Add Population"):
        if selected_option:
            for pop in selected_option:
                # Remove all text before the last ":"
                parts = pop.split(':')
                if len(parts) > 1:
                    pop = parts[-1]
                if pop not in st.session_state.textbox_content:
                    st.session_state.textbox_content += "\n" + pop
            st.rerun()

with col2:
    if st.button("🧹 Clear Populations"):
        st.session_state.textbox_content = ""
        st.rerun()

# Display the Textbox with the entire selected options
data_input = st.text_area('Enter data in PCA coordinates format:',
                          st.session_state.textbox_content.strip(), height=300, key='textbox_input')

# Check if the Textbox content has changed manually and clear session state if it has
if data_input != st.session_state.textbox_content.strip():
    st.session_state.textbox_content = data_input.strip()
    st.rerun()

# Create tabs for different plots
tab1, tab2 = st.tabs(
    ["🌳 Tree", "📈 PCA"])

with tab1:
    with st.spinner("Creating Tree..."):
        if data_input:
            # Remove leading/trailing whitespace and empty lines
            cleaned_data_input = "\n".join(
                line.strip() for line in data_input.splitlines() if line.strip())

            # Read the data and select all columns except the first one (which contains population labels)
            data = pd.read_csv(io.StringIO(
                cleaned_data_input), header=None).iloc[:, 1:]
            populations = pd.read_csv(io.StringIO(
                cleaned_data_input), header=None, usecols=[0])[0]

            # Check if data is not empty and there are at least 3 populations
            if not data.empty and len(populations) >= 3:
                labels = [i for i in populations]
                height = max(20 * len(populations), 500)

                # Create the dendrogram using hierarchical clustering
                fig = ff.create_dendrogram(
                    data,
                    orientation="right",
                    labels=labels,
                    linkagefun=lambda x: linkage(x, method="ward"),
                )

                # Update the layout of the dendrogram
                fig.update_layout(
                    height=height,
                    yaxis={'side': 'right'}
                )
                fig.update_yaxes(
                    automargin=True,
                    range=[0, len(populations)*10]
                )

                # Add a caption and display the dendrogram
                st.caption(
                    'Close branches indicate recent common ancestors and highlight genetic mixing from migrations or conquests.')
                st.plotly_chart(fig, theme=None, use_container_width=True)
            else:
                st.info(
                    "Please add at least 3 populations before plotting.")
        else:
            st.info(
                "Please add at least 3 populations before plotting.")

with tab2:
    with st.spinner("Creating PCA Plot..."):
        if data_input:
            # Remove leading/trailing whitespace and empty lines
            cleaned_data_input = "\n".join(
                line.strip() for line in data_input.splitlines() if line.strip())

            # Read the data and select all columns except the first one (which contains population labels)
            data = pd.read_csv(io.StringIO(
                cleaned_data_input), header=None).iloc[:, 1:]

            populations = pd.read_csv(io.StringIO(
                cleaned_data_input), header=None, usecols=[0])[0]

            if not data.empty and len(populations) >= 3:
                # Perform PCA with all columns
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data)

                # Create a DataFrame for the PCA results
                pca_df = pd.DataFrame(
                    data=pca_result, columns=['PCA1', 'PCA2'])

                # Add the population labels back to the PCA DataFrame
                pca_df['Populations'] = populations

                # Create a 2D scatter plot with labels
                fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='Populations',
                                 title='', text='Populations')

                # Customize hover text to show only the label (population name)
                fig.update_traces(textposition='top center',
                                  hovertemplate='%{text}')

                # Change the legend title to "Populations"
                fig.update_layout(legend_title_text='Populations')
                # Remove the axis labels
                fig.update_xaxes(title_text='')
                fig.update_yaxes(title_text='')

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "Please add at least 3 populations before plotting.")
        else:
            st.info(
                "Please add at least 3 populations before plotting.")
