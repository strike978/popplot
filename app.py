from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import streamlit as st
import pandas as pd
import io
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# Keep only these session state initializations
if 'textbox_content' not in st.session_state:
    st.session_state.textbox_content = ""
if 'textbox_history' not in st.session_state:
    st.session_state.textbox_history = []
if 'redo_history' not in st.session_state:
    st.session_state.redo_history = []
if 'decomposition_method' not in st.session_state:
    st.session_state.decomposition_method = "PCA"

# Keep only these constants:
LINKAGE_METHOD = "ward"
DISTANCE_METRIC = "euclidean"

# Setting the layout of the page to wide and the title of the page to PopPlot
st.set_page_config(layout="wide", page_title="PopPlot", page_icon="🌎")
st.subheader('Pop:green[Plot]')

# Define the available data files
data_files = {
    "Modern Period": "Modern Ancestry.txt",
    "Stone Age": "Mesolithic and Neolithic.txt",
    "Bronze Age": "Bronze Age.txt",
    "Iron Age": "Iron Age.txt",
    "Migration Period": "Migration Period.txt",
    "Medieval Period": "Middle Ages.txt",
}

# Define a function to read data from a file with UTF-8 encoding


def read_data_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


# Create a multiselect checkbox to choose data files
selected_files = st.multiselect("Historical Period:", list(
    data_files.keys()), default=["Modern Period"])

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
populations_in_textbox = []
for line in st.session_state.textbox_content.strip().split('\n'):
    if line:  # Only process non-empty lines
        parts = line.split(',')
        if len(parts) > 1:
            # Store the data part (everything after the first comma)
            data_part = ','.join(parts[1:])
            populations_in_textbox.append(data_part)

# Create a filtered list of available populations based on content after the comma
available_populations = []
for pop in selected_data:
    parts = pop.split(',')
    if len(parts) > 1:
        data_part = ','.join(parts[1:])
        if data_part not in populations_in_textbox:
            available_populations.append(pop)

group_pop_toggle = st.checkbox('Group by Population')

# Group populations based on the word after the first ":" when the toggle is enabled
grouped_populations = {}
if group_pop_toggle:
    # First pass: collect all groups in order
    population_options = []
    seen_groups = set()

    # First, populate groups with available populations only
    for pop in available_populations:  # Changed from selected_data to available_populations
        parts = pop.split(',')
        if len(parts) > 1:
            name_parts = parts[0].split(':')
            if len(name_parts) > 1:
                # Take only first two parts if they exist
                for group in name_parts[:-1][:2]:
                    group = group.strip()
                    if group not in seen_groups:
                        population_options.append(group)
                        seen_groups.add(group)
                        grouped_populations[group] = []
            else:
                # Handle single-part names
                group = name_parts[0].strip()
                if group not in seen_groups:
                    population_options.append(group)
                    seen_groups.add(group)
                    grouped_populations[group] = []

    # Second pass: populate the groups
    for pop in available_populations:
        parts = pop.split(',')
        if len(parts) > 1:
            name_parts = parts[0].split(':')
            if len(name_parts) > 1:
                # Add to first two groups if they exist
                for group in name_parts[:-1][:2]:
                    group = group.strip()
                    if group in grouped_populations:
                        grouped_populations[group].append(pop)
            else:
                # Handle single-part names
                group = name_parts[0].strip()
                if group in grouped_populations:
                    grouped_populations[group].append(pop)

    # Remove empty groups and update population_options
    non_empty_groups = [
        group for group in population_options if grouped_populations[group]]
    population_options = non_empty_groups
    grouped_populations = {k: v for k, v in grouped_populations.items() if v}

else:
    population_options = available_populations

# Initialize session state for indices and callback
if 'last_group_index' not in st.session_state:
    st.session_state.last_group_index = 0
if 'last_individual_index' not in st.session_state:
    st.session_state.last_individual_index = 0
if 'selectbox_changed' not in st.session_state:
    st.session_state.selectbox_changed = False

# Define callback function for selectbox


def on_selectbox_change():
    st.session_state.selectbox_changed = True
    current_index = population_options.index(
        st.session_state.population_selectbox)
    if group_pop_toggle:
        st.session_state.last_group_index = current_index
    else:
        st.session_state.last_individual_index = current_index


# Use the appropriate stored index based on toggle state
default_index = st.session_state.last_group_index if group_pop_toggle else st.session_state.last_individual_index
default_index = min(default_index, len(
    population_options) - 1) if population_options else 0

# Modify the selectbox to use callback
selected_option = st.selectbox(
    "Populations:",
    population_options,
    format_func=lambda x: x.split(',')[0] if not group_pop_toggle else x,
    key='population_selectbox',
    index=default_index,
    on_change=on_selectbox_change
)

# Modify the selection logic
if group_pop_toggle:
    selected_populations = grouped_populations[selected_option]
else:
    selected_populations = [selected_option]

# Preserve the previous state of the textbox content
if 'previous_textbox_content' not in st.session_state:
    st.session_state.previous_textbox_content = ""

# Create a row of buttons for Add and Clear
# Make the first column smaller since Clear button is smaller
col1, col2 = st.columns([1, 5])

with col1:
    if st.button("➕ Add"):
        if selected_populations:
            st.session_state.textbox_history.append(
                st.session_state.textbox_content)
            st.session_state.redo_history.clear()
            for pop in selected_populations:
                # Split by comma to separate population name from data
                parts = pop.split(',')
                if len(parts) > 1:
                    # Get the last part after colon for the population name
                    name_parts = parts[0].split(':')
                    display_name = name_parts[-1].strip()
                    # Get the data part
                    data_part = ','.join(parts[1:])
                    # Combine the display name with the data
                    formatted_pop = display_name + ',' + data_part

                    # Check if this data is already in the textbox
                    existing_data = [line.split(',', 1)[1] if ',' in line else ''
                                     for line in st.session_state.textbox_content.strip().split('\n')
                                     if line.strip()]

                    if data_part not in existing_data:
                        st.session_state.textbox_content += "\n" + formatted_pop.strip()

            # Get current index before calculating next
            current_index = population_options.index(selected_option)
            next_index = (current_index + 1) % len(population_options)
            if group_pop_toggle:
                st.session_state.last_group_index = next_index
            else:
                st.session_state.last_individual_index = next_index

            st.session_state.selectbox_changed = False
            st.rerun()

with col2:
    if st.button("🧹 Clear"):
        st.session_state.textbox_history.append(
            st.session_state.textbox_content)
        st.session_state.redo_history.clear()
        st.session_state.textbox_content = ""
        st.rerun()

# Commenting out Undo and Redo buttons
# with col3:
#     if st.button("↩️ Undo"):
#         if st.session_state.textbox_history:
#             st.session_state.redo_history.append(
#                 st.session_state.textbox_content)
#             st.session_state.textbox_content = st.session_state.textbox_history.pop()
#         st.rerun()

# with col4:
#     if st.button("↪️ Redo"):
#         if st.session_state.redo_history:
#             st.session_state.textbox_history.append(
#                 st.session_state.textbox_content)
#             st.session_state.textbox_content = st.session_state.redo_history.pop()
#         st.rerun()

# Add this function near the top of the file, after the imports


def parse_input_data(data_input):
    """Parse input data with mixed CSV/TSV formats."""
    parsed_lines = []
    for line in data_input.splitlines():
        if line.strip():
            # Count tabs and commas in the line
            tab_count = line.count('\t')
            comma_count = line.count(',')

            # Determine which delimiter splits the line into more fields
            if tab_count > comma_count:
                delimiter = '\t'
            else:
                delimiter = ','

            # Split the line and clean the fields
            fields = [field.strip() for field in line.split(delimiter)]
            parsed_lines.append(fields)

    # Convert to DataFrame
    if parsed_lines:
        df = pd.DataFrame(parsed_lines)
        # First column is population names, rest is data
        return df.iloc[:, 0], df.iloc[:, 1:].astype(float)
    return pd.Series(), pd.DataFrame()


# Display the Textbox with the entire selected options
data_input = st.text_area('Enter data in CSV or TSV format:',
                          st.session_state.textbox_content.strip(),
                          height=300,
                          key='textbox_input')

# Check if the Textbox content has changed manually and clear session state if it has
if data_input != st.session_state.textbox_content.strip():
    st.session_state.textbox_content = data_input.strip()
    st.rerun()

# Replace the decomposition method selector with this new organization

plot_type = st.radio(
    "Plot Type:",
    ["Tree Plot", "Scatter Plot"],
    horizontal=True,
    help="Choose between hierarchical clustering tree or dimensionality reduction scatter plot"
)

if plot_type == "Scatter Plot":
    method = st.radio(
        "Scatter Plot Method:",
        ["PCA", "t-SNE"],
        horizontal=True,
        help="""
        PCA: Principal Component Analysis - Standard linear dimensionality reduction
        t-SNE: t-Distributed Stochastic Neighbor Embedding - Best for visualizing clusters
        """
    )

    st.session_state.decomposition_method = method

# Update the plotting buttons
if plot_type == "Tree Plot":
    plot_button = st.button("🌳 Plot")
else:
    plot_button = st.button("📈 Plot")

# Simplify the model initialization section to only include these 5 methods
if plot_button and plot_type == "Scatter Plot":
    # Check for data first, before creating spinner
    if not data_input:
        st.warning(
            "Please add at least 3 populations before plotting.", icon="⚠️")
    else:
        with st.spinner(f"Creating {method} Plot..."):
            try:
                # Data preparation
                cleaned_data_input = "\n".join(
                    line.strip() for line in data_input.splitlines() if line.strip())

                # Parse mixed format data
                populations, data = parse_input_data(cleaned_data_input)

                # Check population count
                if not data.empty and len(populations) >= 3:
                    # Update model initialization based on selected method
                    if method == "PCA":
                        model = PCA(n_components=2)
                    else:  # t-SNE
                        n_samples = len(populations)
                        perplexity = 30.0 if n_samples > 30 else n_samples - 1
                        model = TSNE(
                            n_components=2,
                            perplexity=perplexity,
                            method='exact',  # For better accuracy with small datasets
                            learning_rate='auto'
                        )

                    # Perform dimensionality reduction
                    result = model.fit_transform(data.values)

                    # Create DataFrame for plotting
                    plot_df = pd.DataFrame(
                        data=result,
                        columns=[f'{method}1',
                                 f'{method}2']
                    )
                    plot_df['Populations'] = populations

                    # Create scatter plot
                    fig = px.scatter(
                        plot_df,
                        x=f'{method}1',
                        y=f'{method}2',
                        color='Populations',
                        title='',
                        text='Populations'
                    )

                    # Customize plot
                    fig.update_traces(
                        textposition='top center',
                        hovertemplate='%{text}'
                    )
                    fig.update_layout(
                        legend_title_text='Populations',
                        xaxis_title="",
                        yaxis_title=""
                    )

                    # Update method explanations
                    method_explanations = {
                        "PCA": "Principal Component Analysis finds the directions of maximum variance in the data.",
                        "t-SNE": "t-SNE visualizes genetic clusters by preserving local structure in the data."
                    }

                    st.caption(method_explanations[method])
                    st.plotly_chart(fig, use_container_width=True,
                                    config={'displayModeBar': True})

                else:
                    st.warning(
                        "Please add at least 3 populations before plotting.", icon="⚠️")

            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
                st.info("Try a different method or check your data.")

# Add back the tree plotting code
if plot_button and plot_type == "Tree Plot":
    with st.spinner("Creating Tree..."):
        if data_input:
            try:
                # Remove leading/trailing whitespace and empty lines
                cleaned_data_input = "\n".join(
                    line.strip() for line in data_input.splitlines() if line.strip())

                # Parse mixed format data
                populations, data = parse_input_data(cleaned_data_input)

                # Check if data is not empty and there are at least 3 populations
                if not data.empty and len(populations) >= 3:
                    labels = [i for i in populations]
                    height = max(20 * len(populations), 500)

                    # Convert data to float type to ensure numerical operations work
                    data_array = data.values

                    # Calculate distances using euclidean metric (required for Ward's method)
                    distances = pdist(data_array, metric=DISTANCE_METRIC)

                    # Create linkage matrix using Ward's method
                    linkage_matrix = linkage(
                        distances,
                        method=LINKAGE_METHOD
                    )

                    # Create dendrogram
                    fig = ff.create_dendrogram(
                        data_array,
                        orientation="right",
                        labels=labels,
                        distfun=lambda x: distances,
                        linkagefun=lambda x: linkage_matrix
                    )

                    # Update the layout and add captions
                    fig.update_layout(
                        height=height,
                        yaxis={'side': 'right'}
                    )
                    fig.update_yaxes(
                        automargin=True,
                        range=[0, len(populations)*10]
                    )

                    st.caption(
                        "Using Ward's method with Euclidean distance for optimal clustering")
                    st.caption(
                        'Close branches indicate recent common ancestors and highlight genetic mixing from migrations or conquests.')

                    st.plotly_chart(fig, theme=None, use_container_width=True, config={
                        'displayModeBar': True})
                else:
                    st.warning(
                        "Please add at least 3 populations before plotting.", icon="⚠️")
            except ValueError as e:
                st.error(f"Error creating dendrogram: {str(e)}")
                st.info(
                    "Try a different combination of clustering method and distance metric.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
        else:
            st.warning(
                "Please add at least 3 populations before plotting.", icon="⚠️")
