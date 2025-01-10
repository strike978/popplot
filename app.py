from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import streamlit as st
import pandas as pd
import io
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.decomposition import PCA, FastICA
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

# Preserve the selected index in session state
if 'selected_option_index' not in st.session_state:
    st.session_state.selected_option_index = 0

# Ensure the selected index is within the valid range
if st.session_state.selected_option_index is None or st.session_state.selected_option_index >= len(population_options):
    st.session_state.selected_option_index = 0

selected_option_index = st.selectbox(
    "Populations:",
    range(len(population_options)),
    format_func=lambda i: population_options[i] if group_pop_toggle else population_options[i].split(',')[
        0],
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

# Preserve the previous state of the textbox content
if 'previous_textbox_content' not in st.session_state:
    st.session_state.previous_textbox_content = ""

# Create a row of buttons for Add and Clear
# Make the first column smaller since Clear button is smaller
col1, col2 = st.columns([1, 5])

with col1:
    if st.button("➕ Add"):
        if selected_option:
            st.session_state.textbox_history.append(
                st.session_state.textbox_content)
            st.session_state.redo_history.clear()
            for pop in selected_option:
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

# Display the Textbox with the entire selected options
data_input = st.text_area('Enter data in CSV format:',
                          st.session_state.textbox_content.strip(), height=300, key='textbox_input')

# Check if the Textbox content has changed manually and clear session state if it has
if data_input != st.session_state.textbox_content.strip():
    st.session_state.textbox_content = data_input.strip()
    st.rerun()

# Add buttons for plotting
col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    plot_tree = st.button("🌳 Plot Tree")

with col2:
    plot_scatter = st.button("📈 Plot Scatter")

with col3:
    # Keep only the decomposition method selector for scatter plots
    decomposition_method = st.selectbox(
        "Dimensionality Reduction:",
        [
            "PCA",
            "t-SNE (Optimized)",
            "t-SNE (Classic)",
            "ICA"
        ],
        help="""
        PCA: Principal Component Analysis - Standard linear dimensionality reduction
        t-SNE (Optimized): Barnes-Hut t-SNE implementation - efficient for larger datasets
        t-SNE (Classic): Original t-SNE implementation - best precision for smaller datasets
        ICA: Independent Component Analysis - For finding independent patterns
        """,
        key='decomposition_method'
    )

# Modify the tree plotting section to include error handling
if plot_tree:
    with st.spinner("Creating Tree..."):
        if data_input:
            try:
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

                    # Convert data to float type to ensure numerical operations work
                    data_array = data.astype(float).values

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

# Replace the tab2 content with PCA button condition
if plot_scatter:
    with st.spinner(f"Creating {decomposition_method} Plot..."):
        if data_input:
            try:
                # Data preparation (same as before)
                cleaned_data_input = "\n".join(
                    line.strip() for line in data_input.splitlines() if line.strip())
                data = pd.read_csv(io.StringIO(
                    cleaned_data_input), header=None).iloc[:, 1:]
                populations = pd.read_csv(io.StringIO(
                    cleaned_data_input), header=None, usecols=[0])[0]

                if not data.empty and len(populations) >= 3:
                    # Update model initialization
                    if decomposition_method == "PCA":
                        model = PCA(n_components=2, random_state=42)
                    elif decomposition_method == "t-SNE (Optimized)":
                        model = TSNE(
                            n_components=2,
                            method='barnes_hut',
                            random_state=42,
                            perplexity=min(30, len(populations)-1),
                            n_jobs=-1
                        )
                    elif decomposition_method == "t-SNE (Classic)":
                        model = TSNE(
                            n_components=2,
                            method='exact',
                            random_state=42,
                            perplexity=min(30, len(populations)-1),
                            n_jobs=-1
                        )
                    else:  # ICA
                        model = FastICA(
                            n_components=2,
                            random_state=42,
                            max_iter=1000,
                            tol=0.01
                        )

                    # Perform dimensionality reduction
                    result = model.fit_transform(data.astype(float).values)

                    # Create DataFrame for plotting
                    plot_df = pd.DataFrame(
                        data=result,
                        columns=[f'{decomposition_method}1',
                                 f'{decomposition_method}2']
                    )
                    plot_df['Populations'] = populations

                    # Create scatter plot
                    fig = px.scatter(
                        plot_df,
                        x=f'{decomposition_method}1',
                        y=f'{decomposition_method}2',
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

                    # Add method-specific explanations
                    method_explanations = {
                        "PCA": "Principal Component Analysis finds the directions of maximum variance in the data.",
                        "t-SNE (Optimized)": "Efficient t-SNE implementation that preserves data structure using Barnes-Hut algorithm.",
                        "t-SNE (Classic)": "Classic t-SNE implementation that provides maximum precision.",
                        "ICA": "Independent Component Analysis separates independent genetic components."
                    }

                    st.caption(method_explanations[decomposition_method])
                    st.plotly_chart(fig, use_container_width=True,
                                    config={'displayModeBar': True})

            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
                st.info("Try a different decomposition method or check your data.")
        else:
            st.warning(
                "Please add at least 3 populations before plotting.", icon="⚠️")
