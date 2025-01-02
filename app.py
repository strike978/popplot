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
if 'textbox_history' not in st.session_state:
    st.session_state.textbox_history = []
if 'redo_history' not in st.session_state:
    st.session_state.redo_history = []

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
col1, col2 = st.columns([1, 5])

with col1:
    plot_tree = st.button("🌳 Plot Tree")

with col2:
    plot_pca = st.button("📈 Plot PCA")

# Replace the tab1 content with tree button condition
if plot_tree:
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
                st.plotly_chart(fig, theme=None, use_container_width=True, config={
                    'displayModeBar': True})
            else:
                st.warning(
                    "Please add at least 3 populations before plotting.", icon="⚠️")
        else:
            st.warning(
                "Please add at least 3 populations before plotting.", icon="⚠️")

# Replace the tab2 content with PCA button condition
if plot_pca:
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

                st.plotly_chart(fig, use_container_width=True,
                                config={'displayModeBar': True})
            else:
                st.warning(
                    "Please add at least 3 populations before plotting.", icon="⚠️")
        else:
            st.warning(
                "Please add at least 3 populations before plotting.", icon="⚠️")
