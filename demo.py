import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import linkage

# Load the data into a Pandas DataFrame, skipping the first row
data = pd.read_csv("Modern Ancestry.txt", index_col=0, header=None)

# Compute the linkage matrix using Ward's method
Z = linkage(data, method='ward')

# Get the branches of the dendrogram


def get_tree_branches(Z, data):
    n = len(data)
    branches = []
    for i in range(n-1):
        branch = []
        for j in range(2):
            if Z[i, j] < n:
                branch.append(data.index[int(Z[i, j])])
            else:
                branch += branches[int(Z[i, j] - n)]
        branches.append(branch)
    return branches


# Setting the default value of the cluster_index to 0.
if 'cluster_index' not in st.session_state:
    st.session_state['cluster_index'] = 0

selected_pop = st.selectbox("Select a population", data.index)
if selected_pop != st.session_state.get('selected_pop'):
    st.session_state['selected_pop'] = selected_pop
    st.session_state['cluster_index'] = 0

# Find the branches that contain the selected population
pop_branches = []
for branch in get_tree_branches(Z, data):
    if selected_pop in branch:
        pop_branches.append(branch)

# Display the selected population and its clusters
if pop_branches:
    cluster_index = st.session_state['cluster_index']
    branch_title = f"Cluster {cluster_index + 1}"
    with st.expander(branch_title, expanded=True):
        branch_text = ", ".join(pop_branches[cluster_index])
        st.text_area(branch_title, branch_text,
                     height=300, label_visibility="hidden")

# Move to the next and previous clusters
col1, col2 = st.columns(2)
with col1:
    if st.button("Previous cluster") and cluster_index > 0:
        st.session_state['cluster_index'] -= 1

with col2:
    if st.button("Next cluster") and cluster_index < len(pop_branches) - 1:
        st.session_state['cluster_index'] += 1

# Go back to the first cluster
if cluster_index > 0:
    if st.button("Go back to first cluster"):
        st.session_state['cluster_index'] = 0

# Writing the value of the cluster_index to the screen.
cluster_index = st.session_state['cluster_index']
st.write(cluster_index)
